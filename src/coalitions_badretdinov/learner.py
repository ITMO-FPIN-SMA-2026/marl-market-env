"""Контекстный бандит над стратегическим рычагом политики.

Этап 3 плана (раздел 4.6). Рычаг — верхний предел падения SW на блок из нескольких шагов.
Варианты совпадают с уровнями PRICE_LEVELS, поэтому у бандита есть честные контрольные версии:
каждый уровень, зафиксированный на весь эпизод. Если бандит их не обгоняет, он не нужен.

Метод — линейный Thompson Sampling: для каждого варианта своя байесовская линейная регрессия
награды по контексту. Обучение идёт только при training=True, при оценке выбор жадный.
Модель сохраняется в JSON: без pickle, как у агента Аскарова.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coalitions_badretdinov.settings import PRICE_LEVELS

ARMS: tuple[str, ...] = tuple(PRICE_LEVELS)
FEATURES: tuple[str, ...] = (
    'bias', 'density', 'largest_share', 'mean_history', 'compliance', 'link_cost', 'pressure', 'welfare_gap',
    'steps_left',
)
CHECKPOINT_VERSION: int = 1


@dataclass
class Block:
    """Незавершённый блок шагов: что выбрано и сколько набрано."""

    arm: int
    context: np.ndarray
    total: float = 0.0
    steps: int = 0


class LinearThompson:
    """Байесовская линейная регрессия на каждый вариант с выборкой из апостериорного."""

    def __init__(self, *, exploration: float = 0.5, ridge: float = 1.0) -> None:
        self.exploration = exploration
        size = len(FEATURES)
        self.precision = np.stack([ridge * np.eye(size) for _ in ARMS])
        self.moment = np.zeros((len(ARMS), size))
        self.updates = np.zeros(len(ARMS), dtype=int)
        self.rng = np.random.default_rng(0)

    def reseed(self, seed: int) -> None:
        """Задать случайность эпизода."""
        self.rng = np.random.default_rng(seed)

    def choose(self, context: np.ndarray, *, explore: bool) -> int:
        """Вернуть номер варианта: случайная выборка при обучении, жадный выбор при оценке."""
        scores = []
        for arm in range(len(ARMS)):
            covariance = np.linalg.inv(self.precision[arm])
            mean = covariance @ self.moment[arm]
            if explore:
                # Симметризация убирает ошибки округления, иначе NumPy предупреждает о несимметрии.
                mean = self.rng.multivariate_normal(mean, self.exploration ** 2 * (covariance + covariance.T) / 2)
            scores.append(float(context @ mean))
        return int(np.argmax(scores))

    def update(self, *, arm: int, context: np.ndarray, reward: float) -> None:
        """Учесть награду варианта в контексте."""
        self.precision[arm] += np.outer(context, context)
        self.moment[arm] += reward * context
        self.updates[arm] += 1

    def save(self, path: str | Path) -> None:
        """Сохранить модель в JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': CHECKPOINT_VERSION, 'arms': list(ARMS), 'features': list(FEATURES),
            'exploration': self.exploration, 'precision': self.precision.tolist(), 'moment': self.moment.tolist(),
            'updates': self.updates.tolist(),
        }
        destination.write_text(json.dumps(payload) + '\n', encoding='utf-8')

    def load(self, path: str | Path) -> None:
        """Загрузить модель и проверить, что она от той же постановки."""
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        expected = (CHECKPOINT_VERSION, list(ARMS), list(FEATURES))
        if (payload.get('version'), payload.get('arms'), payload.get('features')) != expected:
            raise ValueError('Checkpoint does not match current arms or features')
        self.exploration = payload['exploration']
        self.precision = np.array(payload['precision'])
        self.moment = np.array(payload['moment'])
        self.updates = np.array(payload['updates'])
