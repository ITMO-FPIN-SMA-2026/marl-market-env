"""Настройки политики коалиций.

Этап 1 плана. Все режимы экспериментов ТЗ задаются здесь, а не отдельным кодом
(план, раздел 6): один класс политики покрывает антагонизм, частные коалиции,
гранд-коалицию и адаптивный режим.
"""

from dataclasses import dataclass

STRUCTURES: frozenset[str] = frozenset({'antagonism', 'partial', 'grand', 'adaptive'})
PAYOUTS: frozenset[str] = frozenset({'equal', 'shapley', 'myerson'})

# Уровень цены — допустимое падение SW относительно конкурентного уровня (план, раздел 3.3).
# Уровень задаётся падением SW, а не числом цены: так он не зависит от параметров рынка.
# На этапе 3 уровень будет выбирать бандит, пока это фиксированная настройка.
PRICE_LEVELS: dict[str, float] = {
    'cautious': 0.05,
    'moderate': 0.07,
    'bold': 0.15,
    'cartel': 1.0,
}


@dataclass(frozen=True)
class PolicySettings:
    """Параметры одного режима политики."""

    structure: str = 'adaptive'
    """Какие коалиции разрешены: antagonism, partial, grand или adaptive."""
    payout: str = 'myerson'
    """Схема долей котла: equal, shapley или myerson."""
    price_level: str = 'moderate'
    """Название уровня цены из PRICE_LEVELS."""
    partial_share: float = 0.5
    """Для structure='partial': наибольший размер группы как доля активных агентов."""
    hysteresis: float = 0.1
    """Новое разбиение заменяет прошлое, только если выгоднее на эту долю.

    Постоянные партнёры копят H_ij, а частая смена состава эту память обнуляет.
    """
    max_packings: int = 32
    """Сколько разбиений перебирать при выборе из кандидатов."""
    max_evaluations: int = 3
    """Сколько лучших разбиений оценивать через равновесие: одно стоит 10-80 мс."""
    max_exact_players: int = 14
    """Предел точного счёта долей: таблица ценностей занимает 2^n элементов."""

    def __post_init__(self) -> None:
        if self.structure not in STRUCTURES:
            raise ValueError(f'structure must be one of {sorted(STRUCTURES)}')
        if self.payout not in PAYOUTS:
            raise ValueError(f'payout must be one of {sorted(PAYOUTS)}')
        if self.price_level not in PRICE_LEVELS:
            raise ValueError(f'price_level must be one of {sorted(PRICE_LEVELS)}')
        if not 0 < self.partial_share <= 1:
            raise ValueError('partial_share must be in (0, 1]')
        if self.hysteresis < 0:
            raise ValueError('hysteresis must be non-negative')
        if min(self.max_packings, self.max_evaluations, self.max_exact_players) < 1:
            raise ValueError('search limits must be positive')

    @property
    def welfare_gap(self) -> float:
        """Допустимое падение SW для выбранного уровня цены."""
        return PRICE_LEVELS[self.price_level]
