"""Настройки политики коалиций.

Все режимы экспериментов ТЗ задаются здесь, а не отдельным кодом (план, раздел 6):
один класс политики покрывает антагонизм, частные коалиции, гранд-коалицию и адаптивный режим.
"""

from dataclasses import dataclass

STRUCTURES: frozenset[str] = frozenset({'antagonism', 'partial', 'grand', 'adaptive'})
PAYOUTS: frozenset[str] = frozenset({'equal', 'shapley', 'myerson', 'owen', 'hybrid'})
PRICINGS: frozenset[str] = frozenset({'fixed', 'adaptive'})

# Уровень цены — допустимое падение SW относительно конкурентного уровня (план, раздел 3.3).
# Уровень задаётся падением SW, а не числом цены: так он не зависит от параметров рынка.
# При pricing='adaptive' это только верхняя граница: фактический уровень находит лестница.
PRICE_LEVELS: dict[str, float] = {
    'cautious': 0.05,
    'moderate': 0.07,
    'bold': 0.15,
    'cartel': 1.0,
}


@dataclass(frozen=True)
class PolicySettings:
    """Параметры одного режима политики."""

    # --- Этап 1: структура и деление ---
    structure: str = 'adaptive'
    """Какие коалиции разрешены: antagonism, partial, grand или adaptive."""
    payout: str = 'hybrid'
    """Схема долей котла: equal, shapley, myerson, owen или hybrid."""
    price_level: str = 'cartel'
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
    """Сколько лучших разбиений оценивать."""
    max_fresh_equilibria: int = 2
    """Сколько новых равновесий можно считать за шаг.

    Именно они дают пики времени act. Предел задан числом, а не временем: иначе результат
    зависел бы от загрузки машины и терялась бы детерминированность.
    """
    max_exact_players: int = 14
    """Предел точного счёта долей: таблица ценностей занимает 2^n элементов."""

    # --- Этап 2: цена ---
    pricing: str = 'adaptive'
    """fixed — постоянный уровень; adaptive — лестница и ограничение на соблазн."""
    horizon: int = 200
    """Длина эпизода. Её нет в наблюдении, передаётся через kwargs."""
    patience: float = 0.9
    """Базовая терпеливость агентов: множитель будущего дохода в ограничении на соблазн."""
    start_gap: float = 0.02
    """С какого падения SW лестница начинает эпизод."""
    gap_growth: float = 1.15
    """Рост уровня после чистого шага."""
    partial_growth: float = 1.1
    """Рост уровня, если соблюдающих мало, но они заработали выше Нэша."""
    gap_cut: float = 0.6
    """Снижение уровня после предательства или реакции арбитра."""
    ceiling_margin: float = 0.85
    """После реакции арбитра потолок ставится на эту долю от падения SW, вызвавшего реакцию."""
    compliance_floor: float = 0.7
    """Если соблюдающих меньше этой доли, шаг считается неудачным."""

    # --- Этап 2: память об агентах ---
    memory: bool = True
    """Учитывать ли доверие и исключения. Нужно для вопроса ТЗ об эффекте памяти."""
    trust_decay: float = 0.9
    """Затухание накопленных наблюдений за агентом."""
    trust_prior_good: float = 4.0
    """Априорное число соблюдений: стартовое доверие 0.8."""
    trust_prior_bad: float = 1.0
    """Априорное число нарушений."""
    ban_trust: float = 0.3
    """Демпингующий агент исключается, только если доверие к нему упало ниже этого уровня.

    Перебор показал: 0.5 снижает охват и прибыль, а 0.3 почти не отличается от полного отказа
    от исключений. Такое исключение срабатывает только на постоянных нарушителях.
    """
    ban_steps: int = 5
    """На сколько шагов исключается агент."""

    # --- Этап 2: оценка группы ---
    discount: float = 0.5
    """Вес будущего в оценке V(C).

    Намеренно меньше настоящего дисконта: потеря связности не конец коалиции, группа собирается
    заново. При 0.95 оценка слишком сильно предпочитала мелкие группы.
    """
    network_mode: str = 'evolving'
    """Режим сети: в static связи не рвутся. Его нет в наблюдении, передаётся через kwargs."""
    insurance_rate: float = 0.25
    """Какая доля невыплаченной компенсации за штраф идёт в веса на каждом шаге."""
    diagnostics: bool = False
    """Считать показатели для отчётов: они стоят времени и на решения не влияют."""

    # --- Этап 3: обучение ---
    learning: str = 'off'
    """off — потолок задан price_level; bandit — потолок выбирает контекстный бандит."""
    block_steps: int = 8
    """Сколько шагов держится выбор бандита: эффект цены проявляется не сразу."""
    exploration: float = 0.5
    """Масштаб случайности Thompson Sampling при обучении."""
    link_penalty: float = 3.0
    """Штраф в награде бандита за долю пар, подорожавших за шаг.

    Награда блока не видит будущего, а поднятая арбитром стоимость связи не затухает.
    Без штрафа бандит не отличил бы высокую прибыль сейчас от разрушения сети.
    """

    def __post_init__(self) -> None:
        choices = [(self.structure, STRUCTURES, 'structure'), (self.payout, PAYOUTS, 'payout'),
                   (self.price_level, PRICE_LEVELS, 'price_level'), (self.pricing, PRICINGS, 'pricing'),
                   (self.network_mode, {'static', 'evolving'}, 'network_mode')]
        for value, allowed, name in choices:
            if value not in allowed:
                raise ValueError(f'{name} must be one of {sorted(allowed)}')
        shares = [self.partial_share, self.patience, self.gap_cut, self.ceiling_margin, self.compliance_floor,
                  self.trust_decay, self.ban_trust, self.discount, self.insurance_rate]
        if not all(0 < value <= 1 for value in shares):
            raise ValueError('shares and rates must be in (0, 1]')
        if self.learning not in {'off', 'bandit'}:
            raise ValueError('learning must be off or bandit')
        if self.learning == 'bandit' and self.pricing != 'adaptive':
            raise ValueError('bandit needs adaptive pricing')
        if self.block_steps < 1 or self.exploration <= 0 or self.link_penalty < 0:
            raise ValueError('block_steps and exploration must be positive, link_penalty non-negative')
        if self.hysteresis < 0 or self.ban_steps < 0 or self.max_fresh_equilibria < 0:
            raise ValueError('hysteresis, ban_steps and max_fresh_equilibria must be non-negative')
        if min(self.max_packings, self.max_evaluations, self.max_exact_players, self.horizon) < 1:
            raise ValueError('search limits and horizon must be positive')
        if not 0 < self.start_gap <= 1 or self.gap_growth < 1 or self.partial_growth < 1:
            raise ValueError('start_gap must be in (0, 1] and growth rates at least 1')
        if self.trust_prior_good <= 0 or self.trust_prior_bad <= 0:
            raise ValueError('trust priors must be positive')

    @property
    def welfare_gap(self) -> float:
        """Допустимое падение SW для выбранного уровня цены."""
        return PRICE_LEVELS[self.price_level]
