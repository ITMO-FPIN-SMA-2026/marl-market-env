"""Рекомендованные цены планов.

Главный вывод раздела 3.3 плана: цена картеля редко удерживается. Поэтому цена идёт от Нэша
к цене разбиения вдоль одного пути с долей lam от 0 до 1, и эта доля ограничена сверху:

    этап 1: потолком падения SW (mean_price_cap и cap_share);
    этап 2: лестницей PriceLadder, которая сама находит этот потолок,
            и ограничением на соблазн предать (largest_incentive_share).
"""

import math
from collections.abc import Callable

import numpy as np

from coalitions_badretdinov.rewards import StepOutcome
from coalitions_badretdinov.settings import PolicySettings
from social_market_env.config import MarketConfig
from social_market_env.core.market import clear_market
from social_market_env.gametheory.equilibrium import best_response_price

BISECTION_STEPS: int = 6


def mean_price_cap(*, target_welfare: float, gap: float, market: MarketConfig) -> float:
    """Вернуть среднюю цену, при которой SW ниже конкурентного на долю gap.

    SW = Q^2 / (2B) и Q = A - B * p, поэтому потолок считается в явном виде.
    """
    quantity = math.sqrt(max(0.0, 1.0 - gap) * 2.0 * market.demand_b * target_welfare)
    return (market.demand_a - quantity) / market.demand_b


def cap_share(*, nash: dict[int, float], target: dict[int, float], members: set[int], mean_cap: float) -> float:
    """Вернуть наибольшую долю пути к target, не поднимающую среднюю цену выше потолка."""
    mean_nash = sum(nash.values()) / len(nash)
    rise = sum(target[member] - nash[member] for member in members) / len(nash)
    # При tz_linear сговор снижает цену: потолок не мешает, и план идёт к цене разбиения целиком.
    return 1.0 if rise <= 0 else min(1.0, max(0.0, (mean_cap - mean_nash) / rise))


def path_prices(
    *, nash: dict[int, float], target: dict[int, float], members: set[int], share: float, market: MarketConfig
) -> dict[int, float]:
    """Вернуть цены всех продавцов: участники на доле share пути, остальные по Нэшу.

    Одна доля на всех сохраняет разницу цен между участниками: она идёт от издержек,
    переводит сбыт к дешёвым продавцам и немного затрудняет арбитру поиск одинаковых цен.
    """
    return {
        agent: min(market.p_max, max(market.p_min, nash[agent] + share * (target[agent] - nash[agent])))
        if agent in members else nash[agent]
        for agent in nash
    }


def temptation(*, agent: int, prices: dict[int, float], costs: dict[int, float], market: MarketConfig) -> float:
    """Вернуть T_i: прибыль агента, если он один ответит наилучшей ценой на цены остальных."""
    others = np.array([price for other, price in prices.items() if other != agent])
    deviated = {**prices, agent: best_response_price(cost=costs[agent], other_prices=others, market=market)}
    return clear_market(prices=deviated, costs=costs, market=market).gross_profits[agent]


def future_weight(*, patience: float, steps_left: int) -> float:
    """Вернуть сумму дисконтов оставшихся шагов.

    При бесконечном горизонте это delta / (1 - delta). Конечный горизонт важен: к концу эпизода
    будущего всё меньше, поэтому удерживаемая цена падает сама.
    """
    if patience >= 1.0:
        return float(max(0, steps_left))
    return patience * (1.0 - patience ** max(0, steps_left)) / (1.0 - patience)


def incentive_holds(
    *,
    prices: dict[int, float],
    groups: dict[int, dict[int, float]],
    future: dict[int, float],
    nash_profits: dict[int, float],
    costs: dict[int, float],
    market: MarketConfig,
) -> bool:
    """Проверить для всех участников: T_i - выплата <= будущий вес * (выплата - Нэш).

    groups — словарь plan -> веса участников. Левая часть — разовый выигрыш от предательства,
    правая — будущий доход, который предатель потеряет, вернувшись к Нэшу (раздел 4.4 плана).
    """
    gross = clear_market(prices=prices, costs=costs, market=market).gross_profits
    for weights in groups.values():
        pool = sum(gross[member] for member in weights)
        for member, weight in weights.items():
            payout = weight * pool
            if temptation(agent=member, prices=prices, costs=costs, market=market) - payout > (
                future[member] * (payout - nash_profits[member]) + 1e-6
            ):
                return False
    return True


def largest_incentive_share(*, upper: float, holds: Callable[[float], bool]) -> float:
    """Найти наибольшую долю пути в [0, upper], где ограничение выполнено.

    Соблазн растёт с ценой, поэтому годится бисекция. Монотонность не строгая, поэтому
    результат — хорошее приближение, а не точный максимум.
    """
    if upper <= 0 or holds(upper):
        return upper
    low, high = 0.0, upper
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2.0
        if holds(middle):
            low = middle
        else:
            high = middle
    return low


class PriceLadder:
    """Допустимое падение SW, которое подстраивается под агентов и арбитра.

    Рост и снижение мультипликативные: от 2 % до картеля надо пройти в 36 раз больше,
    прибавками это заняло бы слишком много шагов эпизода.

    Этап 2 заменяет фиксированный порог: лестница не знает порог арбитра, а находит его
    по реакции. Это нужно потому, что ТЗ запрещает настраиваться на одну неизменную политику.
    """

    def __init__(self, *, settings: PolicySettings) -> None:
        self.settings = settings
        self.ceiling = settings.welfare_gap
        self.gap = min(settings.start_gap, self.ceiling)

    def update(self, outcome: StepOutcome) -> None:
        """Сдвинуть уровень по исходу шага. Шаги без планов уровень не меняют.

        Снижает цену не само низкое соблюдение, а убыток: если соблюдавшие заработали меньше
        Нэша, план вреден. При частичном соблюдении и приросте выше Нэша цена просто держится.
        Первая версия снижала цену при любом низком соблюдении и проигрывала фиксированному
        уровню почти на всех агентах (журнал разработки, этап 2).
        """
        settings = self.settings
        compliance = outcome.compliance
        if compliance is None:
            return
        if outcome.arbiter_reacted:
            # Ниже стартового уровня потолок не опускается: разовая ложная реакция арбитра
            # не должна закрывать коалиции всю оставшуюся часть эпизода.
            remembered = settings.ceiling_margin * max(outcome.welfare_gap, 0.0)
            self.ceiling = min(self.ceiling, max(settings.start_gap, remembered))
            self._cut(min(self.gap, self.ceiling))
        elif outcome.reward < 0:
            self._cut(self.gap)
        else:
            full = compliance >= settings.compliance_floor
            growth = settings.gap_growth if full else settings.partial_growth
            self.gap = min(self.ceiling, self.gap * growth)

    def _cut(self, gap: float) -> None:
        # Снизу уровень ограничен: иначе подъём от нуля занял бы весь остаток эпизода.
        self.gap = max(self.settings.start_gap / 2.0, gap * self.settings.gap_cut)
