"""Рекомендованные цены планов.

Этап 1 плана: фиксированный уровень цены. Главный вывод раздела 3.3 плана — цена картеля
редко удерживается, поэтому цена идёт от Нэша к цене разбиения только до потолка падения SW.

Этап 2 заменит фиксированный потолок ограничением на соблазн предать и лестницей AIMD.
"""

import math

from social_market_env.config import MarketConfig


def mean_price_cap(*, target_welfare: float, gap: float, market: MarketConfig) -> float:
    """Вернуть среднюю цену, при которой SW ниже конкурентного на долю gap.

    SW = Q^2 / (2B) и Q = A - B * p, поэтому потолок считается в явном виде.
    """
    quantity = math.sqrt(max(0.0, 1.0 - gap) * 2.0 * market.demand_b * target_welfare)
    return (market.demand_a - quantity) / market.demand_b


def blended_prices(
    *, nash: dict[int, float], target: dict[int, float], members: set[int], mean_cap: float, market: MarketConfig
) -> dict[int, float]:
    """Вернуть цены всех продавцов: участники идут от Нэша к target, не поднимая среднюю выше потолка.

    Одна доля пути lam на всех сохраняет разницу цен между участниками: она идёт от издержек,
    переводит сбыт к дешёвым продавцам и немного затрудняет арбитру поиск одинаковых цен.
    """
    mean_nash = sum(nash.values()) / len(nash)
    rise = sum(target[member] - nash[member] for member in members) / len(nash)
    # При tz_linear сговор снижает цену: потолок не мешает, и план идёт к цене разбиения целиком.
    lam = 1.0 if rise <= 0 else min(1.0, max(0.0, (mean_cap - mean_nash) / rise))
    return {
        agent_id: _clip(nash[agent_id] + lam * (target[agent_id] - nash[agent_id]), market)
        if agent_id in members else nash[agent_id]
        for agent_id in nash
    }


def _clip(price: float, market: MarketConfig) -> float:
    return min(market.p_max, max(market.p_min, price))
