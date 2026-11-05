"""Доли котла: быстрые Шепли и Майерсон на игре запланированных цен.

Этап 1 плана (раздел 3.8). Здесь два отличия от baseline среды.

1. Игра считается при ценах плана, а не при цене картеля, иначе доли не соответствуют
   реальному котлу. Равновесия для этого не нужны: ценность подмножества — один расчёт рынка.
2. Подмножества кодируются битовыми масками, и вся таблица считается векторно. Майерсон
   для 12 игроков занимает миллисекунды, а не 2 с, как в gametheory.payouts.

Этап 2 добавит Оуэна и гибрид: проекцию на ядро и страховку от штрафов.
"""

from collections.abc import Iterable
from math import factorial

import numpy as np

from social_market_env.config import MarketConfig
from social_market_env.contracts import Pair
from social_market_env.core.market import share_vector
from social_market_env.gametheory.payouts import to_weights


def level_game(
    *,
    players: list[int],
    prices: dict[int, float],
    nash_prices: dict[int, float],
    nash_profits: dict[int, float],
    costs: dict[int, float],
    market: MarketConfig,
) -> np.ndarray:
    """Вернуть таблицу v(S) по маскам: S держит план, остальные игроки плана уходят к Нэшу.

    Продавцы вне плана сохраняют свои цены. Снизу ценность ограничена суммой прибылей Нэша:
    подмножество всегда может отказаться от плана, а при высокой цене без этого его
    ценность была бы около нуля.
    """
    sellers = sorted(prices)
    columns = [sellers.index(player) for player in players]
    masks = np.arange(1 << len(players))
    # inside[m, k] истинно, если игрок k входит в подмножество m.
    inside = (masks[:, None] >> np.arange(len(players))) & 1 == 1
    matrix = np.tile(np.array([prices[seller] for seller in sellers]), (masks.size, 1))
    reverted = np.array([nash_prices[player] for player in players])
    matrix[:, columns] = np.where(inside, matrix[:, columns], reverted)
    # Те же формулы, что в clear_market, только сразу для всех подмножеств.
    quantity = np.maximum(0.0, market.demand_a - market.demand_b * matrix.mean(axis=1))
    margins = matrix[:, columns] - np.array([costs[player] for player in players])
    profits = quantity[:, None] * share_vector(prices=matrix, market=market)[:, columns] * margins
    floor = inside @ np.array([nash_profits[player] for player in players])
    return np.maximum((profits * inside).sum(axis=1), floor)


def shapley_table(values: np.ndarray) -> np.ndarray:
    """Вернуть вектор Шепли по таблице ценностей: формула из Шепли.docx на масках."""
    count = values.size.bit_length() - 1
    masks = np.arange(values.size)
    sizes = np.array([bin(mask).count('1') for mask in masks])
    weights = np.array([factorial(s) * factorial(count - s - 1) / factorial(count) for s in range(count)])
    result = np.zeros(count)
    for player in range(count):
        without = masks[(masks >> player) & 1 == 0]
        result[player] = weights[sizes[without]] @ (values[without | (1 << player)] - values[without])
    return result


def restrict_to_graph(values: np.ndarray, *, players: list[int], edges: Iterable[Pair]) -> np.ndarray:
    """Вернуть игру Майерсона: подмножество стоит сумму ценностей своих связных частей."""
    position = {player: index for index, player in enumerate(players)}
    neighbors = [0] * len(players)
    for first, second in edges:
        if first in position and second in position:
            neighbors[position[first]] |= 1 << position[second]
            neighbors[position[second]] |= 1 << position[first]
    restricted = np.zeros(values.size)
    # Маска без части всегда меньше самой маски, поэтому результат для неё уже посчитан.
    for mask in range(1, values.size):
        part = _component(mask=mask, neighbors=neighbors)
        restricted[mask] = values[part] + restricted[mask ^ part]
    return restricted


def payout_weights(*, scheme: str, players: list[int], values: np.ndarray, edges: Iterable[Pair]) -> dict[int, float]:
    """Вернуть веса котла по схеме shapley или myerson: неотрицательные и в сумме 1."""
    game = restrict_to_graph(values, players=players, edges=edges) if scheme == 'myerson' else values
    return to_weights({player: float(value) for player, value in zip(players, shapley_table(game))})


def _component(*, mask: int, neighbors: list[int]) -> int:
    """Вернуть маску связной части, в которую входит младший игрок из mask."""
    part = frontier = mask & -mask
    while frontier:
        lowest = frontier & -frontier
        frontier ^= lowest
        fresh = neighbors[lowest.bit_length() - 1] & mask & ~part
        part |= fresh
        frontier |= fresh
    return part
