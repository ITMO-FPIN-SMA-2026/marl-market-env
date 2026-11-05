"""Доли котла на игре запланированных цен.

Здесь два отличия от baseline среды (план, раздел 3.8).

1. Игра считается при ценах плана, а не при цене картеля, иначе доли не соответствуют
   реальному котлу. Равновесия для этого не нужны: ценность подмножества — один расчёт рынка.
2. Подмножества кодируются битовыми масками, и вся таблица считается векторно. Майерсон
   для 12 игроков занимает миллисекунды, а не 2 с, как в gametheory.payouts.

Этап 1: Шепли и Майерсон. Этап 2: Оуэн и гибрид (проекция на ядро и страховка от штрафов).
"""

from collections.abc import Iterable
from math import factorial

import numpy as np

from social_market_env.config import MarketConfig
from social_market_env.contracts import Pair
from social_market_env.core.market import share_vector
from social_market_env.gametheory.payouts import to_weights

CORE_ITERATIONS: int = 200
CORE_TOLERANCE: float = 1e-6


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
    inside = _membership(len(players))
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
    sizes = _popcounts(values.size)
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


def owen_table(values: np.ndarray, *, blocks: list[int]) -> np.ndarray:
    """Вернуть вектор Оуэна; blocks — маски априорных союзов, вместе покрывающие всех игроков.

    Сначала ценность делится между союзами как между игроками, затем внутри союза: случайный
    порядок сохраняет участников одного союза рядом. Именно это нужно для коалиции, собранной
    из исторических суб-коалиций.
    """
    count = values.size.bit_length() - 1
    result = np.zeros(count)
    total_blocks = len(blocks)
    outer_weight = np.array([factorial(s) * factorial(total_blocks - s - 1) / factorial(total_blocks)
                             for s in range(total_blocks)])
    for block in blocks:
        others = [other for other in blocks if other != block]
        outer_masks, outer_sizes = _unions_of(others)
        for player in range(count):
            bit = 1 << player
            if not block & bit:
                continue
            inner_masks = np.array(_submasks(block ^ bit))
            inner_sizes = np.array([bin(int(mask)).count('1') for mask in inner_masks])
            size = bin(block).count('1')
            inner_weight = np.array([factorial(s) * factorial(size - s - 1) / factorial(size) for s in range(size)])
            base = outer_masks[:, None] | inner_masks[None, :]
            gains = values[base | bit] - values[base]
            result[player] = outer_weight[outer_sizes] @ gains @ inner_weight[inner_sizes]
    return result


def project_to_core(allocation: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Сдвинуть дележ к ядру: на каждом шаге закрыть самую сильную претензию подмножества.

    Шепли может не лежать в ядре (раздел 3.5 плана): {1, 3} из Шепли.docx стоит 7, а получает
    6.83. Если ядро пусто, вернётся лучший из найденных вариантов. Сумма выплат не меняется.
    """
    inside = _membership(allocation.size)
    current = allocation.astype(float).copy()
    best, best_excess = current.copy(), core_excess(current, values)
    middle = slice(1, values.size - 1)
    for _ in range(CORE_ITERATIONS):
        excess = values[middle] - inside[middle] @ current
        index = int(np.argmax(excess))
        if excess[index] <= CORE_TOLERANCE:
            return current
        group = inside[middle][index]
        current[group] += excess[index] / group.sum()
        current[~group] -= excess[index] / (~group).sum()
        worst = core_excess(current, values)
        if worst < best_excess:
            best, best_excess = current.copy(), worst
    return best


def core_excess(allocation: np.ndarray, values: np.ndarray) -> float:
    """Вернуть наибольшую претензию подмножества: 0 или меньше значит, что дележ в ядре."""
    if values.size <= 2:
        return 0.0
    inside = _membership(allocation.size)
    return float((values[1:-1] - inside[1:-1] @ allocation).max())


def payout_vector(
    *, scheme: str, players: list[int], values: np.ndarray, edges: Iterable[Pair], blocks: list[int] | None = None
) -> np.ndarray:
    """Вернуть выплаты по схеме shapley, myerson, owen или hybrid."""
    if scheme == 'shapley':
        return shapley_table(values)
    if scheme == 'owen':
        return owen_table(values, blocks=blocks or [1 << index for index in range(len(players))])
    game = restrict_to_graph(values, players=players, edges=edges)
    vector = shapley_table(game)
    return project_to_core(vector, game) if scheme == 'hybrid' else vector


def payout_weights(
    *,
    scheme: str,
    players: list[int],
    values: np.ndarray,
    edges: Iterable[Pair],
    blocks: list[int] | None = None,
    bonus: dict[int, float] | None = None,
) -> dict[int, float]:
    """Вернуть веса котла: неотрицательные и в сумме 1. bonus — надбавки страховки к выплатам."""
    vector = payout_vector(scheme=scheme, players=players, values=values, edges=edges, blocks=blocks)
    extra = bonus or {}
    return to_weights({player: float(value) + extra.get(player, 0.0) for player, value in zip(players, vector)})


def _membership(count: int) -> np.ndarray:
    """Вернуть матрицу: строка — маска, inside[m, k] истинно, если игрок k входит в подмножество m."""
    return (np.arange(1 << count)[:, None] >> np.arange(count)) & 1 == 1


def _popcounts(size: int) -> np.ndarray:
    return np.array([bin(mask).count('1') for mask in range(size)])


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


def _submasks(mask: int) -> list[int]:
    """Вернуть все подмножества маски, включая пустое."""
    found = [0]
    sub = mask
    while sub:
        found.append(sub)
        sub = (sub - 1) & mask
    return found


def _unions_of(blocks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть маски всех объединений союзов и число союзов в каждом."""
    masks = [0]
    sizes = [0]
    for block in blocks:
        masks += [mask | block for mask in masks]
        sizes += [size + 1 for size in sizes]
    return np.array(masks), np.array(sizes)
