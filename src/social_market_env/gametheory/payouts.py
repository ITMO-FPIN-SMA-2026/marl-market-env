"""Ways to split a coalition's value between its members."""

from collections.abc import Callable, Iterable, Sequence
from itertools import combinations
from math import factorial

from social_market_env.contracts import Pair
from social_market_env.gametheory.graphs import connected_components

ValueFunction = Callable[[frozenset[int]], float]


def equal_split(members: Iterable[int]) -> dict[int, float]:
    """Give every member the same share."""
    players = sorted(members)
    return {player: 1.0 / len(players) for player in players}


def to_weights(values: dict[int, float]) -> dict[int, float]:
    """Turn payout values into pool weights: negatives clipped, sum 1, equal split if
    nothing is positive."""
    clipped = {player: max(0.0, value) for player, value in values.items()}
    total = sum(clipped.values())
    if total <= 0:
        return equal_split(values)
    return {player: value / total for player, value in clipped.items()}


def coalition_surplus(
    *, members: frozenset[int], value: ValueFunction, expected_fines: float = 0.0
) -> float:
    """Return Delta(C) = v(C) - sum of standalone values - expected fines."""
    standalone = sum(value(frozenset({member})) for member in members)
    return value(members) - standalone - expected_fines


def shapley_values(*, members: Sequence[int], value: ValueFunction) -> dict[int, float]:
    """Return exact Shapley values by enumerating subsets."""
    players = sorted(members)
    result: dict[int, float] = {}
    for player in players:
        others = [other for other in players if other != player]
        result[player] = _inner_sum(
            player=player, inner=others, outer=frozenset(), value=value
        )
    return result


def graph_restricted(
    *, value: ValueFunction, edges: Iterable[Pair]
) -> ValueFunction:
    """Return the Myerson graph-restricted game: a subset is worth the sum over its
    connected parts."""
    edge_list = list(edges)

    def restricted(subset: frozenset[int]) -> float:
        return sum(
            value(component)
            for component in connected_components(nodes=subset, edges=edge_list)
        )

    return restricted


def myerson_values(
    *, members: Sequence[int], value: ValueFunction, edges: Iterable[Pair]
) -> dict[int, float]:
    """Return Myerson values: Shapley values of the graph-restricted game."""
    return shapley_values(members=members, value=graph_restricted(value=value, edges=edges))


def owen_values(
    *,
    members: Sequence[int],
    value: ValueFunction,
    unions: Sequence[frozenset[int]],
) -> dict[int, float]:
    """Return Owen values for a priori unions; members outside unions form singleton
    unions."""
    blocks = _complete_partition(members=members, unions=unions)
    result: dict[int, float] = {}
    for index, block in enumerate(blocks):
        other_blocks = [other for position, other in enumerate(blocks) if position != index]
        for player in sorted(block):
            result[player] = _owen_player(
                player=player, block=block, other_blocks=other_blocks, value=value
            )
    return result


def _owen_player(
    *,
    player: int,
    block: frozenset[int],
    other_blocks: list[frozenset[int]],
    value: ValueFunction,
) -> float:
    blocks_total = len(other_blocks) + 1
    inner = sorted(block - {player})
    total = 0.0
    for size in range(len(other_blocks) + 1):
        weight = (
            factorial(size)
            * factorial(blocks_total - size - 1)
            / factorial(blocks_total)
        )
        for chosen in combinations(other_blocks, size):
            outer = frozenset().union(*chosen)
            total += (
                weight
                * _inner_sum(player=player, inner=inner, outer=outer, value=value)
            )
    return total


def _inner_sum(
    *, player: int, inner: Sequence[int], outer: frozenset[int], value: ValueFunction
) -> float:
    players_total = len(inner) + 1
    total = 0.0
    for size in range(len(inner) + 1):
        weight = (
            factorial(size)
            * factorial(players_total - size - 1)
            / factorial(players_total)
        )
        for chosen in combinations(inner, size):
            coalition = outer | frozenset(chosen)
            total += weight * (value(coalition | {player}) - value(coalition))
    return total


def _complete_partition(
    *, members: Sequence[int], unions: Sequence[frozenset[int]]
) -> list[frozenset[int]]:
    member_set = set(members)
    blocks = [frozenset(union) & member_set for union in unions]
    blocks = [block for block in blocks if block]
    covered: set[int] = set()
    for block in blocks:
        if block & covered:
            raise ValueError('Owen unions must not overlap')
        covered |= block
    return blocks + [frozenset({member}) for member in sorted(member_set - covered)]
