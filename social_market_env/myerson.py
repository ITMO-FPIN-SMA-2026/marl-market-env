"""Graph-restricted coalition values and Myerson allocation."""

from __future__ import annotations

from itertools import combinations
from math import factorial
import random
from typing import Callable, Iterable, Optional

import networkx as nx


def graph_matrices(
    graph: nx.Graph, coalitions: Iterable[Iterable[int]], nodes: Iterable[int]
) -> dict[str, tuple[tuple[int, ...], ...]]:
    ordered = tuple(nodes)
    selected = graph.subgraph(ordered)
    component = {node: index for index, part in enumerate(nx.connected_components(selected)) for node in part}
    group = {node: index for index, members in enumerate(coalitions) for node in members}
    return {
        "adjacency": tuple(tuple(int(u != v and selected.has_edge(u, v)) for v in ordered) for u in ordered),
        "reachability": tuple(tuple(int(component[u] == component[v]) for v in ordered) for u in ordered),
        "coalition_membership": tuple(
            tuple(int(u in group and group[u] == group.get(v)) for v in ordered) for u in ordered
        ),
    }


def myerson_value(
    graph: nx.Graph,
    players: Iterable[int],
    worth: Callable[[frozenset[int]], float],
    *,
    max_exact_players: int = 12,
    samples: int = 2048,
    rng: Optional[random.Random] = None,
) -> dict[int, float]:
    """Use exact Shapley weights up to the threshold, then sampled permutations."""
    members = tuple(sorted(set(players)))
    if any(node not in graph for node in members):
        raise ValueError("all players must be graph nodes")
    n = len(members)
    if not n:
        return {}

    cache: dict[frozenset[int], float] = {frozenset(): 0.0}

    def restricted(coalition: frozenset[int]) -> float:
        if coalition not in cache:
            cache[coalition] = sum(
                float(worth(frozenset(component)))
                for component in nx.connected_components(graph.subgraph(coalition))
            )
        return cache[coalition]

    result = {player: 0.0 for player in members}
    if n > max_exact_players:
        if samples < 1:
            raise ValueError("samples must be positive for approximate Myerson allocation")
        generator = rng if rng is not None else random.Random(0)
        for _ in range(samples):
            ordering = list(members)
            generator.shuffle(ordering)
            previous = frozenset()
            previous_worth = 0.0
            for player in ordering:
                current = previous | {player}
                current_worth = restricted(current)
                result[player] += (current_worth - previous_worth) / samples
                previous, previous_worth = current, current_worth
        return result

    denominator = factorial(n)
    for player in members:
        others = tuple(x for x in members if x != player)
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / denominator
            for subset in combinations(others, size):
                before = frozenset(subset)
                result[player] += weight * (restricted(before | {player}) - restricted(before))
    return result
