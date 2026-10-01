"""Small graph helpers shared by the environment and the game theory library."""

from collections.abc import Iterable
from itertools import combinations

from social_market_env.contracts import Pair


def as_pair(nodes: Iterable[int]) -> Pair:
    """Return two nodes as an ordered pair."""
    first, second = sorted(nodes)
    return first, second


def all_pairs(nodes: Iterable[int]) -> list[Pair]:
    """Return every unordered pair of distinct nodes in sorted order."""
    return [as_pair(pair) for pair in combinations(sorted(nodes), 2)]


def connected_components(
    *, nodes: Iterable[int], edges: Iterable[Pair]
) -> list[frozenset[int]]:
    """Split nodes into connected components using only edges between them."""
    node_set = set(nodes)
    adjacency: dict[int, set[int]] = {node: set() for node in node_set}
    for first, second in edges:
        if first in node_set and second in node_set:
            adjacency[first].add(second)
            adjacency[second].add(first)
    components: list[frozenset[int]] = []
    seen: set[int] = set()
    for start in sorted(node_set):
        if start not in seen:
            component = _collect_component(start=start, adjacency=adjacency)
            seen |= component
            components.append(component)
    return components


def is_connected(*, nodes: Iterable[int], edges: Iterable[Pair]) -> bool:
    """Return True when the nodes form exactly one connected component."""
    return len(connected_components(nodes=nodes, edges=edges)) == 1


def _collect_component(
    *, start: int, adjacency: dict[int, set[int]]
) -> frozenset[int]:
    found = {start}
    stack = [start]
    while stack:
        fresh = adjacency[stack.pop()] - found
        found |= fresh
        stack.extend(fresh)
    return frozenset(found)
