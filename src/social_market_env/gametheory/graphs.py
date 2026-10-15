"""Small graph helpers shared by the environment and the game theory library."""

from collections.abc import Iterable
from itertools import combinations

import networkx as nx

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
    graph = nx.Graph()
    graph.add_nodes_from(sorted(set(nodes)))
    graph.add_edges_from((first, second) for first, second in edges if first in graph and second in graph)
    return [frozenset(part) for part in sorted(nx.connected_components(graph), key=min)]


def is_connected(*, nodes: Iterable[int], edges: Iterable[Pair]) -> bool:
    """Return True when the nodes form exactly one connected component."""
    return len(connected_components(nodes=nodes, edges=edges)) == 1


def graph_matrices(
    *, nodes: Iterable[int], edges: Iterable[Pair], coalitions: Iterable[Iterable[int]] = ()
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Myerson graph inputs: adjacency, reachability and actual coalition membership.

    All matrices use the same stable node order, including isolated agents.
    """
    ordered = tuple(sorted(set(nodes)))
    graph = nx.Graph()
    graph.add_nodes_from(ordered)
    graph.add_edges_from((first, second) for first, second in edges if first in graph and second in graph)
    components = {
        node: index for index, part in enumerate(nx.connected_components(graph)) for node in part
    }
    groups = {
        node: index for index, members in enumerate(coalitions) for node in members if node in graph
    }
    return {
        'adjacency': tuple(tuple(int(first != second and graph.has_edge(first, second)) for second in ordered)
                           for first in ordered),
        'reachability': tuple(tuple(int(components[first] == components[second]) for second in ordered)
                              for first in ordered),
        'coalition_membership': tuple(
            tuple(int(first in groups and groups[first] == groups.get(second)) for second in ordered)
            for first in ordered
        ),
    }
