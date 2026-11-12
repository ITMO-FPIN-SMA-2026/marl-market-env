"""Кандидаты в коалиции: связные группы агентов, которым можно сделать план.

Этап 1 плана. Условие среды: группа связна по рёбрам между её же участниками.
Поэтому все генераторы здесь возвращают только такие группы.

На этапе 2 сюда добавлено исключение нарушителей (параметр blocked).
"""

import math
from collections.abc import Iterable

import networkx as nx

from coalitions_badretdinov.settings import PolicySettings
from coalitions_badretdinov.value import Group
from social_market_env.contracts import CoalitionObservation, Pair
from social_market_env.gametheory.graphs import connected_components, is_connected


def eligible_agents(observation: CoalitionObservation) -> set[int]:
    """Вернуть агентов, которых среда примет в план."""
    return set(observation.active_agents) - set(observation.restricted)


def generate_candidates(
    *,
    observation: CoalitionObservation,
    previous: Iterable[Group],
    settings: PolicySettings,
    blocked: set[int] | frozenset[int] = frozenset(),
) -> list[Group]:
    """Вернуть кандидатов для выбранного режима структуры.

    blocked — временно исключённые агенты (этап 2). Группа считается заново без них, поэтому
    исключение точки сочленения может разделить бывшую группу на части.
    """
    eligible = eligible_agents(observation) - blocked
    components = _groups(connected_components(nodes=eligible, edges=observation.edges))
    if settings.structure == 'antagonism':
        return []
    if settings.structure == 'grand':
        # Гранд-коалиция — все активные. Без кого-то это уже другой режим (план, раздел 3.1).
        everyone = frozenset(observation.active_agents)
        return [everyone] if everyone in components else []
    kept = [group for group in previous if _still_valid(group=group, eligible=eligible, edges=observation.edges)]
    found = {*components, *kept}
    for component in components:
        found.update(_communities(component=component, edges=observation.edges))
    if settings.structure == 'partial':
        limit = max(2, math.floor(settings.partial_share * len(observation.active_agents)))
        found = {piece for group in found for piece in _split(group=group, limit=limit, edges=observation.edges)}
    return sorted(found, key=sorted)


def _groups(parts: Iterable[Group]) -> list[Group]:
    return [part for part in parts if len(part) >= 2]


def _still_valid(*, group: Group, eligible: set[int], edges: Iterable[Pair]) -> bool:
    return group <= eligible and is_connected(nodes=group, edges=edges)


def _member_graph(*, group: Group, edges: dict[Pair, float]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(sorted(group))
    # Вес ребра — память кооперации H_ij из [-1, 1]; сдвиг делает его неотрицательным.
    graph.add_weighted_edges_from(
        (first, second, 1.0 + weight) for (first, second), weight in edges.items() if {first, second} <= group
    )
    return graph


def _communities(*, component: Group, edges: dict[Pair, float]) -> list[Group]:
    """Разбить компоненту на плотные группы с учётом истории кооперации.

    Сообщества дают покрывающие разбиения, которые в разреженном графе часто лучше одной
    большой коалиции. Сообщество может оказаться несвязным, поэтому его делим на части.
    """
    if len(component) < 4:
        return []
    graph = _member_graph(group=component, edges=edges)
    found: list[Group] = []
    for community in nx.community.greedy_modularity_communities(graph, weight='weight'):
        found.extend(_groups(connected_components(nodes=community, edges=edges)))
    return found


def _split(*, group: Group, limit: int, edges: dict[Pair, float]) -> list[Group]:
    """Разрезать группу на связные куски не больше limit, идя обходом в ширину."""
    if len(group) <= limit:
        return [group]
    graph = _member_graph(group=group, edges=edges)
    pieces: list[Group] = []
    while graph.number_of_nodes() >= 2:
        # Начинаем с наименее связанного агента: так он реже остаётся один в конце.
        start = min(sorted(graph.nodes), key=graph.degree)
        piece = frozenset(list(nx.bfs_tree(graph, start))[:limit])
        pieces.append(piece)
        graph.remove_nodes_from(piece)
    return _groups(pieces)
