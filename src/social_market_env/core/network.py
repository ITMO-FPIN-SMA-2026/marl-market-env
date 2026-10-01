"""Hidden communication network: initial graph, per-pair memory, evolution and link costs."""

import math
from dataclasses import dataclass

import numpy as np

from social_market_env.config import NetworkConfig
from social_market_env.contracts import AgentAction, LinkActionKind, Pair
from social_market_env.gametheory.graphs import all_pairs, as_pair

POLYNOMIAL_TERMS: int = 10


@dataclass
class NetworkState:
    """Network state that only the environment and the coalition policy see."""

    agent_ids: tuple[int, ...]
    """All agents, including eliminated ones."""
    affinity: dict[Pair, float]
    """Standardized polynomial score of pair features z_ij."""
    edges: set[Pair]
    """Current links G_t."""
    history: dict[Pair, float]
    """Cooperation memory H_ij in [-1, 1]; shown as the link weight."""
    utility: dict[Pair, float]
    """Received utility memory U_ij in [-1, 1]."""
    link_costs: dict[Pair, float]
    """Arbiter-controlled cost c_ij of forming a link."""

    def edge_weights(self, *, active: set[int]) -> dict[Pair, float]:
        """Return links between active agents with their weights."""
        return {pair: self.history[pair] for pair in sorted(self.edges) if set(pair) <= active}

    def neighbor_weights(self, *, agent_id: int, active: set[int]) -> dict[int, float]:
        """Return active neighbors of an agent with link weights."""
        weights = self.edge_weights(active=active)
        return {
            other: weight
            for pair, weight in weights.items()
            if agent_id in pair
            for other in pair
            if other != agent_id
        }


@dataclass(frozen=True)
class LinkRequests:
    """Agents' link requests aggregated by pair."""

    removals: frozenset[Pair]
    """Pairs that at least one endpoint wants to drop."""
    bonuses: dict[Pair, int]
    """Number of endpoints asking to add or maintain each pair."""
    adders: dict[Pair, frozenset[int]]
    """Agents who asked to add each pair; they pay c_ij if it forms."""


@dataclass(frozen=True)
class EdgeChanges:
    """Links that appeared and disappeared during one evolution step."""

    formed: frozenset[Pair]
    """New links."""
    dropped: frozenset[Pair]
    """Removed links."""


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    """Return the logistic function of the value."""
    return 1.0 / (1.0 + np.exp(-value))


def create_network(
    *, agent_ids: list[int], costs: dict[int, float], config: NetworkConfig, rng: np.random.Generator
) -> NetworkState:
    """Draw pair features, the polynomial affinity and the initial graph."""
    ids = sorted(agent_ids)
    pairs = all_pairs(ids)
    affinity = _affinity_scores(features=_pair_features(agent_ids=ids, costs=costs, rng=rng), rng=rng)
    return NetworkState(
        agent_ids=tuple(ids),
        affinity=affinity,
        edges=_initial_edges(config=config, agent_ids=ids, affinity=affinity, rng=rng),
        history={pair: 0.0 for pair in pairs},
        utility={pair: 0.0 for pair in pairs},
        link_costs={pair: 0.0 for pair in pairs},
    )


def collect_link_requests(actions: dict[int, AgentAction]) -> LinkRequests:
    """Aggregate agents' link actions by pair."""
    removals: set[Pair] = set()
    bonuses: dict[Pair, int] = {}
    adders: dict[Pair, set[int]] = {}
    for agent_id, action in sorted(actions.items()):
        request = action.link_action
        if request is None or request.target == agent_id:
            continue
        pair = as_pair((agent_id, request.target))
        if request.kind == LinkActionKind.remove:
            removals.add(pair)
            continue
        bonuses[pair] = bonuses.get(pair, 0) + 1
        if request.kind == LinkActionKind.add:
            adders.setdefault(pair, set()).add(agent_id)
    return LinkRequests(
        removals=frozenset(removals), bonuses=bonuses, adders={pair: frozenset(ids) for pair, ids in adders.items()}
    )


def update_pair_memory(
    *, state: NetworkState, cooperation: dict[Pair, float], utility: dict[Pair, float], rate: float
) -> None:
    """Move H_ij and U_ij towards this step's signals."""
    for pair in state.history:
        state.history[pair] += rate * (cooperation.get(pair, 0.0) - state.history[pair])
        state.utility[pair] += rate * (utility.get(pair, 0.0) - state.utility[pair])


def apply_link_cost_changes(*, state: NetworkState, changes: dict[Pair, float], decay: float) -> None:
    """Fade all link costs, then add the arbiter's changes, never below zero."""
    for pair in state.link_costs:
        state.link_costs[pair] *= 1.0 - decay
    for pair, delta in changes.items():
        if pair in state.link_costs:
            state.link_costs[pair] = max(0.0, state.link_costs[pair] + delta)


def evolve_edges(
    *,
    state: NetworkState,
    active: set[int],
    requests: LinkRequests,
    expected_gain: dict[Pair, float],
    config: NetworkConfig,
    rng: np.random.Generator,
) -> EdgeChanges:
    """Draw G_{t+1}: each pair keeps or forms a link with its own probability."""
    if config.mode == 'static':
        return EdgeChanges(formed=frozenset(), dropped=frozenset())
    new_edges: set[Pair] = set()
    for pair in sorted(state.history):
        # Draw for every pair so skipped pairs do not shift the random stream of the others.
        draw = rng.uniform()
        if not set(pair) <= active or pair in requests.removals:
            continue
        gain = expected_gain.get(pair, 0.0)
        bonus = requests.bonuses.get(pair, 0)
        if draw < _link_probability(state=state, pair=pair, expected_gain=gain, bonus=bonus, config=config):
            new_edges.add(pair)
    changes = EdgeChanges(formed=frozenset(new_edges - state.edges), dropped=frozenset(state.edges - new_edges))
    state.edges = new_edges
    return changes


def link_fees(*, formed: frozenset[Pair], requests: LinkRequests, link_costs: dict[Pair, float]) -> dict[int, float]:
    """Return what each agent pays for the links it asked for and that formed."""
    fees: dict[int, float] = {}
    for pair in sorted(formed):
        for agent_id in requests.adders.get(pair, frozenset()):
            fees[agent_id] = fees.get(agent_id, 0.0) + link_costs[pair]
    return fees


def _link_probability(
    *, state: NetworkState, pair: Pair, expected_gain: float, bonus: int, config: NetworkConfig
) -> float:
    base = config.a_keep if pair in state.edges else config.a_form
    logit = (
        base
        + config.w_affinity * state.affinity[pair]
        + config.w_history * state.history[pair]
        + config.w_utility * state.utility[pair]
        + config.w_coop * expected_gain
        - config.w_link_cost * state.link_costs[pair]
        + config.agent_link_bonus * bonus
    )
    return float(sigmoid(logit))


def _pair_features(
    *, agent_ids: list[int], costs: dict[int, float], rng: np.random.Generator
) -> dict[Pair, np.ndarray]:
    locations = {agent_id: rng.uniform(size=2) for agent_id in agent_ids}
    spread = (max(costs[agent_id] for agent_id in agent_ids) - min(costs[agent_id] for agent_id in agent_ids)) or 1.0
    features: dict[Pair, np.ndarray] = {}
    for first, second in all_pairs(agent_ids):
        cost_similarity = 1.0 - abs(costs[first] - costs[second]) / spread
        closeness = 1.0 - float(np.linalg.norm(locations[first] - locations[second])) / math.sqrt(2.0)
        features[(first, second)] = np.array([cost_similarity, closeness, rng.uniform()])
    return features


def _polynomial_terms(z: np.ndarray) -> np.ndarray:
    x1, x2, x3 = z
    return np.array([1.0, x1, x2, x3, x1 * x1, x2 * x2, x3 * x3, x1 * x2, x1 * x3, x2 * x3])


def _affinity_scores(*, features: dict[Pair, np.ndarray], rng: np.random.Generator) -> dict[Pair, float]:
    weights = rng.normal(size=POLYNOMIAL_TERMS)
    raw = {pair: float(_polynomial_terms(z) @ weights) for pair, z in features.items()}
    values = np.array(list(raw.values()))
    spread = float(values.std()) or 1.0
    center = float(values.mean())
    return {pair: (score - center) / spread for pair, score in raw.items()}


def _density_offset(*, affinity: dict[Pair, float], density: float) -> float:
    scores = np.array(list(affinity.values()))
    low, high = -20.0, 20.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if float(np.mean(sigmoid(scores + middle))) < density:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _initial_edges(
    *, config: NetworkConfig, agent_ids: list[int], affinity: dict[Pair, float], rng: np.random.Generator
) -> set[Pair]:
    if config.topology == 'complete':
        return set(all_pairs(agent_ids))
    if config.topology == 'star':
        return {as_pair((agent_ids[0], other)) for other in agent_ids[1:]}
    if config.topology == 'path':
        return {as_pair(agent_ids[index:index + 2]) for index in range(len(agent_ids) - 1)}
    if config.topology == 'inner_coalitions':
        blocks = np.array_split(np.array(agent_ids), max(1, config.inner_groups))
        return {pair for block in blocks for pair in all_pairs(int(agent_id) for agent_id in block)}
    offset = _density_offset(affinity=affinity, density=config.initial_density)
    return {pair for pair in sorted(affinity) if rng.uniform() < float(sigmoid(affinity[pair] + offset))}
