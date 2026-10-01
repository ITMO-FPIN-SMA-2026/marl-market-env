import numpy as np
import pytest

from social_market_env.config import NetworkConfig
from social_market_env.contracts import AgentAction, LinkAction, LinkActionKind
from social_market_env.core.network import apply_link_cost_changes, collect_link_requests, create_network, evolve_edges

IDS = list(range(12))
COSTS = {agent_id: 15.0 + agent_id * 0.5 for agent_id in IDS}


def build(config: NetworkConfig, seed: int = 0):
    return create_network(agent_ids=IDS, costs=COSTS, config=config, rng=np.random.default_rng(seed))


def test_random_topology_matches_target_density_on_average() -> None:
    config = NetworkConfig(topology='random', initial_density=0.3)

    densities = [len(build(config, seed).edges) / 66 for seed in range(40)]

    assert np.mean(densities) == pytest.approx(0.3, abs=0.05)


def test_preset_topologies_have_expected_edge_counts() -> None:
    kinds = ('complete', 'star', 'path', 'inner_coalitions')

    counts = {kind: len(build(NetworkConfig(topology=kind, inner_groups=3)).edges) for kind in kinds}

    assert counts == {'complete': 66, 'star': 11, 'path': 11, 'inner_coalitions': 18}


def test_removal_request_drops_link_and_static_mode_freezes_graph() -> None:
    removal = AgentAction(price=20.0, link_action=LinkAction(kind=LinkActionKind.remove, target=1))
    requests = collect_link_requests({0: removal})
    evolving = build(NetworkConfig(topology='complete', a_keep=50.0))
    frozen = build(NetworkConfig(topology='complete', mode='static'))

    changes = evolve_edges(
        state=evolving,
        active=set(IDS),
        requests=requests,
        expected_gain={},
        config=NetworkConfig(topology='complete', a_keep=50.0),
        rng=np.random.default_rng(1),
    )
    evolve_edges(
        state=frozen,
        active={0, 1},
        requests=requests,
        expected_gain={},
        config=NetworkConfig(mode='static'),
        rng=np.random.default_rng(1),
    )

    assert changes.dropped == frozenset({(0, 1)})
    assert len(frozen.edges) == 66


def test_link_costs_never_go_negative_and_decay() -> None:
    state = build(NetworkConfig())

    apply_link_cost_changes(state=state, changes={(0, 1): 2.0, (0, 2): -5.0}, decay=0.0)
    apply_link_cost_changes(state=state, changes={}, decay=0.5)

    assert state.link_costs[(0, 1)] == pytest.approx(1.0)
    assert state.link_costs[(0, 2)] == 0.0
