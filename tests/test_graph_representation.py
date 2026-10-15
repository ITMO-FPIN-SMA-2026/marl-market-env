"""The graph and per-agent state used by the MARL environment."""

import pytest

from social_market_env.baselines.agents import CartelAgent
from social_market_env.baselines.arbiters import AlwaysObserve
from social_market_env.baselines.coalitions import ConnectedComponentsCoalitions
from social_market_env.config import config_from_dict
from social_market_env.environment import SocialMarketEnvironment
from social_market_env.gametheory.graphs import graph_matrices


def test_three_graph_matrices_share_stable_node_order() -> None:
    matrices = graph_matrices(nodes=[3, 0, 2, 1], edges=[(0, 1), (1, 2)], coalitions=[(0, 1)])

    assert matrices['adjacency'][0] == (0, 1, 0, 0)
    assert matrices['reachability'][0] == (1, 1, 1, 0)
    assert matrices['reachability'][3][3] == 1
    assert matrices['coalition_membership'][0] == (1, 1, 0, 0)
    assert matrices['coalition_membership'][3][3] == 0


def test_agent_histories_and_join_vectors_follow_realized_steps() -> None:
    ids = (0, 1, 2)
    env = SocialMarketEnvironment(
        agents={agent_id: CartelAgent() for agent_id in ids},
        costs={agent_id: 18.0 for agent_id in ids},
        coalition_policy=ConnectedComponentsCoalitions(),
        arbiter=AlwaysObserve(),
        config=config_from_dict({
            'horizon': 2,
            'network': {'topology': 'path', 'mode': 'static', 'w_coop': 0.0},
        }),
    )
    env.reset(seed=0, training=False)

    first = env.step().record
    assert first['coalition_plan_ids'] == [0]
    assert first['coalition_join_vectors'] == {agent_id: [1] for agent_id in ids}
    assert first['graph_matrices']['adjacency'][0] == (0, 1, 0)
    assert first['graph_matrices']['reachability'][0] == (1, 1, 1)
    assert first['graph_matrices']['coalition_membership'][0] == (1, 1, 1)
    assert len(env.agent_state[0].observations) == 1
    assert env.agent_state[0].observations[0].expectations.reward is None
    assert env.agent_state[0].evaluations[0].reward == pytest.approx(first['agent_rewards'][0])

    second = env.step().record
    previous = env.agent_state[0].observations[1]
    assert previous.last_evaluation == env.agent_state[0].evaluations[0]
    assert previous.expectations.reward == pytest.approx(first['agent_rewards'][0])
    assert previous.last_coalition_plan_ids == (0,)
    assert previous.last_coalition_join_vector == (1,)
    assert len(env.agent_state[0].observations) == 2
    assert len(env.agent_state[0].expectation_history) == 3
    assert len(env.agent_state[0].evaluations) == 2
    assert second['agent_evaluations'][0]['reward_error'] == pytest.approx(
        env.agent_state[0].evaluations[1].reward - env.agent_state[0].evaluations[0].reward
    )

    env.reset(seed=1, training=False)
    assert env.agent_state[0].observations == []
    assert env.agent_state[0].evaluations == []
    assert env.agent_state[0].coalition_join_vector == ()
