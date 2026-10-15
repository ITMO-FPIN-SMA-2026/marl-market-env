import unittest

import networkx as nx

from social_market_env import EnvironmentConfig, SocialMarketEnvironment
from social_market_env.contracts import (
    AgentAction,
    ArbiterAction,
    ArbiterAPI,
    CoalitionDecision,
    CoalitionPolicyAPI,
    MarketAgentAPI,
)
from social_market_env.myerson import graph_matrices, myerson_value


class FixedAgent(MarketAgentAPI):
    def __init__(self, action):
        self.action = action

    def reset(self, agent_id, seed=None):
        pass

    def act(self, observation):
        return self.action


class FixedCoalitions(CoalitionPolicyAPI):
    def __init__(self, coalitions=()):
        self.coalitions = coalitions

    def reset(self, seed=None):
        pass

    def propose(self, observation):
        return CoalitionDecision(self.coalitions)


class PassiveArbiter(ArbiterAPI):
    def reset(self, seed=None):
        pass

    def act(self, observation):
        return ArbiterAction()


def make_env(actions, coalitions=()):
    return SocialMarketEnvironment(
        agents={aid: FixedAgent(action) for aid, action in actions.items()},
        agent_costs={aid: 10.0 for aid in actions},
        coalition_policy=FixedCoalitions(coalitions),
        arbiter=PassiveArbiter(),
        config=EnvironmentConfig(initial_edge_probability=0.0, horizon=2, seed=4),
    )


class GraphAndStateTests(unittest.TestCase):
    def test_myerson_path_rewards_bridge(self):
        graph = nx.path_graph(3)
        worth = lambda group: 1.0 if len(group) >= 2 else 0.0
        value = myerson_value(graph, (0, 1, 2), worth)
        self.assertAlmostEqual(value[0], 1 / 6)
        self.assertAlmostEqual(value[1], 2 / 3)
        self.assertAlmostEqual(value[2], 1 / 6)
        self.assertAlmostEqual(sum(value.values()), 1.0)

    def test_large_coalition_sampling_preserves_total_worth(self):
        graph = nx.path_graph(13)
        worth = lambda group: 5.0 if len(group) >= 2 else 0.0
        value = myerson_value(graph, range(13), worth, samples=100)
        self.assertAlmostEqual(sum(value.values()), 5.0)

    def test_three_matrices_have_distinct_meanings(self):
        graph = nx.path_graph(3)
        matrices = graph_matrices(graph, ((0, 1),), (0, 1, 2))
        self.assertEqual(matrices["adjacency"][0][2], 0)
        self.assertEqual(matrices["reachability"][0][2], 1)
        self.assertEqual(matrices["coalition_membership"][0][1], 1)
        self.assertEqual(matrices["coalition_membership"][1][2], 0)

    def test_forced_new_link_needs_both_agents(self):
        both = make_env({0: AgentAction(20, link_action=("add", 1)), 1: AgentAction(20, link_action=("add", 0))})
        both.reset()
        both.rng.random = lambda: 1.0
        both.step()
        self.assertEqual(both.edges, {(0, 1)})

        one = make_env({0: AgentAction(20, link_action=("add", 1)), 1: AgentAction(20)})
        one.reset()
        one.rng.random = lambda: 1.0
        one.step()
        self.assertEqual(one.edges, set())

    def test_join_choice_and_oee_history(self):
        env = make_env(
            {0: AgentAction(20), 1: AgentAction(20), 2: AgentAction(20, coalition_choice=-1)},
            ((0, 1, 2),),
        )
        env.reset()
        env.graph.add_edges_from(((0, 1), (1, 2)))
        next_obs, _, _, info = env.step()
        self.assertEqual(env.current_coalitions, ((0, 1),))
        self.assertEqual(info["record"]["coalition_join_vectors"][2], (0,))
        self.assertEqual(len(env.agent_state[0].observations), 3)
        self.assertEqual(len(env.agent_state[0].expectation_history), 2)
        self.assertEqual(len(env.agent_state[0].evaluations), 1)
        self.assertEqual(next_obs[0].last_evaluation.profit, info["record"]["profits"][0])
        self.assertEqual(next_obs[0].expectations.profit, info["record"]["profits"][0])
        self.assertEqual(next_obs[0].market_average_price, info["record"]["average_price"])
        self.assertEqual(env.agent_state[0].last_profit, info["record"]["profits"][0])

    def test_offer_indices_stay_global_and_remove_is_unilateral(self):
        env = make_env(
            {
                0: AgentAction(20, link_action=("remove", 1)),
                1: AgentAction(20, coalition_choice=1),
                2: AgentAction(20, coalition_choice=1),
            },
            ((0, 1), (1, 2)),
        )
        env.reset()
        env.graph.add_edges_from(((0, 1), (1, 2)))
        env.rng.random = lambda: 1.0
        env.step()
        self.assertEqual(env.agent_state[1].observations[-2].candidate_coalitions, ((0, (0, 1)), (1, (1, 2))))
        self.assertEqual(env.current_coalitions, ((1, 2),))
        self.assertNotIn((0, 1), env.edges)


if __name__ == "__main__":
    unittest.main()
