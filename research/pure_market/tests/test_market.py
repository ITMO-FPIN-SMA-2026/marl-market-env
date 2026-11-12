import copy
from dataclasses import fields, replace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from market_rl.agents import LearnerConfig, TabularAgent, load_agents, load_auxiliary, make_agents, save_agents
from market_rl.coalition_policy import CoalitionPolicy
from market_rl.coalitions import allocate, clear_market
from market_rl.environment import ArbiterAction, MarketConfig, MarketEnv, PublicObservation
from market_rl.train import evaluate, run_episode


class EconomicsTests(unittest.TestCase):
    def test_demand_revenue_and_welfare_conservation(self):
        p = np.array([2., 4., 6., 8.])
        costs = np.array([1., 2., 3., 4.])
        for active in [np.ones(4, bool), np.array([True, False, False, False]), np.zeros(4, bool)]:
            q, profit, price, cs = clear_market(p, costs, active, 100, 5)
            expected_q = max(0, 100 - 5 * price) if active.any() else 0
            self.assertAlmostEqual(q.sum(), expected_q)
            self.assertAlmostEqual(p @ q, price * q.sum())
            self.assertAlmostEqual(profit.sum(), p @ q - costs @ q)
            self.assertAlmostEqual(cs, q.sum()**2 / 10)
            self.assertTrue((q >= 0).all())
        q, profits, _, cs = clear_market(p * 20, costs, np.ones(4, bool), 100, 5)
        self.assertEqual(q.sum() + profits.sum() + cs, 0)

    def test_exact_shapley_myerson_and_owen(self):
        group = (0, 1, 2)
        path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        value = lambda s: float({0, 2} <= s)
        shapley = allocate(group, value, "shapley")
        self.assertEqual(shapley, {0: 0.5, 1: 0., 2: 0.5})
        myerson = allocate(group, value, "myerson", path)
        np.testing.assert_allclose(list(myerson.values()), [1/3] * 3)
        unanimity = lambda s: float(len(s) == 3)
        owen = allocate(group, unanimity, "owen", history=[(0, 1), (2,)])
        self.assertEqual(owen, {0: 0.25, 1: 0.25, 2: 0.5})

    def test_allocations_conserve_real_cash_even_with_negative_surplus(self):
        for method in ["equal", "shapley", "myerson", "owen"]:
            with self.subTest(method=method):
                env = MarketEnv(MarketConfig(topology="path", allocation=method, dynamic_network=False))
                env.reset(12)
                # All agents voluntarily choose label 0; path connects them.
                actions = [1, 6, 16, 26]
                env.market_step(actions, {frozenset(range(4)): 1})
                initial = env.balances.copy()
                _, rewards, _, info = env.settle(ArbiterAction("sanction", 7, 2))
                self.assertAlmostEqual(rewards.sum(), info["operating_profits"].sum() - 7)
                np.testing.assert_allclose(env.balances - initial, rewards)
                self.assertEqual(info["coalitions"], ((0, 1, 2, 3),))


class EnvironmentTests(unittest.TestCase):
    def test_voluntary_connected_coalitions_and_topologies(self):
        expected = {"complete": 6, "star": 3, "path": 3, "inner": 2}
        for topology, edges in expected.items():
            env = MarketEnv(MarketConfig(topology=topology, dynamic_network=False))
            env.reset(0)
            self.assertEqual(env.graph.sum(), edges * 2)
            self.assertFalse(np.diag(env.graph).any())
            self.assertEqual(env.form_coalitions([0] * 4), ((0,), (1,), (2,), (3,)))
            group_sizes = sorted(map(len, env.form_coalitions([1] * 4)))
            self.assertEqual(group_sizes, [2, 2] if topology == "inner" else [4])

    def test_two_phases_and_public_boundary(self):
        env = MarketEnv(MarketConfig(horizon=2))
        env.reset()
        with self.assertRaises(RuntimeError):
            env.settle()
        public = env.market_step([0] * 4)
        self.assertIsInstance(public, PublicObservation)
        self.assertEqual({f.name for f in fields(public)}, {
            "step", "prices", "total_quantity", "mean_price", "social_welfare", "price_change", "welfare_change"})
        with self.assertRaises(RuntimeError):
            env.market_step([0] * 4)
        with self.assertRaises(ValueError):
            env.settle(ArbiterAction("sanction", 10, -1))
        env.settle()
        later = env.market_step([5] * 4)
        self.assertAlmostEqual(later.price_change, later.mean_price - public.mean_price)
        _, _, done, _ = env.settle()
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):
            env.market_step([0] * 4)

    def test_link_cost_does_not_directly_delete_existing_edges(self):
        env = MarketEnv(MarketConfig(topology="path"))
        env.reset(19)
        other = copy.deepcopy(env)
        existing = env.graph.copy()
        for e in (env, other):
            e.market_step([0] * 4)
        _, r1, _, info = env.settle()
        _, r2, _, _ = other.settle(ArbiterAction("link_cost", 100))
        np.testing.assert_array_equal(env.graph[existing], other.graph[existing])
        self.assertFalse(other.graph[~existing].any())
        self.assertAlmostEqual(r1.sum(), info["operating_profits"].sum() - info["link_expenses"].sum())
        self.assertTrue((r2 >= r1 - 1e-10).all())

    def test_debt_restrictions_and_permanent_exit(self):
        env = MarketEnv(MarketConfig(dynamic_network=False, horizon=5, bankruptcy_patience=2))
        env.reset()
        env.market_step([0] * 4)
        env.settle(ArbiterAction("sanction", 10000, 3))
        allowed = np.flatnonzero(env.action_mask(3))
        for index in allowed:
            action = env.decode_action(index)
            self.assertEqual(action.coalition, -1)
            self.assertGreaterEqual(env.config.prices[action.price], env.costs[3])
        env.market_step([0, 0, 0, int(allowed[-1])])
        env.settle()
        self.assertFalse(env.active[3])
        self.assertFalse(env.graph[3].any())
        public = env.market_step([0] * 4)
        self.assertIsNone(public.prices[3])
        _, reward, _, _ = env.settle()
        self.assertEqual(reward[3], 0)


class LearningTests(unittest.TestCase):
    def test_distinct_td_targets(self):
        expected = {"q_learning": 3.0, "sarsa": 2.0, "expected_sarsa": 2.9}
        for algorithm, target in expected.items():
            agent = TabularAgent(LearnerConfig(algorithm, learning_rate=1., gamma=0.5, epsilon=0.2), 2)
            agent.q[(1,)] = np.array([2., 4.])
            agent.learn((0,), 0, 1., (1,), np.ones(2, bool), False, next_action=0)
            self.assertAlmostEqual(agent.q[(0,)][0], target)
        agent = TabularAgent(LearnerConfig("double_q", learning_rate=1., gamma=0.5), 2)
        agent.q[(1,)] = np.array([2., 9.])
        agent.q2[(1,)] = np.array([8., 1.])
        agent.learn((0,), 0, 1., (1,), np.ones(2, bool), False)
        if (0,) in agent.q:
            self.assertEqual(agent.q[(0,)][0], 1.5)
        else:
            self.assertEqual(agent.q2[(0,)][0], 2.0)

    def test_coalition_uses_discounted_returns_across_absence(self):
        env = MarketEnv()
        env.reset()
        policy = CoalitionPolicy(6, gamma=0.5, learning_rate=1.)
        actions = [1, 1, 0, 0]
        chosen = policy.plan(env, actions)
        old_state = policy.state(env, (0, 1))
        old_action = chosen[frozenset((0, 1))]
        policy.observe(np.array([100, 100, 0, 0]), False)
        # The coalition can be absent; every intervening market reward counts.
        policy.plan(env, [0] * 4)
        policy.observe(np.array([200, 200, 0, 0]), False)
        env.t = 2
        env.public = replace(env.public, mean_price=12.)
        new_state = policy.state(env, (0, 1))
        policy.q[new_state] = np.array([10., 0., 0., 0., 0., 0.])
        policy.plan(env, actions)
        self.assertAlmostEqual(policy.q[old_state][old_action], 2 + 0.5 * 4 + 0.25 * 10)

    def test_training_serialization_and_frozen_evaluation(self):
        config = MarketConfig(horizon=8)
        agents = make_agents(30, 7)
        solo_agents = make_agents(6, 300)
        policy = CoalitionPolicy(6, 9)
        run_episode(MarketEnv(config), solo_agents, solo=True)
        seen_public = []

        def external_callback(public):
            seen_public.append(public)
            return ArbiterAction()

        run_episode(MarketEnv(config), agents, coalition_policy=policy, arbiter=external_callback)
        self.assertEqual(len(seen_public), 8)
        self.assertTrue(all(a.updates == 8 for a in agents))
        self.assertGreater(policy.updates, 0)
        before = [a.rng.bit_generator.state for a in agents]
        counts = [len(a.q) + len(a.q2) for a in agents]
        result = evaluate(config, agents, policy, episodes=2)
        self.assertEqual(before, [a.rng.bit_generator.state for a in agents])
        self.assertEqual(counts, [len(a.q) + len(a.q2) for a in agents])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_agents(path, agents, config, policy, solo_agents)
            loaded_config, loaded_agents = load_agents(path)
            loaded_policy, loaded_solo = load_auxiliary(path)
            self.assertEqual(config, loaded_config)
            self.assertEqual(result, evaluate(config, loaded_agents, loaded_policy, episodes=2))
            self.assertEqual(loaded_solo[0].updates, solo_agents[0].updates)
            self.assertEqual(loaded_policy.updates, policy.updates)


if __name__ == "__main__":
    unittest.main()
