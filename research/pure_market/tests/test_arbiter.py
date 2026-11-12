from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from market_rl import ArbiterAction, MarketConfig, MarketEnv, PublicObservation
from market_rl.agents import make_agents
from market_rl.arbiter_scoring import ArbiterScoring, score_arbiter
from market_rl.coalition_policy import CoalitionPolicy
from market_rl.train import evaluate, run_episode

class TransitionTests(unittest.TestCase):
    def test_observe_comparison_uses_the_same_reward_target(self):
        scoring = ArbiterScoring(target_welfare=650.)

        class Observe:
            def __init__(self):
                self.scoring = scoring

            def __call__(self, public):
                return ArbiterAction()

        config = MarketConfig(horizon=5)
        agents, coalitions = make_agents(30), CoalitionPolicy(6)
        baseline = evaluate(config, agents, coalitions, episodes=2, arbiter_scoring=scoring)
        regulated = evaluate(config, agents, coalitions, episodes=2, arbiter=Observe())
        self.assertEqual(baseline, regulated)

    def test_stateful_callback_gets_full_consecutive_transitions_and_scalar_only(self):
        class Probe:
            def reset(self, *, seed, training):
                self.seed, self.training, self.pending, self.transitions = seed, training, None, []

            def __call__(self, observation):
                self_outer.assertIsNone(self.pending)
                self.pending = observation.step
                return ArbiterAction()

            def observe_transition(self, **transition):
                self_outer.assertEqual(set(transition), {"observation", "action", "reward", "next_observation", "done"})
                self_outer.assertEqual(transition["observation"].step, self.pending)
                self_outer.assertIsInstance(transition["reward"], float)
                if transition["done"]:
                    self_outer.assertIsNone(transition["next_observation"])
                else:
                    self_outer.assertEqual(transition["next_observation"].step, self.pending + 1)
                self.transitions.append(transition)
                self.pending = None

        self_outer = self
        probe = Probe()
        run_episode(MarketEnv(MarketConfig(horizon=5)), make_agents(30), training=False,
                    coalition_policy=CoalitionPolicy(6), arbiter=probe, arbiter_training=True)
        self.assertTrue(probe.training)
        self.assertEqual(len(probe.transitions), 5)
        self.assertIsNone(probe.pending)

    def test_private_harm_labels_affect_reward_but_do_not_change_public_data(self):
        config = MarketConfig(dynamic_network=False)
        for target, expected_fp, expected_fn in [(0, False, False), (2, True, True)]:
            env = MarketEnv(config)
            env.reset()
            public = env.market_step([1, 1, 10, 15], {frozenset((0, 1)): 5})
            action = ArbiterAction("sanction", 25, target)
            _, _, _, info = env.settle(action)
            reward, verdict = score_arbiter(public, action, info, ArbiterScoring())
            self.assertEqual(verdict["harmful_coalitions"], 1)
            self.assertEqual(verdict["false_positive"], expected_fp)
            self.assertEqual(verdict["false_negative"], expected_fn)
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(hasattr(public, "counterfactuals"))
        env.reset()
        public = env.market_step([25] * 4)  # high prices alone are not coalition guilt
        action = ArbiterAction("sanction", 25, 0)
        _, _, _, info = env.settle(action)
        _, verdict = score_arbiter(public, action, info, ArbiterScoring())
        self.assertTrue(verdict["false_positive"])
        self.assertEqual(verdict["harmful_coalitions"], 0)



if __name__ == "__main__":
    unittest.main()
