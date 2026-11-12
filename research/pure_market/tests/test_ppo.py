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

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from market_rl.arbiter_v2 import ArbiterAdapter, PPOGRUArbiterV2
    from market_rl.arbiter_v2.adapter import to_observation
    from market_rl.arbiter_v2.contracts import public_market
    from market_rl.arbiter_v2.features import PublicMemory
    from market_rl.arbiter_v2.reference import PublicReference
    from market_rl.arbiter_v2.settings import Settings


@unittest.skipUnless(HAS_TORCH, "Install requirements-arbiter.txt to test PPO+GRU")
class PPOIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_adapter(self, config=None, batch=1):
        config = config or MarketConfig(horizon=8)
        settings = Settings(screening_mode="learned", hidden_size=8, min_history=3, history_window=5,
                            cooldown=1, batch_episodes=batch, epochs=1, initial_fine_probability=0.8)
        reference = PublicReference(asdict(public_market(config)), 10., 12., 1.000001, 3, 5,
                                    ("ref:1",), ("cal:2",))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            reference.save(path)
            policy = PPOGRUArbiterV2(market=asdict(public_market(config)), settings=asdict(settings), reference_path=path)
        return ArbiterAdapter(policy, config)

    def test_conversion_reconstructs_public_sales_and_preserves_native_welfare(self):
        config = MarketConfig()
        public = PublicObservation(1, (2., 4., 6., None), 60., 0., 999., 0., 500.)
        obs = to_observation(public, public_market(config))
        self.assertEqual(public_market(config).share_model, "tz_linear_weighted")
        self.assertEqual(public_market(config).welfare_scale, config.welfare_bound)
        self.assertEqual(obs.active_agents, (0, 1, 2))
        np.testing.assert_allclose(list(obs.quantities.values()), [25., 20., 15.])
        self.assertEqual(obs.social_welfare, 999.)
        self.assertEqual(obs.delta_social_welfare, 500.)
        self.assertEqual(to_observation(replace(public, prices=(None,) * 4, total_quantity=0.), public_market(config)).quantities, {})
        single = to_observation(replace(public, prices=(None, 4., None, None)), public_market(config))
        self.assertEqual(single.quantities, {1: 60.})

    def test_logit_conversion_matches_clearing_and_cannot_cross_demand_models(self):
        from market_rl.coalitions import clear_market
        config = MarketConfig(demand_model="logit", horizon=8)
        env = MarketEnv(config)
        env.reset(11)
        public = env.market_step([0, 5, 10, 15])
        obs = to_observation(public, public_market(config))
        quantities = clear_market(np.array(public.prices), np.array(config.costs), np.ones(4, bool),
            config.demand_a, config.demand_b, model="logit")[0]
        np.testing.assert_allclose(list(obs.quantities.values()), quantities)
        self.assertEqual(obs.social_welfare, public.social_welfare)
        adapter = self.make_adapter(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logit.pt"
            adapter.save_checkpoint(path)
            with self.assertRaises(ValueError):
                ArbiterAdapter.load(path, replace(config, demand_model="assignment"))
            with self.assertRaises(ValueError):
                ArbiterAdapter.load(path, replace(config, harm_baseline="proposals"))

    def test_learned_screen_opens_at_price_cap_without_removing_budget_masks(self):
        adapter = self.make_adapter()
        policy = adapter.policy
        memory = PublicMemory(policy.market, policy.settings, policy.reference)
        frame = None
        for step in range(1, 4):
            public = PublicObservation(step, (12.,) * 4, 40., 12., 540., 0., 0.)
            frame = memory.update(to_observation(public, policy.market))
        self.assertTrue(frame.eligible.all())
        self.assertFalse(frame.risks.any())
        original = replace(policy.settings, screening_mode="calibrated")
        memory = PublicMemory(policy.market, original, policy.reference)
        for step in range(1, 4):
            frame = memory.update(to_observation(replace(public, step=step), policy.market))
        self.assertFalse(frame.eligible.any())
        memory = PublicMemory(policy.market, policy.settings, policy.reference)
        memory.paid = {0: policy.settings.seller_budget}
        for step in range(1, 4):
            frame = memory.update(to_observation(replace(public, step=step), policy.market))
        self.assertFalse(frame.eligible[0])

    def test_training_updates_networks_and_does_not_train_sellers(self):
        config = MarketConfig(horizon=8)
        adapter = self.make_adapter(config)
        agents = make_agents(30)
        before = {k: v.clone() for k, v in adapter.policy.actor.state_dict().items()}
        metrics = run_episode(MarketEnv(config), agents, training=False, coalition_policy=CoalitionPolicy(6),
                              arbiter=adapter, arbiter_training=True)
        self.assertEqual(adapter.policy.updates, 1)
        self.assertTrue(any(not torch.equal(before[k], value) for k, value in adapter.policy.actor.state_dict().items()))
        self.assertGreater(metrics["sanctions"], 0)
        self.assertTrue(all(a.updates == 0 for a in agents))
        self.assertIsNone(adapter.policy.pending)
        self.assertFalse(adapter.policy.episode_active)
        self.assertTrue(all(np.isfinite(value) for value in adapter.policy.last_losses.values()))

    def test_partial_batch_checkpoint_resumes_exactly(self):
        config = MarketConfig(horizon=8)
        adapter = self.make_adapter(config, batch=2)
        run_episode(MarketEnv(config), make_agents(30), training=False, coalition_policy=CoalitionPolicy(6),
                    arbiter=adapter, arbiter_training=True)
        self.assertEqual(len(adapter.policy.completed), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arbiter.pt"
            adapter.save_checkpoint(path)
            restored = ArbiterAdapter.load(path, config)
            for candidate in (adapter, restored):
                run_episode(MarketEnv(config), make_agents(30, 11), seed=11, training=False,
                            coalition_policy=CoalitionPolicy(6, 11), arbiter=candidate, arbiter_training=True)
            self.assertEqual(restored.policy.updates, 1)
            for left, right in zip(adapter.policy.actor.parameters(), restored.policy.actor.parameters()):
                self.assertTrue(torch.equal(left, right))
            for left, right in zip(adapter.policy.critic.parameters(), restored.policy.critic.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_evaluation_preserves_arbiter_weights_memory_and_rng(self):
        config = MarketConfig(horizon=8)
        adapter = self.make_adapter(config)
        run_episode(MarketEnv(config), make_agents(30), training=False, coalition_policy=CoalitionPolicy(6),
                    arbiter=adapter, arbiter_training=True)
        before = {k: v.clone() for k, v in adapter.policy.actor.state_dict().items()}
        generator = adapter.policy.generator.get_state().clone()
        evaluate(config, make_agents(30), CoalitionPolicy(6), episodes=2, arbiter=adapter)
        for name, value in adapter.policy.actor.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))
        self.assertTrue(torch.equal(generator, adapter.policy.generator.get_state()))
        self.assertEqual(adapter.policy.updates, 1)

    def test_old_checkpoint_and_wrong_market_are_rejected(self):
        config = MarketConfig(horizon=8)
        adapter = self.make_adapter(config)
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.pt"
            torch.save({"schema_version": 1}, old)
            with self.assertRaises(ValueError):
                ArbiterAdapter.load(old, config)
            native = Path(directory) / "native.pt"
            adapter.save_checkpoint(native)
            with self.assertRaises(ValueError):
                ArbiterAdapter.load(native, replace(config, demand_b=6.))


if __name__ == "__main__":
    unittest.main()
