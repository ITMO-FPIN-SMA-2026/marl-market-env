import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from market_rl import ArbiterAction, MarketConfig, MarketEnv, PublicObservation
from market_rl.agents import encode_state, load_agents, load_auxiliary, make_agents, save_agents
from market_rl.coalition_policy import CoalitionPolicy
from market_rl.learn_arbiter import adapt, fit
from market_rl.sanction import LearnedArbiter, PublicEncoder, SanctionSettings, market_scales
from market_rl.train import run_episode


class SanctionTests(unittest.TestCase):
    def test_own_fine_survives_observation_binning(self):
        env = MarketEnv(MarketConfig(dynamic_network=False))
        env.reset()
        other = copy.deepcopy(env)
        for current in (env, other):
            current.market_step([20]*4)
        normal, _, _, _ = env.settle()
        fined, _, _, _ = other.settle(ArbiterAction("sanction", 25, 0))
        a = encode_state(normal[0], env.config.prices)
        b = encode_state(fined[0], env.config.prices)
        self.assertEqual(a[:-1], b[:-1])
        self.assertEqual((a[-1], b[-1]), (0, 1))
        self.assertEqual(fined[0].last_fine, 25)
        self.assertEqual(env.public, other.public)

    def test_legacy_market_tables_are_migrated_without_losing_values(self):
        config = MarketConfig(horizon=5)
        agents, coalition = make_agents(30), CoalitionPolicy(6)
        run_episode(MarketEnv(config), agents, coalition_policy=coalition)
        with tempfile.TemporaryDirectory() as directory:
            native, legacy = Path(directory)/"native.json", Path(directory)/"legacy.json"
            save_agents(native, agents, config, coalition)
            payload = json.loads(native.read_text())
            payload["format"] = 1
            for agent in payload["agents"]:
                for table in ("q", "q2"):
                    for row in agent[table]:
                        row[0] = row[0][:-1]
            payload["coalition_policy"].pop("state_version")
            for row in payload["coalition_policy"]["q"]:
                row[0] = row[0][:-1]
            legacy.write_text(json.dumps(payload))
            _, migrated = load_agents(legacy)
            migrated_coalition, _ = load_auxiliary(legacy)
            for original, loaded in zip(agents, migrated):
                for table in ("q", "q2"):
                    self.assertEqual(getattr(original, table).keys(), getattr(loaded, table).keys())
                    for state, value in getattr(original, table).items():
                        np.testing.assert_array_equal(value, getattr(loaded, table)[state])
            self.assertEqual(coalition.q.keys(), migrated_coalition.q.keys())

    def test_encoder_is_public_and_equivariant_under_seller_relabeling(self):
        config = MarketConfig()
        first = PublicObservation(1, (2., 4., 8., 12.), 60., 6., 700., 0., 0.)
        second = replace(first, step=2, prices=(4., 6., 10., 8.))
        permutation = np.array([2, 0, 3, 1])
        original, permuted = PublicEncoder(market_scales(config)), PublicEncoder(market_scales(config))
        for obs in (first, second):
            a = original.encode(obs)
            b = permuted.encode(replace(obs, prices=tuple(obs.prices[i] for i in permutation)))
            np.testing.assert_allclose(a[permutation], b, atol=1e-7)

    def test_financial_masks_and_roundtrip_without_torch(self):
        config = MarketConfig()
        policy = LearnedArbiter(config, SanctionSettings(threshold=.7, fine=25, seller_budget=25,
                                episode_budget=50, cooldown=2, min_history=1))
        policy.weights = [(np.zeros((1,25),np.float32), np.array([5.],np.float32))]
        public = PublicObservation(1, (None, 12., None, None), 40., 12., 540., 0., 0.)
        one = policy(public)
        self.assertEqual(one.target, 1)
        self.assertEqual(policy(replace(public, step=2)).kind, "observe")
        self.assertEqual(policy(replace(public, step=3)).kind, "observe")  # seller budget used up
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"arbiter.json"
            policy.save(path)
            loaded = LearnedArbiter.load(path, config)
            self.assertEqual(loaded(public), one)
            self.assertNotIn("graph", json.loads(path.read_text()))
            self.assertNotIn("costs", json.loads(path.read_text()))
            payload = json.loads(path.read_text())
            payload["settings"].update(minimum_effect=0., fine_rate=0.)
            payload["effect_weights"] = []
            path.write_text(json.dumps(payload))
            self.assertEqual(LearnedArbiter.load(path, config)(public), one)
            payload["settings"]["fine_rate"] = .25
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "archiv"):
                LearnedArbiter.load(path, config)
            payload["settings"]["fine_rate"] = 0.
            payload["effect_weights"] = [[[1.], [0.]]]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "archiv"):
                LearnedArbiter.load(path, config)

    def test_adaptation_saves_both_sides_and_an_index(self):
        config = MarketConfig(horizon=3)
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)/"weights"/"adaptation"
            agents, _, _, _ = adapt(config, make_agents(30), CoalitionPolicy(6), None,
                                   episodes=2, seed=1, checkpoint_dir=stage)
            _, before = load_agents(stage/"market_before.json")
            _, after = load_agents(stage/"market_after.json")
            self.assertEqual(before[0].updates, 0)
            self.assertEqual(after[0].updates, agents[0].updates)
            records = json.loads((stage.parent/"index.json").read_text())
            self.assertEqual([r["phase"] for r in records], ["before", "after"])
            self.assertTrue(all(len(r["files"][0]["sha256"]) == 64 for r in records))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Training checkpoints need optional PyTorch")
    def test_every_predictor_epoch_and_optimizer_is_saved(self):
        import torch

        config = MarketConfig()
        x = np.random.default_rng(0).normal(size=(64,25)).astype(np.float32)
        y = (x[:,0]>0).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)/"weights"/"fit"
            fit(config, x, y, epochs=2, checkpoint_dir=stage)
            for epoch in (0,1,2):
                self.assertTrue((stage/f"epoch_{epoch:03d}.json").exists())
                self.assertTrue((stage/f"epoch_{epoch:03d}.pt").exists())
            initial = torch.load(stage/"epoch_000.pt", weights_only=True)
            trained = torch.load(stage/"epoch_002.pt", weights_only=True)
            self.assertFalse(initial["optimizer"]["state"])
            self.assertTrue(trained["optimizer"]["state"])
            self.assertIn("shuffle_rng", trained)
            policy = LearnedArbiter.load(stage/"epoch_000.json", config)
            public = PublicObservation(1, (12.,)*4, 40., 12., 540., 0., 0.)
            self.assertEqual(policy(public).kind,"observe")
            partial = Path(directory)/"weights"/"partial"
            fit(config, x, y, epochs=1, checkpoint_dir=partial)
            resumed = fit(config, x, y, epochs=1, resume_path=partial/"epoch_001.pt")
            uninterrupted = LearnedArbiter.load(stage/"epoch_002.json", config)
            for (left_w,left_b),(right_w,right_b) in zip(resumed.weights, uninterrupted.weights):
                np.testing.assert_array_equal(left_w,right_w)
                np.testing.assert_array_equal(left_b,right_b)


if __name__ == "__main__":
    unittest.main()
