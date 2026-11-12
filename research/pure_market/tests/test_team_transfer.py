from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT.parents[1] if (ROOT.parents[1] / "src/social_market_env").is_dir() else ROOT.parent / "sma_itmo/marl-market-env"
DATA = ROOT / "runs/team_transfer"
AVAILABLE = (importlib.util.find_spec("torch") is not None
             and (NATIVE / "src/social_market_env").is_dir()
             and (DATA / "prepared.json").exists())


@unittest.skipUnless(AVAILABLE, "Requires the prepared local team environment integration")
class NativeTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from market_rl.team_sources import activate_sources
        activate_sources(NATIVE, DATA / "team_sources")
        sys.path.insert(0, str(ROOT / "team_transfer"))
        import torch
        torch.set_num_threads(1)

    def test_conversion_preserves_both_networks_and_rejects_wrong_market(self):
        import torch
        from transferred_ppo_gru import TeamPPOGRU
        from transferred_ppo_gru.cli import native_config
        from transferred_ppo_gru.convert import state_digest
        prepared = json.loads((DATA / "prepared.json").read_text())
        for i, conversion in enumerate(prepared["models"]):
            source_path = DATA / 'source_models' / f'model_{i}.pt'
            source = torch.load(source_path, map_location="cpu", weights_only=True)
            target = torch.load(DATA / f"initial_model_{i}.pt", map_location="cpu", weights_only=True)
            for network in ("actor", "critic"):
                self.assertEqual(state_digest(source[network]), state_digest(target[network]))
            self.assertFalse(target["actor_optimizer"]["state"])
            self.assertFalse(target["critic_optimizer"]["state"])
            self.assertFalse(target["completed"])
            self.assertEqual(target["updates"], 0)
            self.assertEqual(target["market"]["welfare_model"], "consumer_surplus")
            self.assertEqual(target["market"]["market_family"], "team_linear_total")
        config = native_config(8)
        with self.assertRaises(ValueError):
            TeamPPOGRU(market=asdict(replace(config.market, tau=6.)),
                       checkpoint_path=DATA / "initial_model_0.pt", device="cpu")
        with self.assertRaises(ValueError):
            TeamPPOGRU(market=asdict(config.market), checkpoint_path=DATA / 'source_models/model_0.pt', device="cpu")

    def test_native_api_public_boundary_evaluation_and_training(self):
        from social_market_env.contracts import ArbiterObservation
        from transferred_ppo_gru.api import to_observation
        from transferred_ppo_gru.cli import native_config, build, play, network_digest
        public = ArbiterObservation(0, (0, 1), {0: 25., 1: 30.}, {0: 40., 1: 30.},
                                   70., 27.5, 2450., 4000., 0., 0., None, ())
        self.assertEqual(to_observation(public), to_observation(replace(public, target_social_welfare=1e9)))
        self.assertNotIn("target_social_welfare", asdict(to_observation(public)))
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "model"
            cfg = native_config(6, checkpoint=DATA / "initial_model_0.pt", state_dir=state)
            env = build(cfg, 970000000, DATA / "sellers_initial")
            before = network_digest(env.arbiter.policy)
            play(env, 971000000, training=False)
            self.assertEqual(before, network_digest(env.arbiter.policy))
            self.assertFalse(state.exists())
            for i in range(8):
                play(env, 971000001 + i, training=True)
            self.assertEqual(env.arbiter.policy.updates, 1)
            self.assertEqual(env.arbiter.policy.episodes_seen, 8)
            self.assertIsNone(env.arbiter.pending)
            self.assertIsNone(env.arbiter.policy.pending)
            self.assertEqual(len(json.loads((state / "index.json").read_text())), 9)


if __name__ == "__main__":
    unittest.main()
