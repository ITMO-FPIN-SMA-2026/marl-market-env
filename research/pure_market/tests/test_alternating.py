from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from market_rl import MarketConfig
from market_rl.agents import load_agents, load_auxiliary, make_agents, save_agents
from market_rl.coalition_policy import CoalitionPolicy

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from market_rl.arbiter_v2 import alternating
    from market_rl.arbiter_v2.settings import Settings


@unittest.skipUnless(HAS_TORCH, "Install requirements-arbiter.txt to test PPO alternation")
class AlternatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, directory):
        market = Path(directory) / "market.json"
        save_agents(market, make_agents(30), MarketConfig(demand_model="logit", horizon=6),
                    CoalitionPolicy(6), make_agents(6))
        schedule = alternating.Schedule(cycles=2, ppo_episodes=2, adapt_episodes=2,
            reference_episodes=1, validation_worlds=2, test_worlds=2, replicas=1,
            adaptation_checkpoint_every=1)
        settings = Settings(screening_mode="learned", hidden_size=8, min_history=3,
                            history_window=5, batch_episodes=2, epochs=1,
                            initial_fine_probability=.8)
        return market, schedule, settings

    def test_separate_learning_stages_and_matched_control(self):
        with tempfile.TemporaryDirectory() as directory:
            market, schedule, settings = self.fixture(directory)
            out = Path(directory) / "study"
            report = alternating.study(market, out, schedule, device="cpu", settings=settings)
            weights = out / "replica_0/weights"
            _, frozen = load_agents(weights / "cycle_001_ppo_0001/regulated.json")
            self.assertTrue(all(a.updates == 0 for a in frozen))
            for arm in ("regulated", "baseline"):
                _, adapted = load_agents(weights / f"cycle_002_adapt_00002/{arm}.json")
                self.assertTrue(all(a.updates > 0 for a in adapted))
            ppo = torch.load(weights / "cycle_001_ppo_0001/arbiter.pt", weights_only=True)
            adapted = torch.load(weights / "cycle_001_adapt_00002/arbiter.pt", weights_only=True)
            initial = torch.load(weights / "initial/arbiter.pt", weights_only=True)
            for network in ("actor", "critic"):
                self.assertTrue(any(not torch.equal(initial[network][k], ppo[network][k])
                                    for k in ppo[network]))
                for name in ppo[network]:
                    self.assertTrue(torch.equal(ppo[network][name], adapted[network][name]))
            self.assertFalse(report["qualified"])
            self.assertFalse(report["qualification"]["independent_training_replicas"])
            self.assertEqual(len(json.loads((out / "test_worlds.json").read_text())), 1)
            config = json.loads((out / "run_config.json").read_text())
            self.assertFalse(config["reward_changed"])
            self.assertEqual(config["settings"], asdict(settings))

    def test_interrupted_alternation_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            market, schedule, settings = self.fixture(directory)
            schedule = replace(schedule, learn_during_ppo=True)
            complete, interrupted = Path(directory) / "complete", Path(directory) / "interrupted"
            expected = alternating.study(market, complete, schedule, device="cpu", settings=settings)
            original_save = alternating.save_stage

            def interrupt_after_commit(root, name, *args, **kwargs):
                result = original_save(root, name, *args, **kwargs)
                if name == "cycle_001_adapt_00001":
                    raise InterruptedError("simulated interruption after durable checkpoint")
                return result

            with patch.object(alternating, "save_stage", side_effect=interrupt_after_commit):
                with self.assertRaises(InterruptedError):
                    alternating.study(market, interrupted, schedule, device="cpu", settings=settings)
            actual = alternating.study(market, interrupted, schedule, device="cpu", resume=True, settings=settings)
            self.assertEqual(expected, actual)
            relative = "replica_0/weights/cycle_002_adapt_00002"
            self.assertEqual((complete / relative / "regulated.json").read_bytes(),
                             (interrupted / relative / "regulated.json").read_bytes())
            self.assertEqual((complete / relative / "baseline.json").read_bytes(),
                             (interrupted / relative / "baseline.json").read_bytes())
            left = torch.load(complete / relative / "arbiter.pt", weights_only=True)
            right = torch.load(interrupted / relative / "arbiter.pt", weights_only=True)
            for network in ("actor", "critic"):
                for key in left[network]:
                    self.assertTrue(torch.equal(left[network][key], right[network][key]))

    def test_recipients_learn_in_ppo_rollouts_with_equal_control_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            market, schedule, settings = self.fixture(directory)
            schedule = replace(schedule, cycles=1, learn_during_ppo=True)
            settings = replace(settings, initial_fine_probability=.5)
            out = Path(directory) / "study"
            alternating.study(market, out, schedule, device="cpu", settings=settings)
            stage = out / "replica_0/weights/cycle_001_ppo_0001"
            for arm in ("regulated", "baseline"):
                _, agents = load_agents(stage / f"{arm}.json")
                self.assertTrue(all(a.updates > 0 for a in agents))
                config, _ = load_agents(stage / f"{arm}.json")
                self.assertTrue(all(a.updates <= config.horizon * schedule.ppo_episodes for a in agents))
                coalition, solo = load_auxiliary(stage / f"{arm}.json")
                self.assertTrue(coalition.q)
                self.assertTrue(all(a.updates == 0 for a in solo))
            initial = torch.load(out / "replica_0/weights/initial/arbiter.pt", weights_only=True)
            self.assertTrue(torch.equal(initial["actor"]["gate.bias"], torch.zeros(1)))
            self.assertEqual(initial["metadata"]["protocol"], "learning_recipients")

    def test_no_interventions_leave_learning_branches_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            market, schedule, settings = self.fixture(directory)
            schedule = replace(schedule, cycles=1, learn_during_ppo=True)
            settings = replace(settings, initial_fine_probability=1e-12)
            out = Path(directory) / "study"
            report = alternating.study(market, out, schedule, device="cpu", settings=settings)
            for name in ("cycle_001_ppo_0001", "cycle_001_adapt_00002"):
                stage = out / "replica_0/weights" / name
                self.assertEqual((stage / "regulated.json").read_bytes(),
                                 (stage / "baseline.json").read_bytes())
            self.assertEqual(report["net_welfare_gain"]["mean"], 0.)

    def test_original_actor_pretraining_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            market, schedule, settings = self.fixture(directory)
            schedule = replace(schedule, cycles=1, learn_during_ppo=True,
                               pretrain_epochs=2, pretrain_worlds=5, pretrain_validation_worlds=5)
            settings = replace(settings, initial_fine_probability=.5)
            complete, interrupted = Path(directory) / "complete", Path(directory) / "interrupted"
            expected = alternating.study(market, complete, schedule, device="cpu", settings=settings)
            root = complete / "replica_0/weights"
            initial = torch.load(root / "initial/arbiter.pt", weights_only=True)
            first_epoch = torch.load(root / "pretrain_epoch_001/arbiter.pt", weights_only=True)
            selected = torch.load(root / "pretrain_selected/arbiter.pt", weights_only=True)
            self.assertEqual(initial["actor"].keys(), selected["actor"].keys())
            self.assertTrue(any(not torch.equal(initial["actor"][key], selected["actor"][key])
                                for key in initial["actor"]))
            for key in initial["critic"]:
                self.assertTrue(torch.equal(initial["critic"][key], selected["critic"][key]))
            self.assertTrue(first_epoch["metadata"]["pretrain_optimizer"]["state"])
            self.assertFalse(selected["actor_optimizer"]["state"])
            self.assertTrue(selected["metadata"]["actor_pretrained"])
            self.assertEqual(selected["updates"], 0)
            original_save = alternating.save_stage

            def stop_after_epoch(root, name, *args, **kwargs):
                result = original_save(root, name, *args, **kwargs)
                if name == "pretrain_epoch_001":
                    raise InterruptedError("simulated warmup interruption")
                return result

            with patch.object(alternating, "save_stage", side_effect=stop_after_epoch):
                with self.assertRaises(InterruptedError):
                    alternating.study(market, interrupted, schedule, device="cpu", settings=settings)
            actual = alternating.study(market, interrupted, schedule, device="cpu", resume=True, settings=settings)
            self.assertEqual(expected, actual)
            relative = "replica_0/weights/cycle_001_adapt_00002/arbiter.pt"
            left = torch.load(complete / relative, weights_only=True)
            right = torch.load(interrupted / relative, weights_only=True)
            for network in ("actor", "critic"):
                for key in left[network]:
                    self.assertTrue(torch.equal(left[network][key], right[network][key]))


if __name__ == "__main__":
    unittest.main()
