"""Alternating PPO/GRU and seller learning, with a matched observe-only branch."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..agents import load_agents, load_auxiliary, save_agents
from ..environment import MarketEnv
from ..learn_arbiter import (evaluate_rows, paired_bootstrap, record_weights,
                            reseed, summary, write_json)
from ..train import run_episode
from .adapter import ArbiterAdapter
from .calibrate import fit_reference
from .cli import collect_solo, model_digest
from .contracts import public_market
from .policy import PPOGRUArbiterV2, resolve_device
from .reference import PublicReference
from .settings import Settings


@dataclass(frozen=True)
class Schedule:
    cycles: int = 8
    ppo_episodes: int = 32
    adapt_episodes: int = 32
    reference_episodes: int = 8
    validation_worlds: int = 20
    test_worlds: int = 50
    replicas: int = 3
    seed: int = 30_000_000
    epsilon_scale: float = .3
    precision_floor: float = .9
    adaptation_checkpoint_every: int = 16
    learn_during_ppo: bool = False
    pretrain_epochs: int = 0
    pretrain_worlds: int = 400
    pretrain_validation_worlds: int = 80

    def validate(self, settings):
        counts = (self.cycles, self.ppo_episodes, self.adapt_episodes,
                  self.reference_episodes, self.validation_worlds,
                  self.test_worlds, self.replicas, self.adaptation_checkpoint_every,
                  self.pretrain_worlds, self.pretrain_validation_worlds)
        if any(type(n) is not int or n < 1 for n in counts):
            raise ValueError("All episode, cycle and replica counts must be positive integers")
        if self.ppo_episodes % settings.batch_episodes:
            raise ValueError("Each PPO stage must contain complete PPO batches")
        if type(self.pretrain_epochs) is not int or self.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be a nonnegative integer")
        if self.cycles * (self.ppo_episodes + self.adapt_episodes) >= 100_000:
            raise ValueError("Training stages exceed the reserved seed block")
        if max(self.reference_episodes, self.validation_worlds, self.test_worlds,
               self.pretrain_worlds, self.pretrain_validation_worlds) >= 100_000 or self.replicas > 100:
            raise ValueError("Requested scenarios exceed disjoint seed blocks")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.learn_during_ppo) is not bool:
            raise ValueError("learn_during_ppo must be a boolean")
        if not 0 < self.epsilon_scale <= 1 or not 0 <= self.precision_floor <= 1:
            raise ValueError("Invalid exploration scale or precision floor")

    def domain(self, phase, replica=0):
        return self.seed + phase * 10_000_000 + replica * 100_000


@dataclass
class Population:
    agents: list
    coalition: object
    solo: list

    @classmethod
    def load(cls, path):
        _, agents = load_agents(path)
        coalition, solo = load_auxiliary(path)
        if coalition is None or solo is None:
            raise ValueError("A checkpoint must contain coalition and solo policies")
        return cls(agents, coalition, solo)

    def save(self, path, config):
        save_agents(path, self.agents, config, self.coalition, self.solo)


def save_stage(root, name, config, regulated, baseline, adapter, state):
    """Immutable complete stage; the small progress pointer is committed last."""
    stage = root / "weights" / name
    stage.mkdir(parents=True, exist_ok=False)
    adapter.save_checkpoint(stage / "arbiter.pt")
    regulated.save(stage / "regulated.json", config)
    baseline.save(stage / "baseline.json", config)
    write_json(stage / "state.json", state)
    record_weights(stage, tuple(stage / name for name in
                  ("arbiter.pt", "regulated.json", "baseline.json", "state.json")),
                  stage=name, **state, ppo_updates=adapter.policy.updates)
    write_json(root / "progress.json", {**state, "checkpoint": str(stage.relative_to(root))})
    return str(stage.relative_to(root))


def adapt_population(config, population, adapter, seed, epsilon_scale):
    """PPO stays frozen; both financial learner levels update their own rewards."""
    run_episode(MarketEnv(config), population.solo, seed, training=True,
                epsilon_scale=epsilon_scale, solo=True, arbiter=adapter)
    return run_episode(MarketEnv(config), population.agents, seed, training=True,
                       epsilon_scale=epsilon_scale, coalition_policy=population.coalition,
                       arbiter=adapter)


def candidate_result(before, after, cycle, checkpoint, floor):
    control, regulated = summary(before), summary(after)
    gain = regulated["net_welfare"] - control["net_welfare"]
    precision = regulated["precision"]
    admissible = (gain > 0 and precision is not None and precision >= floor
                  and regulated["sanction_count"] >= 10)
    return {"cycle": cycle, "checkpoint": checkpoint, "gain": gain,
            "precision": precision, "sanction_count": regulated["sanction_count"],
            "admissible": admissible, "observe": control, "ppo_gru": regulated}


def train_replica(root, market, config, reference, schedule, settings, replica, device):
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        stage = root / progress["checkpoint"]
        regulated = Population.load(stage / "regulated.json")
        baseline = Population.load(stage / "baseline.json")
        adapter = ArbiterAdapter.load(stage / "arbiter.pt", config, device=device)
    else:
        regulated, baseline = Population.load(market), Population.load(market)
        policy = PPOGRUArbiterV2(market=asdict(public_market(config)), settings=asdict(settings),
                                 model_seed=replica, device=device)
        policy.reference = reference
        policy.memory.reference = reference
        policy.metadata.update(algorithm="PPO+GRU", starting_weights="random",
                               protocol="learning_recipients" if schedule.learn_during_ppo else "alternating",
                               replica=replica)
        adapter = ArbiterAdapter(policy, config)
        progress = {"cycle": 1, "phase": "pretrain" if schedule.pretrain_epochs else "ppo",
                    "ppo_done": 0, "adapt_done": 0}
        save_stage(root, "initial", config, regulated, baseline, adapter, progress)
        progress = json.loads(progress_path.read_text())
    policy = adapter.policy
    policy.evaluation = "greedy"
    if progress["phase"] == "pretrain":
        from .pretrain import pretrain_actor

        def save_epoch(epoch):
            state = {"cycle": 1, "phase": "pretrain", "ppo_done": 0,
                     "adapt_done": 0, "pretrain_epoch": epoch}
            save_stage(root, f"pretrain_epoch_{epoch:03d}", config, regulated, baseline, adapter, state)

        pretrain_actor(adapter, root, root.parent / "pretrain_train.npz",
                       root.parent / "pretrain_validation.npz", epochs=schedule.pretrain_epochs,
                       replica=replica, precision_floor=schedule.precision_floor, save_epoch=save_epoch)
        progress = {"cycle": 1, "phase": "ppo", "ppo_done": 0, "adapt_done": 0}
        save_stage(root, "pretrain_selected", config, regulated, baseline, adapter, progress)
        progress = json.loads(progress_path.read_text())
    selection_path = root / "validation.json"
    candidates = json.loads(selection_path.read_text()) if selection_path.exists() else []
    training_seed = schedule.domain(1, replica)
    while progress["cycle"] <= schedule.cycles:
        cycle = progress["cycle"]
        offset = (cycle - 1) * (schedule.ppo_episodes + schedule.adapt_episodes)
        if progress["phase"] == "ppo":
            if progress["ppo_done"] == 0:
                reseed(regulated.agents, regulated.coalition, training_seed + offset)
                if schedule.learn_during_ppo:
                    reseed(baseline.agents, baseline.coalition, training_seed + offset)
            for episode in range(progress["ppo_done"], schedule.ppo_episodes):
                previous_updates = policy.updates
                run_episode(MarketEnv(config), regulated.agents, training_seed + offset + episode,
                            training=schedule.learn_during_ppo, epsilon_scale=schedule.epsilon_scale,
                            coalition_policy=regulated.coalition,
                            arbiter=adapter, arbiter_training=True)
                if schedule.learn_during_ppo:
                    run_episode(MarketEnv(config), baseline.agents, training_seed + offset + episode,
                                training=True, epsilon_scale=schedule.epsilon_scale,
                                coalition_policy=baseline.coalition)
                if policy.updates != previous_updates:
                    phase = "adapt" if episode + 1 == schedule.ppo_episodes else "ppo"
                    progress = {"cycle": cycle, "phase": phase,
                                "ppo_done": episode + 1, "adapt_done": 0}
                    save_stage(root, f"cycle_{cycle:03d}_ppo_{policy.updates:04d}", config,
                               regulated, baseline, adapter, progress)
                    print(f"replica={replica} cycle={cycle} PPO updates={policy.updates} "
                          f"device={policy.device}", flush=True)
        if progress["phase"] == "adapt":
            seed = training_seed + offset + schedule.ppo_episodes
            if progress["adapt_done"] == 0:
                for population in (regulated, baseline):
                    reseed(population.agents, population.coalition, seed)
                    reseed(population.solo, None, seed + 50_000)
            actor_before = model_digest(policy.actor)
            critic_before = model_digest(policy.critic)
            for episode in range(progress["adapt_done"], schedule.adapt_episodes):
                adapt_population(config, regulated, adapter, seed + episode, schedule.epsilon_scale)
                adapt_population(config, baseline, None, seed + episode, schedule.epsilon_scale)
                if ((episode + 1) % schedule.adaptation_checkpoint_every == 0
                        or episode + 1 == schedule.adapt_episodes):
                    phase = "validate" if episode + 1 == schedule.adapt_episodes else "adapt"
                    progress = {"cycle": cycle, "phase": phase,
                                "ppo_done": schedule.ppo_episodes, "adapt_done": episode + 1}
                    save_stage(root, f"cycle_{cycle:03d}_adapt_{episode+1:05d}", config,
                               regulated, baseline, adapter, progress)
            if model_digest(policy.actor) != actor_before or model_digest(policy.critic) != critic_before:
                raise RuntimeError("PPO weights changed during seller-only adaptation")
        if progress["phase"] == "validate":
            checkpoint = json.loads(progress_path.read_text())["checkpoint"]
            before = evaluate_rows(config, baseline.agents, baseline.coalition, None,
                worlds=schedule.validation_worlds, seed=schedule.domain(2, replica))
            after = evaluate_rows(config, regulated.agents, regulated.coalition, adapter,
                worlds=schedule.validation_worlds, seed=schedule.domain(2, replica))
            result = candidate_result(before, after, cycle, checkpoint, schedule.precision_floor)
            candidates = [row for row in candidates if row["cycle"] != cycle] + [result]
            write_json(selection_path, candidates)
            progress = {"cycle": cycle + 1, "phase": "ppo", "ppo_done": 0, "adapt_done": 0,
                        "checkpoint": checkpoint}
            write_json(progress_path, progress)
            print(f"replica={replica} cycle={cycle} validation gain={result['gain']:.3f} "
                  f"precision={result['precision']} sanctions={result['sanction_count']}", flush=True)
    selected = max(candidates, key=lambda row: (row["admissible"], row["gain"]))
    write_json(root / "selection.json", {"selected": selected,
               "rule": "positive validation welfare gain, precision floor, >=10 sanctions; otherwise unqualified fallback",
               "final_test_used_for_selection": False})
    return selected


def study(market, out, schedule, *, device="auto", resume=False, settings=None, pretrain_sources=None):
    settings = settings or Settings(screening_mode="learned")
    schedule.validate(settings)
    device = resolve_device(device)
    config, _ = load_agents(market)
    if config.horizon < settings.min_history:
        raise ValueError("Market horizon is shorter than the required history")
    source = Population.load(market)
    specification = {"schedule": asdict(schedule), "settings": asdict(settings),
                     "market": asdict(config), "market_sha256": hashlib.sha256(Path(market).read_bytes()).hexdigest(),
                     "model_sha256": hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
                     "evaluation": "greedy", "reward_changed": False}
    if schedule.pretrain_epochs:
        pretrain_sources = pretrain_sources or [market]
        specification["pretrain_sources"] = [{"path": str(p),
            "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in pretrain_sources]
    specification = json.loads(json.dumps(specification))
    out = Path(out)
    if resume:
        saved = json.loads((out / "run_config.json").read_text())
        saved["schedule"].setdefault("learn_during_ppo", False)
        for name in ("pretrain_epochs", "pretrain_worlds", "pretrain_validation_worlds"):
            saved["schedule"].setdefault(name, Schedule.__dataclass_fields__[name].default)
        if saved != specification:
            raise ValueError("Resume requires identical training configuration, market and model architecture")
        if (out / "report.json").exists():
            return json.loads((out / "report.json").read_text())
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "run_config.json", specification)
    write_json(out / "device.json", {"device": str(device), "torch": torch.__version__})
    reference_path = out / "reference.json"
    if reference_path.exists():
        reference = PublicReference.load(reference_path)
    else:
        reference_data, _ = collect_solo(config, source.solo, seed=schedule.domain(0),
            episodes=schedule.reference_episodes, role="reference_train_nonharmful")
        calibration_data, _ = collect_solo(config, source.solo, seed=schedule.domain(0) + 100_000,
            episodes=schedule.reference_episodes, role="calibration_nonharmful")
        reference, calibration = fit_reference(reference_data, calibration_data, settings=settings)
        write_json(out / "reference_public.json", reference_data)
        write_json(out / "calibration_public.json", calibration_data)
        write_json(out / "calibration_report.json", calibration)
        reference.save(reference_path)
    if schedule.pretrain_epochs:
        from .pretrain import collect_dataset

        for name, worlds, seed in (("train", schedule.pretrain_worlds, schedule.domain(0) + 200_000),
                                  ("validation", schedule.pretrain_validation_worlds, schedule.domain(0) + 300_000)):
            path = out / f"pretrain_{name}.npz"
            if not path.exists():
                print(f"Collecting public actor-pretraining {name}: {worlds} worlds", flush=True)
                collect_dataset(config, pretrain_sources, settings, reference, worlds=worlds, seed=seed, path=path)
    print(f"algorithm=PPO+GRU device={device} replicas={schedule.replicas}", flush=True)
    selected = [train_replica(out / f"replica_{i}", market, config, reference,
                schedule, settings, i, device) for i in range(schedule.replicas)]
    # Open final scenarios only after ALL training/validation choices are fixed.
    worlds, deltas, negative = [], [], []
    for i, selection in enumerate(selected):
        stage = out / f"replica_{i}" / selection["checkpoint"]
        regulated, baseline = Population.load(stage / "regulated.json"), Population.load(stage / "baseline.json")
        adapter = ArbiterAdapter.load(stage / "arbiter.pt", config, device=device)
        before = evaluate_rows(config, baseline.agents, baseline.coalition, None,
            worlds=schedule.test_worlds, seed=schedule.domain(3, i))
        after = evaluate_rows(config, regulated.agents, regulated.coalition, adapter,
            worlds=schedule.test_worlds, seed=schedule.domain(3, i))
        negative += evaluate_rows(config, regulated.solo, None, adapter, solo=True,
            worlds=schedule.test_worlds, seed=schedule.domain(4, i))
        worlds.append({"replica": i, "observe": before, "ppo_gru": after})
        deltas.append([b["net_welfare"] - a["net_welfare"] for a, b in zip(before, after)])
        write_json(out / "test_worlds.json", worlds)
        print(f"held-out replica={i} gain={np.mean(deltas[-1]):.3f} "
              f"sanctions={summary(after)['sanction_count']}", flush=True)
    report = {"algorithm": "PPO+GRU", "device": str(device), "market": asdict(config),
              "schedule": asdict(schedule), "selections": selected,
              "net_welfare_gain": paired_bootstrap(deltas),
              "observe": summary([r for world in worlds for r in world["observe"]]),
              "ppo_gru": summary([r for world in worlds for r in world["ppo_gru"]]),
              "independent_negative_control": summary(negative)}
    precision = report["ppo_gru"]["precision"]
    report["qualification"] = {
        "independent_training_replicas": schedule.replicas >= 3,
        "validation_admissible": all(r["admissible"] for r in selected),
        "positive_welfare_interval": report["net_welfare_gain"]["ci95"][0] > 0,
        "sanction_precision_passed": precision is not None and precision >= schedule.precision_floor,
    }
    report["qualified"] = all(report["qualification"].values())
    write_json(out / "report.json", report)
    print(json.dumps({"net_welfare_gain": report["net_welfare_gain"],
                      "qualification": report["qualification"], "qualified": report["qualified"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initial-fine-probability", type=float, default=Settings().initial_fine_probability)
    parser.add_argument("--pretrain-source", action="append", type=Path,
                        help="Additional frozen market source for supervised public-history collection")
    for name, field in Schedule.__dataclass_fields__.items():
        flag = "--" + name.replace("_", "-")
        if type(field.default) is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
        else:
            parser.add_argument(flag, type=type(field.default), default=field.default)
    args = parser.parse_args()
    torch.set_num_threads(1)
    schedule = Schedule(**{key: getattr(args, key) for key in Schedule.__dataclass_fields__})
    settings = Settings(screening_mode="learned", initial_fine_probability=args.initial_fine_probability)
    study(args.market, args.out, schedule, device=args.device, resume=args.resume, settings=settings,
          pretrain_sources=args.pretrain_source)


if __name__ == "__main__":
    main()
