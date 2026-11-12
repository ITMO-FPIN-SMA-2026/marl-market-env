"""Prepare, train and evaluate the plugin on the actual four team implementations."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .sources import IMPLEMENTATIONS, activate_sources, prepare_sources

COALITION_IMPLEMENTATION = 'coalition_policy:AdaptiveCoalitionPolicy'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def native_config(horizon, *, independent=False, checkpoint=None, state_dir=None, device="cpu", training=False):
    from social_market_env.config import config_from_dict
    arbiter = {"impl": "social_market_env.baselines.arbiters:AlwaysObserve"}
    if checkpoint:
        arbiter = {"impl": "transferred_ppo_gru:TeamPPOGRU", "kwargs": {
            "checkpoint_path": str(checkpoint), "device": device, "evaluation": "greedy",
            "state_dir": str(state_dir) if state_dir else None, "resume_state": False}}
    coalition = {"impl": "social_market_env.baselines.coalitions:NoCoalitions"} if independent else {
        "impl": COALITION_IMPLEMENTATION,
        "kwargs": {"horizon": horizon, "network_mode": "evolving"}}
    return config_from_dict({"name": "transferred_ppo_team", "horizon": horizon, "strict": True,
        "mode": "train" if training else "eval",
        "agents": {"count": 4, "default": {"impl": IMPLEMENTATIONS[0]},
                   "overrides": {i: {"impl": impl} for i, impl in enumerate(IMPLEMENTATIONS)}},
        "coalitions": {"policy": coalition}, "network": {"topology": "random", "mode": "evolving"},
        "arbiter": {"policy": arbiter, "min_fine": 1., "max_fine": 100.}})


def build(config, world_seed, sellers):
    from social_market_env.runner.experiment import build_environment
    from .seller_state import load_population
    env = build_environment(config=config, seed=world_seed)
    if sellers:
        env.agents = load_population(env.agents, sellers, config.market)
    return env


def agent_digest(agents):
    def plain(value):
        if isinstance(value, dict):
            return sorted((repr(k), plain(v)) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        return value
    names = ("q_table", "q", "visits", "context_q", "context_visits", "base_damping", "join_bias", "_ema_reward", "updates")
    data = {i: {key: plain(getattr(a, key)) for key in names if hasattr(a, key)} for i, a in agents.items()}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def network_digest(policy):
    from .convert import state_digest
    return (state_digest(policy.actor.state_dict()), state_digest(policy.critic.state_dict()), policy.updates)


def play(env, seed, *, training, log=None):
    before = agent_digest(env.agents)
    neural = network_digest(env.arbiter.policy) if hasattr(env.arbiter, "policy") else None
    env.reset(seed=seed, training=training)
    rows = []
    stream = gzip.open(log, "wt", encoding="utf-8") if log else None
    try:
        while not env.done:
            row = env.step().record
            if row["events"] or row["rejected_plans"]:
                raise RuntimeError("Native team code failed or proposed an invalid plan")
            rows.append(row)
            if stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    finally:
        if stream:
            stream.close()
    if not training:
        if before != agent_digest(env.agents):
            raise RuntimeError("Evaluation changed fitted team-agent parameters")
        if neural is not None and neural != network_digest(env.arbiter.policy):
            raise RuntimeError("Evaluation changed the transferred actor or critic")
    fines = [row for row in rows if row["arbiter"]["kind"] == "sanction"]
    return {"steps": len(rows), "welfare": float(np.mean([r["social_welfare"] for r in rows])),
        "reward": float(sum(r["arbiter"]["reward"] for r in rows)), "sanctions": len(fines),
        "false_positives": sum(r["arbiter"]["false_positive"] for r in rows),
        "false_negatives": sum(r["arbiter"]["false_negative"] for r in rows),
        "harmful_steps": sum(any(p["harmful"] for p in r["plans"]) for r in rows),
        "fine_total": float(sum(sum(r["fines"].values()) for r in rows)),
        "active_end": sum(rows[-1]["active"].values())}


def prepare(args):
    from social_market_env.contracts import ArbiterAction
    from .api import to_observation
    from .calibrate import fit_reference
    from .contracts import public_market
    from .convert import convert_checkpoint
    from .seller_state import save_population
    from .settings import Settings
    root = args.out
    if (root / "prepared.json").exists():
        raise FileExistsError("Use a new output directory to preserve the previous transfer")
    config = native_config(args.horizon)
    prototype = build(config, 900_000_001, args.seller_state_dir)
    initial = root / "sellers_initial"
    save_population(prototype.agents, initial)
    settings = Settings(**torch.load(args.source_checkpoint[0], map_location="cpu", weights_only=True)["settings"])

    class Recorder:
        def reset(self, **kwargs):
            self.rows = []

        def act(self, public):
            self.rows.append(asdict(to_observation(public)))
            return ArbiterAction()

        def observe_transition(self, **kwargs):
            pass

    datasets = []
    for offset, role in ((0, "reference_train_nonharmful"), (100_000, "calibration_nonharmful")):
        dataset = {"schema_version": 1, "market": asdict(public_market(config.market)),
                   "split_role": role, "episodes": []}
        for episode in range(args.calibration_episodes):
            seed = 900_100_000 + offset + episode
            env = build(native_config(args.horizon, independent=True), seed, initial)
            recorder = Recorder()
            env.arbiter = recorder
            play(env, seed + 1_000_000, training=False)
            dataset["episodes"].append({"episode_id": f"native:{seed}", "observations": recorder.rows})
        datasets.append(dataset)
    reference, calibration = fit_reference(*datasets, settings=settings)
    reference.save(root / "reference.json")
    write_json(root / "calibration_report.json", calibration)
    for name, data in zip(("reference_public.json", "calibration_public.json"), datasets):
        write_json(root / name, data)
    models = []
    for i, source in enumerate(args.source_checkpoint):
        models.append(convert_checkpoint(source, root / f"initial_model_{i}.pt", config.market, root / "reference.json"))
    import social_market_env.environment
    native_files = {str(p.relative_to(args.environment_repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (args.environment_repo / "src/social_market_env").rglob("*.py")}
    write_json(root / "prepared.json", {"environment": str(args.environment_repo),
        "environment_module": social_market_env.environment.__file__, "environment_files": native_files,
        "config": asdict(config), "models": models, "real_agents": IMPLEMENTATIONS,
        "seller_input": str(args.seller_state_dir) if args.seller_state_dir else "native defaults",
        "reward_changed": False, "agent_code_changed": False, "environment_code_changed": False})
    print(json.dumps({"prepared": str(root), "models": len(models), "agents": IMPLEMENTATIONS}), flush=True)


def aggregate(rows):
    n = sum(r["sanctions"] for r in rows)
    fp = sum(r["false_positives"] for r in rows)
    result = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    result.update(sanction_count=n, correct_sanctions=n - fp, precision=(n - fp) / n if n else None)
    return result


def train(args, prepared):
    from .seller_state import save_population
    root = args.out
    for model in range(len(prepared["models"])):
        directory = root / "train" / f"model_{model}"
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "games": 0, "checkpoint": str(Path(f"../../initial_model_{model}.pt")),
            "regulated_sellers": "../../sellers_initial", "baseline_sellers": "../../sellers_initial"}
        for game in range(progress["games"], args.games):
            seed = 910_000_000 + model * 100_000 + game
            checkpoint = (directory / progress["checkpoint"]).resolve()
            stage = directory / f"game_{game+1:05d}"
            attempt = 0
            while stage.exists():
                attempt += 1
                stage = directory / f"game_{game+1:05d}_attempt_{attempt}"
            stage.mkdir()
            config = native_config(prepared["config"]["horizon"], checkpoint=checkpoint,
                                   state_dir=directory / "arbiter", device=args.device, training=True)
            env = build(config, seed, directory / progress["regulated_sellers"])
            metrics = play(env, seed + 1_000_000, training=True, log=stage / "regulated.jsonl.gz")
            baseline = build(native_config(config.horizon, training=True), seed, directory / progress["baseline_sellers"])
            control = play(baseline, seed + 1_000_000, training=True, log=stage / "baseline.jsonl.gz")
            save_population(env.agents, stage / "regulated_sellers")
            save_population(baseline.agents, stage / "baseline_sellers")
            from os.path import relpath
            progress = {"games": game + 1, "checkpoint": relpath(env.arbiter.last_checkpoint, directory),
                "regulated_sellers": str((stage / "regulated_sellers").relative_to(directory)),
                "baseline_sellers": str((stage / "baseline_sellers").relative_to(directory)),
                "updates": env.arbiter.policy.updates}
            write_json(stage / "metrics.json", {"regulated": metrics, "observe": control})
            write_json(directory / "progress.json", progress)
            if (game + 1) % 8 == 0:
                print(f"native model={model} games={game+1} PPO updates={env.arbiter.policy.updates} "
                      f"fines={metrics['sanctions']} FP={metrics['false_positives']}", flush=True)


def evaluate(args, prepared):
    root = args.out
    path = root / ("initial_evaluation.json" if args.initial else "target_evaluation.json")
    base, number = path, 0
    while path.exists():
        number += 1
        path = base.with_name(base.stem + f"_{number:03d}" + base.suffix)
    comparisons, negative, deltas = [], [], []
    for model in range(len(prepared["models"])):
        if args.initial:
            checkpoint = root / f"initial_model_{model}.pt"
            regulated = baseline = root / "sellers_initial"
            seed_base = 905_000_000 + model * 100_000
        else:
            directory = root / "train" / f"model_{model}"
            progress = json.loads((directory / "progress.json").read_text())
            checkpoint = directory / progress["checkpoint"]
            regulated, baseline = directory / progress["regulated_sellers"], directory / progress["baseline_sellers"]
            seed_base = args.eval_seed + model * 100_000
        before, after, removed = [], [], []
        for episode in range(args.episodes):
            seed = seed_base + episode
            env = build(native_config(prepared["config"]["horizon"]), seed, baseline)
            before.append(play(env, seed + 1_000_000, training=False))
            cfg = native_config(env.config.horizon, checkpoint=checkpoint, device=args.device)
            after.append(play(build(cfg, seed, regulated), seed + 1_000_000, training=False))
            removed.append(play(build(native_config(cfg.horizon), seed, regulated), seed + 1_000_000, training=False))
            independent = native_config(cfg.horizon, independent=True, checkpoint=checkpoint, device=args.device)
            negative.append(play(build(independent, seed + 5_000_000, regulated), seed + 6_000_000, training=False))
        comparisons.append({"model": model, "checkpoint": str(checkpoint), "observe": before,
                            "ppo_gru": after, "same_regulated_sellers_without_arbiter": removed})
        deltas.append([b["welfare"] - a["welfare"] for a, b in zip(before, after)])
        print(f"native evaluation model={model} welfare gain={np.mean(deltas[-1]):.3f} "
              f"precision={aggregate(after)['precision']}", flush=True)
    rng = np.random.default_rng(91)
    draws = []
    for _ in range(3000):
        models = rng.integers(len(deltas), size=len(deltas))
        draws.append(np.mean([rng.choice(deltas[i], size=len(deltas[i]), replace=True).mean() for i in models]))
    all_before = [r for c in comparisons for r in c["observe"]]
    all_after = [r for c in comparisons for r in c["ppo_gru"]]
    all_removed = [r for c in comparisons for r in c["same_regulated_sellers_without_arbiter"]]
    report = {"initial_transfer": args.initial, "environment": prepared["environment_module"],
        "welfare_definition": "native consumer surplus", "reward": prepared["config"]["scoring"],
        "evaluation_seed": 905_000_000 if args.initial else args.eval_seed,
        "real_agents": IMPLEMENTATIONS, "models": len(comparisons), "worlds_per_model": args.episodes,
        "observe": aggregate(all_before), "ppo_gru": aggregate(all_after),
        "same_regulated_sellers_without_arbiter": aggregate(all_removed),
        "independent_negative_control": aggregate(negative),
        "welfare_gain": {"mean": float(np.mean(deltas)), "ci95": np.quantile(draws, [.025, .975]).tolist()},
        "details": comparisons}
    write_json(path, report)
    print(json.dumps({"report": str(path), **{k: report[k] for k in ("welfare_gain", "ppo_gru")}}), flush=True)


def main(argv=None):
    global IMPLEMENTATIONS, COALITION_IMPLEMENTATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--environment-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, action="append")
    parser.add_argument("--seller-state-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--calibration-episodes", type=int, default=8)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--eval-seed", type=int, default=930_000_000)
    args = parser.parse_args(argv)
    args.environment_repo, args.out = args.environment_repo.resolve(), args.out.resolve()
    if min(args.horizon, args.games, args.episodes, args.calibration_episodes) < 1:
        parser.error("Counts must be positive")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = args.out / "team_sources"
    if not sources.exists():
        prepare_sources(args.environment_repo, sources)
    activate_sources(args.environment_repo, sources)
    manifest = json.loads((sources / "sources.json").read_text())
    IMPLEMENTATIONS = tuple(manifest['implementations'])
    COALITION_IMPLEMENTATION = manifest.get('coalition_implementation', COALITION_IMPLEMENTATION)
    for name, digest in manifest["files"].items():
        if hashlib.sha256((sources / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Cached team implementation changed")
    if args.command == "prepare":
        if not args.source_checkpoint:
            parser.error("prepare requires --source-checkpoint")
        prepare(args)
    else:
        prepared = json.loads((args.out / "prepared.json").read_text())
        for name, digest in prepared["environment_files"].items():
            if hashlib.sha256((args.environment_repo / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Native environment changed after transfer preparation")
        (train if args.command == "train" else evaluate)(args, prepared)


if __name__ == "__main__":
    main()
