"""Counterfactual-feedback learning and paired co-adaptation experiments.

Privileged coalition labels are used ONLY here, as training targets/evaluation
labels. market_rl.sanction never receives an environment or a private record.
"""

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .agents import load_agents, load_auxiliary, save_agents
from .arbiter_scoring import ArbiterScoring, harmful_groups
from .environment import MarketEnv
from .sanction import LearnedArbiter, PublicEncoder, SanctionSettings, market_scales
from .train import run_episode


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_weights(directory, files, **metadata):
    directory = Path(directory)
    index_path = directory.parent / "index.json"
    records = json.loads(index_path.read_text()) if index_path.exists() else []
    records.append({**metadata, "files": [{"path": str(Path(file).relative_to(index_path.parent)),
                    "sha256": hashlib.sha256(Path(file).read_bytes()).hexdigest()} for file in files]})
    write_json(index_path, records)


class DatasetCollector:
    def __init__(self, config):
        self.encoder = PublicEncoder(market_scales(config))
        self.x, self.y = [], []

    def reset(self):
        self.encoder.reset()

    def __call__(self, public, action, info, verdict):
        features = self.encoder.encode(public)
        labels = np.zeros(4, dtype=np.float32)
        for group in harmful_groups(public, info):
            labels[list(group)] = 1
        active = np.array([p is not None for p in public.prices])
        self.x.extend(features[active])
        self.y.extend(labels[active])

    def arrays(self):
        return np.asarray(self.x, dtype=np.float32), np.asarray(self.y, dtype=np.float32)


def reseed(agents, coalition, seed):
    for i, agent in enumerate(agents):
        agent.rng = np.random.default_rng(seed + 1_000_000 + i)
    if coalition is not None:
        coalition.rng = np.random.default_rng(seed + 2_000_000)


def collect(config, agents, coalition, *, worlds, seed, solo=False):
    agents, coalition = copy.deepcopy(agents), copy.deepcopy(coalition)
    reseed(agents, coalition, seed)
    collector = DatasetCollector(config)
    for index in range(worlds):
        collector.reset()
        run_episode(MarketEnv(config), agents, seed + index, training=False, coalition_policy=coalition,
                    solo=solo, evaluator=collector)
    return collector.arrays()


def fit(config, x, y, *, memory=True, seed=0, epochs=30, initial=None, checkpoint_dir=None, resume_path=None):
    import torch
    from torch import nn

    torch.set_num_threads(1)
    if initial is not None and resume_path is not None:
        raise ValueError("Choose either previous-stage initialization or exact optimizer resume")
    if checkpoint_dir is not None and Path(checkpoint_dir).exists() and any(Path(checkpoint_dir).iterdir()):
        raise ValueError("Use a new checkpoint directory to preserve all previous weights")
    features = x if memory else x[:, :11]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(features.shape[1], 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    if initial is not None:
        with torch.no_grad():
            for layer, (weight, bias) in zip((m for m in model if isinstance(m, nn.Linear)), initial.weights):
                layer.weight.copy_(torch.from_numpy(weight))
                layer.bias.copy_(torch.from_numpy(bias))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    rng = np.random.default_rng(100 + seed)
    steps = 0
    policy = LearnedArbiter(config, SanctionSettings(memory=memory))
    policy.metadata = {"algorithm": "supervised harm prediction with validation-selected sanctions", "model_seed": seed,
                       "training_rows": len(y), "positive_fraction": float(y.mean()),
                       "initialization": "own_previous_round" if initial else "random",
                       "data_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
                       "private_inputs_at_inference": False}
    start_epoch = 0
    if resume_path is not None:
        restored = torch.load(resume_path, map_location="cpu", weights_only=True)
        if "model" not in restored or restored["metadata"]["data_sha256"] != policy.metadata["data_sha256"]:
            raise ValueError("Resume requires a training checkpoint and the identical dataset")
        model.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        rng.bit_generator.state = restored["shuffle_rng"]
        start_epoch, seed = restored["epoch"], restored["model_seed"]
        steps = restored["metadata"]["gradient_steps"]
        policy.metadata = dict(restored["metadata"], resumed_from=str(resume_path))

    def snapshot(epoch):
        policy.weights = [(layer.weight.detach().numpy().copy(), layer.bias.detach().numpy().copy())
                          for layer in model if isinstance(layer, nn.Linear)]
        policy.metadata.update(gradient_steps=steps, epochs=epoch, trained=bool(initial is not None or resume_path or epoch > 0))
        if checkpoint_dir is not None:
            directory = Path(checkpoint_dir)
            directory.mkdir(parents=True, exist_ok=True)
            readable = directory / f"epoch_{epoch:03d}.json"
            resumable = directory / f"epoch_{epoch:03d}.pt"
            policy.save(readable)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "model_seed": seed, "shuffle_rng": rng.bit_generator.state,
                        "metadata": policy.metadata.copy()}, resumable)
            record_weights(directory, (readable, resumable), stage=directory.name, epoch=epoch, seed=seed,
                           gradient_steps=steps, data_sha256=policy.metadata["data_sha256"])

    snapshot(start_epoch)
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 1024):
            indices = order[start:start + 1024]
            logits = model(torch.from_numpy(features[indices])).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[indices]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
        snapshot(epoch)
    return policy


def classification(policy, x, y, threshold=.7):
    p = policy.predict(x if policy.settings.memory else x[:, :11])
    selected = p >= threshold
    tp = float(y[selected].sum())
    fp = int(selected.sum()) - tp
    return {"threshold": threshold, "selected": int(selected.sum()),
            "precision": tp / selected.sum() if selected.any() else None,
            "recall": tp / max(1., float(y.sum())),
            "brier": float(np.mean((p - y)**2)),
            "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1-y) * np.log(np.clip(1-p, 1e-7, 1)))),
            "instant_advantage": .4 * tp - .6 * fp}


def adapt(config, agents, coalition, policy, *, episodes, seed, record=False, checkpoint_dir=None):
    agents, coalition, policy = copy.deepcopy(agents), copy.deepcopy(coalition), copy.deepcopy(policy)
    reseed(agents, coalition, seed)
    collector = DatasetCollector(config) if record else None
    rows = []
    if checkpoint_dir:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "market_before.json"
        save_agents(path, agents, config, coalition)
        files = [path]
        if policy is not None:
            policy.save(directory / "arbiter.json")
            files.append(directory / "arbiter.json")
        record_weights(directory, files, stage=directory.name, phase="before", seed=seed)
    for episode in range(episodes):
        if collector:
            collector.reset()
        rows.append(run_episode(MarketEnv(config), agents, seed + episode, training=True, epsilon_scale=.3,
                                coalition_policy=coalition, arbiter=policy, evaluator=collector))
    if checkpoint_dir:
        path = directory / "market_after.json"
        save_agents(path, agents, config, coalition)
        record_weights(directory, (path,), stage=directory.name, phase="after", seed=seed,
                       episodes=episodes, metrics=summary(rows))
    return agents, coalition, rows, collector.arrays() if collector else None


def evaluate_rows(config, agents, coalition, policy, *, worlds, seed, solo=False):
    agents, coalition, policy = copy.deepcopy(agents), copy.deepcopy(coalition), copy.deepcopy(policy)
    reseed(agents, coalition, seed)
    return [run_episode(MarketEnv(config), agents, seed + world, training=False, coalition_policy=coalition,
                        arbiter=policy, solo=solo) for world in range(worlds)]


def summary(rows):
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    sanctions = sum(row["sanctions"] for row in rows)
    fp = sum(row["false_positives"] for row in rows)
    result["precision"] = (sanctions - fp) / sanctions if sanctions else None
    result["sanction_count"] = int(sanctions)
    return result


def paired_bootstrap(deltas, seed=42):
    """Resample independent adaptation replicas, then paired worlds inside each.

    Individual rounds are never treated as independent observations.
    """
    deltas = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(3000):
        replicas = rng.integers(len(deltas), size=len(deltas))
        samples.append(np.mean([rng.choice(deltas[i], size=len(deltas[i]), replace=True).mean() for i in replicas]))
    return {"mean": float(deltas.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "replica_means": deltas.mean(axis=1).tolist()}


def refine(args):
    """Revalidate the actual refreshed weights before opening new test worlds."""
    config, agents = load_agents(args.market)
    coalition, solo = load_auxiliary(args.market)
    output = args.out
    output.mkdir(parents=True, exist_ok=False)
    models = [LearnedArbiter.load(path, config) for path in args.predictor]
    controls = (tuple((threshold,fine) for threshold in (.8,.85,.9) for fine in (25.,100.))
                if args.conservative else ((.65,100.),(.7,100.),(.75,100.),(.7,25.)))
    baseline_cache = {}
    development = []
    for index, model in enumerate(models):
        stage = output / "weights" / f"predictor_input{index}"
        stage.mkdir(parents=True)
        model.save(stage / "arbiter.json")
        record_weights(stage, (stage / "arbiter.json",), stage=stage.name,
                       source=str(args.predictor[index]), source_sha256=hashlib.sha256(args.predictor[index].read_bytes()).hexdigest())
        for replica in range(3):
            seed = args.domain + 300_000 + replica * 1000
            if replica not in baseline_cache:
                ba, bc, _, _ = adapt(config, agents, coalition, None, episodes=args.adapt_episodes, seed=seed,
                    checkpoint_dir=output / "weights" / f"development_{replica}_observe")
                baseline_cache[replica] = summary(evaluate_rows(config, ba, bc, None, worlds=30, seed=args.domain+400_000))
            for threshold, fine in controls:
                candidate = copy.deepcopy(model)
                candidate.settings = replace(candidate.settings, threshold=threshold, fine=fine,
                                             seller_budget=4*fine, episode_budget=8*fine)
                stage = output / "weights" / f"development_{replica}_model{index}_cut{threshold}_fine{fine}"
                aa, cc, _, _ = adapt(config, agents, coalition, candidate, episodes=args.adapt_episodes, seed=seed,
                                     checkpoint_dir=stage)
                result = summary(evaluate_rows(config, aa, cc, candidate, worlds=30, seed=args.domain+400_000))
                result.update(predictor=index, replica=replica, threshold=threshold, fine=fine,
                              cooldown=candidate.settings.cooldown,
                              welfare_gain=result["net_welfare"]-baseline_cache[replica]["net_welfare"])
                development.append(result)
                write_json(output / "development.json", development)
                print(f"Validation model={index} replica={replica} cut={threshold} fine={fine} "
                      f"gain={result['welfare_gain']:.3f} precision={result['precision']}", flush=True)
    choices = []
    for index in range(len(models)):
        for threshold, fine in controls:
            group = [r for r in development if (r["predictor"],r["threshold"],r["fine"]) == (index,threshold,fine)]
            if not group:
                continue
            n = sum(r["sanction_count"] for r in group)
            choices.append({"predictor": index, "threshold": threshold, "fine": fine,
                            "cooldown":models[index].settings.cooldown,
                            "precision": sum((r["precision"] or 0)*r["sanction_count"] for r in group) / max(1,n),
                            "gain": float(np.mean([r["welfare_gain"] for r in group])), "sanctions": n})
    admissible = [r for r in choices if r["precision"] >= args.precision_floor and r["gain"] > 0 and r["sanctions"] >= 10]
    selected = max(admissible or choices, key=lambda r: (r["precision"] >= args.precision_floor, r["gain"]))
    model = copy.deepcopy(models[selected["predictor"]])
    model.settings = replace(model.settings, threshold=selected["threshold"], fine=selected["fine"],
                             seller_budget=4*selected["fine"], episode_budget=8*selected["fine"])
    selected["cooldown"] = model.settings.cooldown
    model.metadata.update(selected_on="independent development populations, before final test",
                          source_checkpoint=str(args.predictor[selected["predictor"]]))
    model.save(output/"arbiter.json")
    write_json(output/"selection.json", {"candidates": choices, "selected": selected, "admissible": bool(admissible)})
    tests, differences = [], []
    for replica in range(args.replicas):
        seed = args.domain + 700_000 + replica*1000
        ba, bc, _, _ = adapt(config, agents, coalition, None, episodes=args.adapt_episodes, seed=seed,
                            checkpoint_dir=output/"weights"/f"test_{replica}_observe")
        aa, cc, _, _ = adapt(config, agents, coalition, model, episodes=args.adapt_episodes, seed=seed,
                            checkpoint_dir=output/"weights"/f"test_{replica}_arbiter")
        before = evaluate_rows(config, ba, bc, None, worlds=args.test_worlds, seed=args.domain+800_000)
        after = evaluate_rows(config, aa, cc, model, worlds=args.test_worlds, seed=args.domain+800_000)
        differences.append([b["net_welfare"]-a["net_welfare"] for a,b in zip(before,after)])
        tests.append({"replica": replica, "observe": before, "arbiter": after})
        save_agents(output/f"market_{replica}.json", aa, config, cc, solo)
        print(f"Held-out replica={replica} gain={np.mean(differences[-1]):.3f} precision={summary(after)['precision']}",flush=True)
        write_json(output/"test_worlds.json",tests)
    report = {"net_welfare_gain": paired_bootstrap(differences),
              "observe": summary([r for world in tests for r in world["observe"]]),
              "arbiter": summary([r for world in tests for r in world["arbiter"]]),
              "selection": selected, "development_admissible": bool(admissible),
              "replicas": args.replicas, "worlds_per_replica": args.test_worlds,
              "economic_rules_unchanged": True, "agents_observe_own_last_fine": True,
              "test_domains": {"adaptation":args.domain+700_000,"evaluation":args.domain+800_000}}
    report["market"] = asdict(config)
    report["precision_floor"] = args.precision_floor
    report["independent_negative_control"] = summary(evaluate_rows(config, solo, None, model,
        worlds=100, seed=args.domain+900_000, solo=True))
    report["qualification"] = {
        "positive_welfare_interval": report["net_welfare_gain"]["ci95"][0] > 0,
        "sanction_precision_passed": (report["arbiter"]["precision"] is not None
                                       and report["arbiter"]["precision"] >= args.precision_floor),
    }
    write_json(output/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)


def experiment(args):
    config, agents = load_agents(args.market)
    coalitions, solo = load_auxiliary(args.market)
    output = args.out
    output.mkdir(parents=True, exist_ok=False)
    print("Collecting disjoint public training / validation histories", flush=True)
    initial = collect(config, agents, coalitions, worlds=args.train_worlds, seed=100_000)
    negative = collect(config, solo, None, worlds=max(10, args.train_worlds // 5), seed=120_000, solo=True)
    x, y = (np.concatenate(parts) for parts in zip(initial, negative))
    validation = collect(config, agents, coalitions, worlds=80, seed=200_000)
    hard_negative = collect(config, solo, None, worlds=20, seed=220_000, solo=True)
    vx, vy = (np.concatenate(parts) for parts in zip(validation, hard_negative))
    np.savez_compressed(output / "train.npz", features=x, labels=y)
    np.savez_compressed(output / "validation.npz", features=vx, labels=vy)
    ablations = []
    candidates = []
    for memory in (False, True):
        for seed in range(3):
            model = fit(config, x, y, memory=memory, seed=seed, epochs=args.epochs,
                        checkpoint_dir=output / "weights" / f"predictor_{int(memory)}_seed{seed}")
            metrics = classification(model, vx, vy)
            row = {"memory": memory, "seed": seed, **metrics}
            ablations.append(row)
            candidates.append((metrics["instant_advantage"], model))
            model.save(output / f"predictor_{int(memory)}_{seed}.json")
            print("Predictor", row, flush=True)
    write_json(output / "ablations.json", ablations)
    _, model = max(candidates, key=lambda pair: pair[0])
    model.save(output / "round0.json")

    # Choose budget/threshold on development populations, never on final worlds.
    development = []
    feedback = []
    for replica in range(3):
        baseline_agents, baseline_coalition, _, _ = adapt(config, agents, coalitions, None,
            episodes=args.adapt_episodes, seed=300_000 + replica * 1000,
            checkpoint_dir=output / "weights" / f"development_{replica}_observe")
        baseline = summary(evaluate_rows(config, baseline_agents, baseline_coalition, None, worlds=30, seed=400_000))
        for threshold, fine in ((.65, 100.), (.7, 100.), (.75, 100.), (.7, 25.)):
            candidate = copy.deepcopy(model)
            candidate.settings = replace(candidate.settings, threshold=threshold, fine=fine,
                                          seller_budget=fine*4, episode_budget=fine*8)
            aa, cc, _, data = adapt(config, agents, coalitions, candidate, episodes=args.adapt_episodes,
                                    seed=300_000 + replica * 1000, record=True,
                                    checkpoint_dir=output / "weights" / f"development_{replica}_cut{threshold}_fine{fine}")
            row = summary(evaluate_rows(config, aa, cc, candidate, worlds=30, seed=400_000))
            row.update(replica=replica, threshold=threshold, fine=fine,
                       welfare_gain=row["net_welfare"] - baseline["net_welfare"])
            development.append(row)
            feedback.append(data)
            print(f"Adapt replica={replica} cut={threshold} fine={fine} gain={row['welfare_gain']:.3f} "
                  f"precision={row['precision']}", flush=True)
            write_json(output / "development.json", development)

    # Refresh on the new learning-agent distribution. Same predictor, no
    # additional network heads or privileged inference features.
    fx, fy = (np.concatenate([base, *(part[k] for part in feedback)]) for k, base in enumerate((x, y)))
    model = fit(config, fx, fy, memory=model.settings.memory, seed=model.metadata["model_seed"],
                epochs=args.epochs, initial=model, checkpoint_dir=output / "weights" / "predictor_round1")
    model.save(output / "round1.json")
    validation_args = copy.copy(args)
    validation_args.predictor = [output / "round1.json"]
    validation_args.out = output / "validated"
    refine(validation_args)
    selected = LearnedArbiter.load(validation_args.out / "arbiter.json", config)
    selected.save(output / "arbiter.json")
    report = json.loads((validation_args.out / "report.json").read_text())
    report["validation_predictor"] = classification(selected, vx, vy, selected.settings.threshold)
    write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, default=Path("runs/check/agents.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-worlds", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapt-episodes", type=int, default=100)
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--test-worlds", type=int, default=50)
    parser.add_argument("--predictor", nargs="+", type=Path, help="Validate existing stage weights; preserve their training")
    parser.add_argument("--domain", type=int, default=2_000_000, help="Disjoint refinement/evaluation seed domain")
    parser.add_argument("--conservative", action="store_true", help="Validate stricter confidence thresholds")
    parser.add_argument("--precision-floor", type=float, default=.8)
    args = parser.parse_args()
    if min(args.train_worlds, args.epochs, args.adapt_episodes, args.replicas, args.test_worlds) < 1:
        parser.error("Experiment sizes must be positive")
    if not 0 < args.precision_floor <= 1:
        parser.error("Precision floor must be in (0,1]")
    if args.predictor:
        refine(args)
    else:
        experiment(args)


if __name__ == "__main__":
    main()
