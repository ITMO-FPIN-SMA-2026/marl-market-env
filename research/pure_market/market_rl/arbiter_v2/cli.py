"""Train the adapted original PPO+GRU from random weights on this market."""

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..agents import load_agents, load_auxiliary, save_agents
from ..arbiter_scoring import ArbiterScoring
from ..environment import MarketEnv
from ..train import evaluate, run_episode
from ..learn_arbiter import record_weights
from .adapter import ArbiterAdapter, to_observation
from .calibrate import fit_reference
from .contracts import public_market
from .policy import PPOGRUArbiterV2
from .settings import Settings


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def collect_solo(config, solo_agents, *, seed, episodes, role):
    """Frozen learned solo policies, disjoint seeds, public data only."""
    agents = copy.deepcopy(solo_agents)
    for i, agent in enumerate(agents):
        agent.rng = np.random.default_rng(seed + 1_000_000 + i)
    data = {"schema_version": 1, "market": asdict(public_market(config)), "split_role": role, "episodes": []}
    welfare = []
    for index in range(episodes):
        observations = []
        def record(public):
            obs = to_observation(public, public_market(config))
            observations.append(asdict(obs))
            welfare.append(public.social_welfare)

        run_episode(MarketEnv(config), agents, seed + index, training=False, solo=True, arbiter=record)
        data["episodes"].append({"episode_id": f"solo:{seed + index}", "observations": observations})
    return data, float(np.mean(welfare))


def train(args, config, agents, coalitions, solo):
    if solo is None:
        raise ValueError("Market checkpoint needs the learned solo reference")
    if config.horizon < 5:
        raise ValueError("Use a market horizon of at least 5 rounds for the public history")
    if args.resume:
        adapter = ArbiterAdapter.load(args.resume, config, device=args.device)
        if args.screening and args.screening != adapter.policy.settings.screening_mode:
            raise ValueError("Cannot change screening masks when resuming PPO")
        if adapter.policy.metadata.get("market_checkpoint_sha256") != hashlib.sha256(args.market.read_bytes()).hexdigest():
            raise ValueError("Resume needs the same frozen market checkpoint")
        for agent, state in zip(agents, adapter.policy.metadata["seller_rng"]):
            agent.rng.bit_generator.state = state
        coalitions.rng.bit_generator.state = adapter.policy.metadata["coalition_rng"]
        offset = adapter.policy.episodes_seen
        seed = adapter.policy.metadata["training_seed"]
    else:
        settings = Settings(screening_mode=args.screening or "learned")
        reference_data, target = collect_solo(config, solo, seed=10_000, episodes=args.reference_episodes,
                                             role="reference_train_nonharmful")
        calibration_data, _ = collect_solo(config, solo, seed=20_000, episodes=args.reference_episodes,
                                          role="calibration_nonharmful")
        reference, report = fit_reference(reference_data, calibration_data, settings=settings)
        reference.save(args.out / "reference.json")
        write_json(args.out / "calibration_report.json", report)
        write_json(args.out / "reference_public.json", reference_data)
        write_json(args.out / "calibration_public.json", calibration_data)
        policy = PPOGRUArbiterV2(market=asdict(public_market(config)), settings=asdict(settings),
                                 reference_path=args.out / "reference.json", model_seed=args.model_seed,
                                 device=args.device)
        adapter = ArbiterAdapter(policy, config, ArbiterScoring(target_welfare=config.welfare_bound))
        policy.metadata = {"starting_weights": "random", "training_seed": args.seed,
                           "market_checkpoint_sha256": hashlib.sha256(args.market.read_bytes()).hexdigest()}
        offset, seed = 0, args.seed
    policy = adapter.policy
    policy.metadata["device"] = str(policy.device)
    if seed + offset + args.episodes > 40_000:
        raise ValueError("Training seeds must stay below 40000, reserved for evaluation")
    if policy.updates == 0 and len(policy.completed) + args.episodes < policy.settings.batch_episodes:
        raise ValueError(f"Need at least {policy.settings.batch_episodes - len(policy.completed)} episodes for a PPO update")
    initial_actor = model_digest(policy.actor)
    stage = args.out / "weights" / "ppo"
    stage.mkdir(parents=True, exist_ok=False)
    market_stage = args.out / "weights" / "market"
    market_stage.mkdir()
    market_path = market_stage / "frozen.json"
    save_agents(market_path, agents, config, coalitions, solo)
    record_weights(market_stage, (market_path,), stage="frozen_market", seed=seed)

    def checkpoint(episode):
        policy.metadata["seller_rng"] = [agent.rng.bit_generator.state for agent in agents]
        policy.metadata["coalition_rng"] = coalitions.rng.bit_generator.state
        path = stage / f"episode_{episode:05d}_update_{policy.updates:04d}.pt"
        if path.exists():
            raise FileExistsError(path)
        adapter.save_checkpoint(path)
        record_weights(stage, (path,), stage="ppo", episode=episode, seed=seed,
                       ppo_updates=policy.updates, device=str(policy.device))

    policy.metadata["seller_rng"] = [agent.rng.bit_generator.state for agent in agents]
    policy.metadata["coalition_rng"] = coalitions.rng.bit_generator.state
    adapter.save_checkpoint(args.out / "initial.pt")
    checkpoint(offset)
    print(f"device={policy.device} algorithm=PPO+GRU starting_updates={policy.updates}", flush=True)
    with (args.out / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for episode in range(args.episodes):
            metrics = run_episode(MarketEnv(config), agents, seed + offset + episode, training=False,
                                  coalition_policy=coalitions, arbiter=adapter, arbiter_training=True)
            row = {"episode": offset + episode + 1, "ppo_updates": policy.updates, **metrics}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            checkpoint(offset + episode + 1)
            if (episode + 1) % policy.settings.batch_episodes == 0:
                print(f"episode={row['episode']} updates={policy.updates} "
                      f"sanctions={metrics['sanctions']:.0f} reward={metrics['arbiter_reward']:.2f}", flush=True)
    policy.metadata["seller_rng"] = [agent.rng.bit_generator.state for agent in agents]
    policy.metadata["coalition_rng"] = coalitions.rng.bit_generator.state
    adapter.save_checkpoint(args.out / "latest.pt")
    comparison = {
        "observe": evaluate(config, agents, coalitions, args.eval_episodes, seed=40_000,
                            arbiter_scoring=adapter.scoring),
        "ppo_gru": evaluate(config, agents, coalitions, args.eval_episodes, seed=40_000, arbiter=adapter),
    }
    write_json(args.out / "evaluation.json", comparison)
    summary = {"starting_weights": policy.metadata["starting_weights"], "old_checkpoint_loaded": False,
               "episodes_seen": policy.episodes_seen, "ppo_updates": policy.updates,
               "initial_actor_sha256": initial_actor, "final_actor_sha256": model_digest(policy.actor),
               "actor_changed": initial_actor != model_digest(policy.actor),
               "screening_mode": policy.settings.screening_mode, "reward": asdict(adapter.scoring),
               "settings": asdict(policy.settings), "market": asdict(public_market(config)),
               "evaluation_episodes": args.eval_episodes, "losses": policy.last_losses,
               "algorithm": "PPO+GRU", "device": str(policy.device),
               "sellers_trained": False, "initialization_checkpoint": str(args.resume) if args.resume else None}
    write_json(args.out / "summary.json", summary)
    print(json.dumps({"ppo_updates": policy.updates, "actor_changed": summary["actor_changed"],
                      "evaluation": comparison}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    evaluation = commands.add_parser("evaluate")
    for command in (training, evaluation):
        command.add_argument("--market", type=Path, required=True, help="Market agents.json")
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    training.add_argument("--episodes", type=int, default=64)
    training.add_argument("--reference-episodes", type=int, default=8)
    training.add_argument("--eval-episodes", type=int, default=10)
    training.add_argument("--seed", type=int, default=30_000)
    training.add_argument("--model-seed", type=int, default=0)
    training.add_argument("--screening", choices=["learned", "economic", "calibrated"])
    training.add_argument("--resume", type=Path)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--episodes", type=int, default=20)
    evaluation.add_argument("--seed", type=int, default=40_000)
    args = parser.parse_args()
    if args.episodes < 1 or (args.command == "train" and min(args.reference_episodes, args.eval_episodes) < 1):
        parser.error("Episode counts must be positive")
    # Training/evaluation seeds never overlap reference/calibration worlds.
    if args.command == "train" and (args.reference_episodes > 10_000 or not 30_000 <= args.seed < 40_000):
        parser.error("Use <=10000 reference episodes and training seed in [30000,40000)")
    if args.command == "evaluate" and args.seed < 40_000:
        parser.error("Use evaluation seeds >=40000, separate from fitting and training")
    torch.set_num_threads(1)
    config, agents = load_agents(args.market)
    coalitions, solo = load_auxiliary(args.market)
    if coalitions is None:
        parser.error("Market checkpoint needs a trained coalition policy")
    if args.command == "train":
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("Use a new output directory to preserve previous checkpoints")
        args.out.mkdir(parents=True, exist_ok=True)
        train(args, config, agents, coalitions, solo)
    else:
        adapter = ArbiterAdapter.load(args.checkpoint, config, device=args.device)
        if adapter.policy.updates < 1:
            parser.error("Evaluate a trained checkpoint, not initial weights")
        metrics = evaluate(config, agents, coalitions, args.episodes, args.seed, arbiter=adapter)
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "evaluation.json", metrics)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
