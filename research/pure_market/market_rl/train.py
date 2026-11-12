"""Reproducible training/evaluation and the optional external arbiter callback."""

import argparse
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .agents import encode_state, load_agents, load_auxiliary, make_agents, save_agents
from .coalition_policy import CoalitionPolicy
from .environment import ArbiterAction, MarketConfig, MarketEnv, PublicObservation


def run_episode(env, agents, seed=0, training=True, epsilon_scale=1.0,
                arbiter: Callable[[PublicObservation], ArbiterAction] | None = None,
                coalition_policy=None, solo=False, arbiter_training=False, arbiter_scoring=None, evaluator=None):
    """arbiter(public_observation) is the integration point, with no policy supplied.

    Stateful arbiters are reset and receive scalar rewards through their
    transition hook. Private evaluation info never enters either callback.
    """
    observations = env.reset(seed)
    if coalition_policy is not None:
        coalition_policy.reset_episode()
    if not solo and coalition_policy is None:
        raise ValueError("The market run needs a learned coalition pricing policy")
    feedback = arbiter is not None and hasattr(arbiter, "observe_transition")
    if arbiter_training and not feedback:
        raise ValueError("Arbiter training requires a transition hook")
    if arbiter is not None and hasattr(arbiter, "reset"):
        arbiter.reset(seed=seed, training=arbiter_training)
    from .arbiter_scoring import ArbiterScoring, score_arbiter

    scoring = arbiter_scoring or getattr(arbiter, "scoring", ArbiterScoring(target_welfare=max(1e-9, env.config.welfare_bound)))
    pending_transition = None
    arbiter_totals = np.zeros(5)

    def action_mask(i):
        mask = env.action_mask(i)
        return mask[::5] if solo else mask

    states = [encode_state(o, env.config.prices) for o in observations]
    epsilons = [max(0.03, a.config.epsilon * epsilon_scale) if training else 0.0 for a in agents]
    actions = [a.act(s, action_mask(i), epsilons[i]) for i, (a, s) in enumerate(zip(agents, states))]
    totals = np.zeros(4)
    records = []
    while True:
        market_actions = [a * 5 for a in actions] if solo else actions
        joint_prices = coalition_policy.plan(env, market_actions, training, max(epsilons)) if coalition_policy else None
        public = env.market_step(market_actions, joint_prices)
        if feedback and pending_transition is not None:
            previous, applied, reward = pending_transition
            arbiter.observe_transition(observation=previous, action=applied, reward=reward,
                                       next_observation=public, done=False)
        intervention = arbiter(public) if arbiter is not None else None
        next_observations, rewards, done, info = env.settle(intervention)
        arbiter_reward, verdict = score_arbiter(public, intervention or ArbiterAction(), info, scoring)
        if evaluator is not None:
            evaluator(public, intervention or ArbiterAction(), info, verdict)
        arbiter_totals += [arbiter_reward, verdict["sanction"], verdict["false_positive"],
                           verdict["false_negative"], verdict["harmful_coalitions"]]
        if feedback:
            if done:
                arbiter.observe_transition(observation=public, action=intervention, reward=arbiter_reward,
                                           next_observation=None, done=True)
            else:
                pending_transition = (public, intervention, arbiter_reward)
        next_states = [encode_state(o, env.config.prices) for o in next_observations]
        next_masks = [action_mask(i) for i in range(4)]
        next_actions = [a.act(s, next_masks[i], epsilons[i]) for i, (a, s) in enumerate(zip(agents, next_states))] if not done else [None] * 4
        if training:
            for i, agent in enumerate(agents):
                if observations[i].active:
                    agent.learn(states[i], actions[i], rewards[i] / 100.0, next_states[i], next_masks[i],
                                done or not next_observations[i].active, next_actions[i], epsilons[i])
            if coalition_policy is not None:
                coalition_policy.observe(rewards, done)
        totals += rewards
        records.append((public.mean_price, public.social_welfare, info["consumer_surplus"],
                        sum(len(c) for c in info["coalitions"] if len(c) > 1) / 4,
                        info["graph"].sum() / 12, info["net_welfare"], info["fines"].sum()))
        if done:
            break
        observations, states, actions = next_observations, next_states, next_actions
    mean = np.mean(records, axis=0)
    return {
        "steps": env.t, **{f"reward_{i}": float(totals[i]) for i in range(4)},
        "mean_price": float(mean[0]), "social_welfare": float(mean[1]),
        "consumer_surplus": float(mean[2]), "coalition_fraction": float(mean[3]),
        "network_density": float(mean[4]), "net_welfare": float(mean[5]),
        "mean_fines": float(mean[6]), "survivors": int(env.active.sum()),
        **dict(zip(("arbiter_reward", "sanctions", "false_positives", "false_negatives", "harmful_coalitions"),
                   map(float, arbiter_totals))),
    }


def evaluate(config, agents, coalition_policy, episodes=20, seed=100_000, arbiter=None, arbiter_scoring=None):
    # Evaluation cannot change training weights, counters or RNG state.
    evaluated = copy.deepcopy(agents)
    policy = copy.deepcopy(coalition_policy)
    evaluated_arbiter = copy.deepcopy(arbiter)
    policy.rng = np.random.default_rng(seed + 2_000_000)
    for i, agent in enumerate(evaluated):
        agent.rng = np.random.default_rng(seed + 1_000_000 + i)
    results = [run_episode(MarketEnv(config), evaluated, seed + i, training=False, arbiter=evaluated_arbiter,
                           coalition_policy=policy, arbiter_scoring=arbiter_scoring)
               for i in range(episodes)]
    return {key: float(np.mean([row[key] for row in results])) for key in results[0]}


def main():
    parser = argparse.ArgumentParser(description="Four RL market agents with an optional learned public arbiter")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology", choices=["complete", "star", "path", "inner", "random"], default=None)
    parser.add_argument("--allocation", choices=["equal", "shapley", "myerson", "owen"], default=None)
    parser.add_argument("--fixed-network", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--load", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--arbiter", type=Path, help="PPO/GRU .pt checkpoint, or experimental MLP .json")
    parser.add_argument("--arbiter-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--demand-model", choices=["assignment", "logit"])
    parser.add_argument("--logit-value", type=float)
    parser.add_argument("--logit-temperature", type=float)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--harm-baseline", choices=["auto", "proposals", "nash"])
    args = parser.parse_args()
    if args.episodes < 1 or args.checkpoint_every < 1 or (args.steps is not None and args.steps < 1):
        parser.error("episodes and steps must be positive")
    if args.evaluate and not args.load:
        parser.error("--evaluate requires --load")
    config, agents = load_agents(args.load) if args.load else (MarketConfig(), None)
    policy, solo_agents = load_auxiliary(args.load) if args.load else (None, None)
    changes = {k: v for k, v in {"horizon": args.steps, "topology": args.topology,
                                "allocation": args.allocation, "demand_model": args.demand_model,
                                "logit_value": args.logit_value, "logit_temperature": args.logit_temperature,
                                "harm_baseline": args.harm_baseline}.items() if v is not None}
    if args.load and any(k in changes and changes[k] != getattr(config, k)
                         for k in ("demand_model", "logit_value", "logit_temperature")):
        parser.error("Start fresh policies when changing the demand model or its parameters")
    if args.fixed_network:
        changes["dynamic_network"] = False
    config = replace(config, **changes)
    env = MarketEnv(config)
    agents = agents if agents is not None else make_agents(env.n_actions, args.seed)
    policy = policy if policy is not None else CoalitionPolicy(len(config.prices), args.seed + 200)
    solo_agents = solo_agents if solo_agents is not None else make_agents(len(config.prices), args.seed + 300)
    arbiter = None
    if args.arbiter:
        if args.arbiter.suffix == ".json":
            from .sanction import LearnedArbiter

            arbiter = LearnedArbiter.load(args.arbiter, config)
        elif args.arbiter.suffix == ".pt":
            from .arbiter_v2 import ArbiterAdapter

            arbiter = ArbiterAdapter.load(args.arbiter, config, device=args.arbiter_device)
        else:
            raise ValueError("Use a PPO/GRU .pt checkpoint or an experimental MLP .json checkpoint")
    if args.evaluate:
        print(json.dumps(evaluate(config, agents, policy, args.episodes, args.seed, arbiter=arbiter), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    from .learn_arbiter import record_weights

    weights = args.output / "weights" / "market_training"
    weights.mkdir(parents=True, exist_ok=False)
    initial = weights / "episode_00000.json"
    save_agents(initial, agents, config, policy, solo_agents)
    record_weights(weights, (initial,), stage="market_training", episode=0, seed=args.seed)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for episode in range(args.episodes):
            # A learned competitive reference, rather than a fixed-price rule.
            run_episode(MarketEnv(config), solo_agents, args.seed + episode, solo=True,
                        epsilon_scale=0.995**episode, arbiter=arbiter)
            metrics = run_episode(env, agents, args.seed + episode, epsilon_scale=0.995**episode,
                                  coalition_policy=policy, arbiter=arbiter)
            row = {"episode": episode + 1, **metrics}
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if (episode + 1) % args.checkpoint_every == 0 or episode + 1 == args.episodes:
                checkpoint = weights / f"episode_{episode+1:05d}.json"
                save_agents(checkpoint, agents, config, policy, solo_agents)
                record_weights(weights, (checkpoint,), stage="market_training", episode=episode+1,
                               seed=args.seed, metrics=metrics)
            if episode == 0 or (episode + 1) % 50 == 0:
                print(f"episode={episode + 1} price={metrics['mean_price']:.2f} "
                      f"SW={metrics['social_welfare']:.2f} coalitions={metrics['coalition_fraction']:.2f}", flush=True)
    save_agents(args.output / "agents.json", agents, config, policy, solo_agents)
    result = evaluate(config, agents, policy, arbiter=arbiter)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    env.reset(100_000)
    values = policy.values(env, solo_agents)
    (args.output / "coalition_values.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
