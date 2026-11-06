"""Train and evaluate our agent without modifying the environment's runner."""

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from askarov_agent import AskarovAgent
from social_market_env.baselines.agents import BestResponseAgent, CartelAgent
from social_market_env.config import (
    ComponentSpec, ExperimentConfig, config_from_dict, config_to_dict, load_config,
)
from social_market_env.environment import SocialMarketEnvironment
from social_market_env.runner.experiment import build_environment, episode_seed
from social_market_env.runner.loader import instantiate

SCENARIOS = ('competition', 'cooperation', 'mixed', 'sanctions')
TOPOLOGIES = ('complete', 'star', 'path', 'inner_coalitions', 'random')
VARIANTS = ('best_response', 'cartel', 'hybrid', 'q_learning')


def scenario_config(base: ExperimentConfig, *, scenario: str, agent_id: int) -> ExperimentConfig:
    """Vary existing opponent/coalition/arbiter implementations, never their code.

    'mixed' preserves the user's opponents, coalition policy and arbiter. The
    other scenarios are controlled baselines, independent of a custom arbiter.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {scenario}')
    config = config_from_dict(config_to_dict(base))
    config.strict = True
    if scenario != 'mixed':
        config.agents.overrides.clear()
        agent_type = 'BestResponseAgent' if scenario == 'competition' else 'CartelAgent'
        config.agents.default = ComponentSpec(f'social_market_env.baselines.agents:{agent_type}')
        coalition_type = 'NoCoalitions' if scenario == 'competition' else 'ConnectedComponentsCoalitions'
        config.coalitions.policy = ComponentSpec(f'social_market_env.baselines.coalitions:{coalition_type}')
        config.arbiter.policy = ComponentSpec('social_market_env.baselines.arbiters:AlwaysObserve')
        if scenario == 'sanctions':
            config.arbiter.policy = ComponentSpec(
                'social_market_env.baselines.arbiters:ThresholdArbiter',
                {'jump_threshold': 0.0, 'fine': config.arbiter.max_fine},
            )
    config.agents.overrides[agent_id] = ComponentSpec('askarov_agent:AskarovAgent')
    return config


def play_episode(
    env: SocialMarketEnvironment, *, agent_id: int, seed: int, training: bool, gamma: float,
) -> dict[str, Any]:
    """Measure the focal seller and welfare using the public step records."""
    env.reset(seed=seed, training=training)
    rewards: list[float] = []
    welfare: list[float] = []
    prices: list[float] = []
    fines = fees = 0.0
    joined = 0
    while not env.done:
        record = env.step().record
        rewards.append(record['agent_rewards'].get(agent_id, 0.0))
        welfare.append(record['social_welfare'])
        prices.append(record['average_price'])
        fines += record['fines'].get(agent_id, 0.0)
        fees += record['link_fees'].get(agent_id, 0.0)
        joined += sum(agent_id in plan['compliers'] for plan in record['plans'])
    account = env.state.accounts[agent_id]
    return {
        'steps': len(rewards), 'net_reward': sum(rewards),
        'discounted_reward': sum(gamma**step * value for step, value in enumerate(rewards)),
        'mean_social_welfare': mean(welfare), 'mean_market_price': mean(prices),
        'fines': fines, 'link_fees': fees, 'coalition_steps': joined,
        'final_balance': account.balance, 'survived': account.active,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def train(args: argparse.Namespace) -> int:
    """Round-robin across opponents, network topologies and training seeds."""
    base = load_config(args.config)
    if args.horizon is not None:
        base.horizon = args.horizon
    if not 0 <= args.agent_id < base.agents.count or not base.seeds:
        raise ValueError('Require a valid agent ID and nonempty training seeds')
    spec = base.agents.overrides.get(args.agent_id, base.agents.default)
    learner = instantiate(spec=spec, defaults={'market': asdict(base.market)})
    if not isinstance(learner, AskarovAgent):
        raise ValueError('The selected agent must be an AskarovAgent in the YAML')
    combinations = list(product(args.scenarios, args.topologies, args.network_modes, base.seeds))
    environments: dict[tuple, SocialMarketEnvironment] = {}
    rows = []
    for episode in range(args.episodes):
        combination = combinations[episode % len(combinations)]
        scenario, topology, mode, seed = combination
        if combination not in environments:
            config = scenario_config(base, scenario=scenario, agent_id=args.agent_id)
            config.network.topology = topology
            config.network.mode = mode
            environments[combination] = build_environment(config=config, seed=seed)
        env = environments[combination]
        env.agents[args.agent_id] = learner
        reset_seed = episode_seed(seed=seed, episode=episode)
        metrics = play_episode(env, agent_id=args.agent_id, seed=reset_seed, training=True, gamma=learner.gamma)
        rows.append({
            'episode': episode, 'scenario': scenario, 'topology': topology,
            'network_mode': mode, 'seed': seed, 'episode_seed': reset_seed, **metrics,
        })
        learner.save(args.output)
        print(
            f'Train {episode + 1}/{args.episodes}: {scenario}/{topology}/{mode} seed={seed} '
            f'reward={metrics["net_reward"]:.2f} updates={learner.updates}', flush=True,
        )
    manifest = {
        'config': config_to_dict(base), 'agent_id': args.agent_id,
        'training_seeds': base.seeds, 'episodes': rows,
        'q_states': len(learner.q_table), 'updates': learner.updates,
        'scenarios': args.scenarios, 'topologies': args.topologies, 'network_modes': args.network_modes,
    }
    _write_json(args.output.with_suffix('.training.json'), manifest)
    evaluation = config_to_dict(base)
    evaluation.update(mode='eval', strict=True, seeds=[100, 101, 102], episodes_per_seed=1)
    payload = json.loads(args.output.read_text(encoding='utf-8'))
    evaluation['agents']['overrides'][args.agent_id] = {
        'impl': 'askarov_agent:AskarovAgent',
        'kwargs': {**payload['settings'], 'checkpoint_path': str(args.output.resolve())},
    }
    eval_path = args.output.with_suffix('.eval.yaml')
    eval_path.write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding='utf-8')
    print(f'Checkpoint: {args.output}\nEvaluation config: {eval_path}', flush=True)
    return 0


def benchmark(args: argparse.Namespace) -> int:
    """Paired evaluation: same focal slot, opponents, costs and reset seeds."""
    base = load_config(args.config)
    if args.horizon is not None:
        base.horizon = args.horizon
    if not 0 <= args.agent_id < base.agents.count:
        raise ValueError('Invalid agent ID')
    trained = AskarovAgent.from_checkpoint(args.checkpoint, market=asdict(base.market))
    payload = json.loads(args.checkpoint.read_text(encoding='utf-8'))
    variants = {
        'best_response': BestResponseAgent(market=asdict(base.market)),
        'cartel': CartelAgent(market=asdict(base.market)),
        'hybrid': AskarovAgent(market=asdict(base.market), **payload['settings']),
        'q_learning': trained,
    }
    frozen = json.dumps(trained.q_table, sort_keys=True), trained.updates
    combinations = list(product(args.scenarios, args.topologies, args.network_modes, args.seeds))
    rows = []
    for index, (scenario, topology, mode, seed) in enumerate(combinations):
        config = scenario_config(base, scenario=scenario, agent_id=args.agent_id)
        config.network.topology = topology
        config.network.mode = mode
        env = build_environment(config=config, seed=seed)
        reset_seed = episode_seed(seed=seed, episode=0)
        for variant, agent in variants.items():
            env.agents[args.agent_id] = agent
            metrics = play_episode(env, agent_id=args.agent_id, seed=reset_seed, training=False, gamma=trained.gamma)
            rows.append({
                'variant': variant, 'scenario': scenario, 'topology': topology,
                'network_mode': mode, 'seed': seed, 'episode_seed': reset_seed,
                'cost': env.costs[args.agent_id], **metrics,
            })
        print(f'Eval {index + 1}/{len(combinations)}: {scenario}/{topology}/{mode} seed={seed}', flush=True)
    if frozen != (json.dumps(trained.q_table, sort_keys=True), trained.updates):
        raise AssertionError('Evaluation changed the trained policy')
    aggregate = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row['variant'] == variant]
        aggregate[variant] = {
            'episodes': len(selected),
            **{key: mean(row[key] for row in selected) for key in (
                'net_reward', 'discounted_reward', 'mean_social_welfare', 'mean_market_price',
                'fines', 'link_fees', 'coalition_steps', 'survived',
            )},
            'net_reward_std': stdev(row['net_reward'] for row in selected) if len(selected) > 1 else 0.0,
        }
    training_path = args.checkpoint.with_suffix('.training.json')
    training_seeds: list[int] | None = None
    if training_path.exists():
        training_seeds = json.loads(training_path.read_text(encoding='utf-8'))['training_seeds']
    report = {
        'checkpoint': str(args.checkpoint.resolve()), 'config': config_to_dict(base),
        'agent_id': args.agent_id, 'evaluation_seeds': args.seeds,
        'training_seeds': training_seeds,
        'seed_overlap': sorted(set(args.seeds) & set(training_seeds)) if training_seeds is not None else None,
        'scenarios': args.scenarios, 'topologies': args.topologies, 'network_modes': args.network_modes,
        'aggregate': aggregate, 'episodes': rows,
    }
    _write_json(args.output, report)
    lines = [
        '# Askarov agent comparison', '',
        f'Focal seller: {args.agent_id}. Evaluation seeds: {args.seeds}. Training seeds: {training_seeds}.',
        'All variants use identical initial costs and reset seeds in each scenario. Evaluation is frozen.', '',
        '| Policy | Episodes | Mean net reward | Std across episodes | Discounted reward | Mean welfare | Survival |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for variant, stats in aggregate.items():
        lines.append(
            f'| {variant} | {stats["episodes"]} | {stats["net_reward"]:.2f} | '
            f'{stats["net_reward_std"]:.2f} | {stats["discounted_reward"]:.2f} | '
            f'{stats["mean_social_welfare"]:.2f} | {stats["survived"]:.1%} |'
        )
    lines.extend(['', 'Dispersion mixes different scenarios; it is not a confidence interval.',
                  'These results concern the individual seller, not completion of all team lab experiments.', ''])
    args.output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'Detailed report: {args.output}', flush=True)
    return 0


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'benchmark'):
        sub = commands.add_parser(command)
        sub.add_argument('--config', type=Path, default=Path('experiments/askarov_agent.yaml'))
        sub.add_argument('--agent-id', type=int, default=0)
        sub.add_argument('--horizon', type=positive_int)
        sub.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS[:3]))
        sub.add_argument('--topologies', nargs='+', choices=TOPOLOGIES, default=list(TOPOLOGIES[:4]))
        sub.add_argument('--network-modes', nargs='+', choices=('static', 'evolving'),
                         default=['evolving'] if command == 'train' else ['static', 'evolving'])
        if command == 'train':
            sub.add_argument('--episodes', type=positive_int, default=60)
            sub.add_argument('--output', type=Path, default=Path('runs/askarov_agent/policy.json'))
            sub.set_defaults(handler=train)
        else:
            sub.add_argument('--checkpoint', type=Path, default=Path('runs/askarov_agent/policy.json'))
            sub.add_argument('--seeds', nargs='+', type=int, default=[100, 101, 102])
            sub.add_argument('--output', type=Path, default=Path('runs/askarov_agent/comparison.json'))
            sub.set_defaults(handler=benchmark)
    args = parser.parse_args(argv)
    if args.command == 'benchmark' and args.output.resolve() == args.checkpoint.resolve():
        parser.error('--output must differ from --checkpoint')
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
