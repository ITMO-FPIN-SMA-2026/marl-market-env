"""Train one seller across changing markets and compare frozen policies."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from statistics import mean

import yaml

from tverdokhleb_agent import TverdokhlebAgent
from social_market_env.baselines.agents import BestResponseAgent
from social_market_env.config import ComponentSpec, ExperimentConfig, config_from_dict, config_to_dict, load_config
from social_market_env.runner.experiment import build_environment, episode_seed

IMPL = 'tverdokhleb_agent:TverdokhlebAgent'
SCENARIOS = ('competition', 'cooperation', 'sanctions', 'mixed')
TOPOLOGIES = ('complete', 'star', 'path', 'inner_coalitions')
MODES = ('static', 'evolving')
CONDITIONS = tuple(product(SCENARIOS, TOPOLOGIES, MODES))
BASELINES = ('untrained', 'best_response')


def scenario_config(base: ExperimentConfig, condition: tuple[str, str, str], agent_id: int) -> ExperimentConfig:
    """Build an independent config using only the environment's existing policies."""
    if not 0 <= agent_id < base.agents.count:
        raise ValueError('agent_id is outside the market')
    scenario, topology, mode = condition
    if condition not in CONDITIONS and condition != ('base', base.network.topology, base.network.mode):
        raise ValueError(f'Unknown condition: {condition}')
    config = config_from_dict(config_to_dict(base))
    config.strict = True
    config.network.topology, config.network.mode = topology, mode
    if scenario not in ('mixed', 'base'):
        config.agents.overrides.clear()
        peer = 'BestResponseAgent' if scenario == 'competition' else 'CartelAgent'
        config.agents.default = ComponentSpec(f'social_market_env.baselines.agents:{peer}')
        coalition = 'NoCoalitions' if scenario == 'competition' else 'ConnectedComponentsCoalitions'
        config.coalitions.policy = ComponentSpec(f'social_market_env.baselines.coalitions:{coalition}')
        config.arbiter.policy = ComponentSpec('social_market_env.baselines.arbiters:AlwaysObserve')
        if scenario == 'sanctions':
            config.arbiter.policy = ComponentSpec('social_market_env.baselines.arbiters:ThresholdArbiter',
                                                {'jump_threshold': 1.0, 'fine': config.arbiter.max_fine})
    config.agents.overrides[agent_id] = ComponentSpec(IMPL)
    return config


def play(env, *, agent_id: int, seed: int, training: bool, gamma: float) -> dict:
    """Count the focal seller's transitions, rather than the market horizon."""
    env.reset(seed=episode_seed(seed=seed, episode=0), training=training)
    total = discounted = fines = welfare = 0.0
    steps = transitions = joined = 0
    while not env.done:
        record = env.step().record
        if record['events']:
            raise RuntimeError(f"Component errors: {record['events']}")
        reward = record['agent_rewards'].get(agent_id, 0.0)
        total += reward
        discounted += gamma ** steps * reward
        fines += record['fines'].get(agent_id, 0.0)
        welfare += record['social_welfare']
        transitions += int(agent_id in record['agent_rewards'])
        joined += sum(agent_id in plan['compliers'] for plan in record['plans'])
        steps += 1
    agent = env.agents[agent_id]
    return {'return': total, 'discounted_return': discounted,
            'survived': env.state.accounts[agent_id].active, 'transitions': transitions,
            'fines': fines, 'coalition_steps': joined, 'mean_welfare': welfare / steps,
            'known_decisions': getattr(agent, 'known_state_decisions', 0),
            'decisions': getattr(agent, 'decisions', 0)}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def aggregate(rows: list[dict]) -> dict:
    decisions = sum(row['decisions'] for row in rows)
    return {'episodes': len(rows),
            **{key: mean(row[key] for row in rows) for key in (
                'return', 'discounted_return', 'survived', 'transitions', 'fines', 'coalition_steps', 'mean_welfare')},
            'known_fraction': sum(row['known_decisions'] for row in rows) / decisions if decisions else None}


def comparison_summary(rows: list[dict]) -> tuple[dict, dict]:
    policies = list(dict.fromkeys(row['policy'] for row in rows))
    summary = {name: aggregate([row for row in rows if row['policy'] == name]) for name in policies}
    by_scenario = {
        scenario: {name: aggregate([row for row in rows
                                   if row['policy'] == name and row['condition'][0] == scenario])
                   for name in policies}
        for scenario in dict.fromkeys(row['condition'][0] for row in rows)
    }
    return summary, by_scenario


def comparison_tables(summary: dict, by_scenario: dict) -> list[str]:
    lines = ['| Policy | Mean return | Survival | Known states |', '|---|---:|---:|---:|']
    for name, values in summary.items():
        known = f"{values['known_fraction']:.1%}" if values['known_fraction'] is not None else '—'
        lines.append(f"| {name} | {values['return']:.2f} | {values['survived']:.1%} | {known} |")
    lines.extend(['', '| Scenario | ' + ' | '.join(summary) + ' |', '|---|' + '---:|' * len(summary)])
    for scenario, policies in by_scenario.items():
        values = ' | '.join(f"{policies[name]['return']:.2f}" for name in summary)
        lines.append(f'| {scenario} | {values} |')
    return lines


def train_model(base: ExperimentConfig, *, state_mode: str, seed: int, episodes: int,
                min_updates: int, agent_id: int, output: Path) -> tuple[TverdokhlebAgent, dict]:
    """Keep one learner while each episode draws new costs and a new condition."""
    if episodes < 1 or min_updates < 0 or min_updates > episodes * base.horizon or seed < 0:
        raise ValueError('Require positive episodes, nonnegative seed and a reachable update budget')
    if not 0 <= agent_id < base.agents.count:
        raise ValueError('agent_id is outside the market')
    spec = base.agents.overrides.get(agent_id, base.agents.default)
    if spec.impl not in (IMPL, 'tverdokhleb_agent.agent:TverdokhlebAgent'):
        raise ValueError('The selected seller must be TverdokhlebAgent in the YAML')
    if set(spec.kwargs) - {'config'}:
        raise ValueError('Training starts fresh; use only kwargs.config in the base YAML')
    settings = {**spec.kwargs.get('config', {}), 'state_mode': state_mode}
    learner = TverdokhlebAgent(config=settings, market=asdict(base.market))
    roots = list(range(seed, seed + episodes))
    learner.metadata.update(training_root_seeds=[], training_protocol='rotating-markets-v1')
    rows = []
    for episode, root_seed in enumerate(roots):
        condition = CONDITIONS[episode % len(CONDITIONS)]
        config = scenario_config(base, condition, agent_id)
        env = build_environment(config=config, seed=root_seed)
        env.agents[agent_id] = learner
        before = learner.updates
        metrics = play(env, agent_id=agent_id, seed=root_seed, training=True, gamma=learner.config.gamma)
        if learner.updates - before != metrics['transitions']:
            raise AssertionError('Lost learning transitions')
        learner.metadata['training_root_seeds'].append(root_seed)
        rows.append({'seed': root_seed, 'condition': list(condition), 'cost': env.costs[agent_id], **metrics})
        if (episode + 1) % 50 == 0 or episode + 1 == episodes:
            learner.save(output)
            print(f'{state_mode}: {episode + 1}/{episodes} episodes, {learner.updates} updates, '
                  f'epsilon={learner.epsilon:.4f}', flush=True)
    report = {'config': config_to_dict(base), 'state_mode': state_mode, 'agent_id': agent_id,
              'training_root_seeds': roots, 'updates': learner.updates, 'states': len(learner.q),
              'final_epsilon': learner.epsilon, 'min_updates': min_updates,
              'budget_met': learner.updates >= min_updates, 'aggregate': aggregate(rows), 'episodes': rows}
    write_json(output.with_suffix('.training.json'), report)
    if not report['budget_met']:
        raise ValueError(f'Only {learner.updates} updates; required {min_updates}. See the training report.')
    return learner, report


def evaluation_config(base: ExperimentConfig, *, checkpoint: Path, seeds: list[int], agent_id: int) -> ExperimentConfig:
    """Emit a config directly usable by the native runner, with a portable relative path."""
    agent = TverdokhlebAgent.load(checkpoint)
    validate_eval_seeds([agent], seeds)
    config = config_from_dict(config_to_dict(base))
    config.mode, config.strict, config.seeds, config.episodes_per_seed = 'eval', True, seeds, 1
    config.agents.overrides[agent_id] = ComponentSpec(IMPL, {'model_path': str(checkpoint)})
    return config


def validate_eval_seeds(agents: list[TverdokhlebAgent], seeds: list[int]) -> None:
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Evaluation seeds must be distinct nonnegative integers')
    training = {seed for agent in agents for seed in agent.metadata.get('training_root_seeds', [])}
    if overlap := training & set(seeds):
        raise ValueError(f'Training/evaluation seed overlap: {sorted(overlap)}')


def evaluate(base: ExperimentConfig, *, models: dict[str, TverdokhlebAgent], seeds: list[int],
             agent_id: int, conditions=None) -> list[dict]:
    validate_eval_seeds(list(models.values()), seeds)
    if not models or set(models) & set(BASELINES):
        raise ValueError('Supply models with names distinct from the comparison baselines')
    reference = next(iter(models.values()))
    for agent in models.values():
        if agent.market != asdict(base.market):
            raise ValueError('Checkpoint market differs from the evaluation market')
    frozen = {name: copy.deepcopy((agent.q, agent.updates, agent.metadata)) for name, agent in models.items()}
    rows = []
    for condition, seed in product(CONDITIONS if conditions is None else conditions, seeds):
        config = scenario_config(base, condition, agent_id)
        for name in (*models, *BASELINES):
            # New instances for the other roles; identical initial costs and reset seeds.
            env = build_environment(config=config, seed=seed)
            if name in models:
                agent = models[name]
            elif name == 'untrained':
                agent = TverdokhlebAgent(config=reference.config, market=reference.market)
            else:
                agent = BestResponseAgent(market=asdict(base.market))
            env.agents[agent_id] = agent
            metrics = play(env, agent_id=agent_id, seed=seed, training=False, gamma=reference.config.gamma)
            rows.append({'policy': name, 'seed': seed, 'condition': list(condition),
                         'cost': env.costs[agent_id], **metrics})
    for name, agent in models.items():
        if (agent.q, agent.updates, agent.metadata) != frozen[name]:
            raise AssertionError('Evaluation changed a trained model')
    return rows


def evaluate_checkpoint(args, base):
    """Compare a saved model with baselines, preserving the input YAML by default."""
    outputs = {args.output.resolve(), args.output.with_suffix('.md').resolve()}
    if args.output.suffix != '.json' or outputs & {args.checkpoint.resolve(), args.config.resolve()}:
        raise ValueError('Evaluation output must be a separate .json file, distinct from checkpoint and config')
    agent = TverdokhlebAgent.load(args.checkpoint)
    conditions = [('base', base.network.topology, base.network.mode)] if args.suite == 'base' else None
    rows = evaluate(base, models={'trained': agent}, seeds=args.eval_seeds,
                    agent_id=args.agent_id, conditions=conditions)
    summary, by_scenario = comparison_summary(rows)
    write_json(args.output, {'config': config_to_dict(base), 'checkpoint': str(args.checkpoint),
                            'agent_id': args.agent_id, 'eval_seeds': args.eval_seeds, 'suite': args.suite,
                            'aggregate': summary, 'by_scenario': by_scenario, 'episodes': rows})
    lines = ['# TverdokhlebAgent: evaluation', '',
             f'Checkpoint: {args.checkpoint}. Suite: {args.suite}. Evaluation seeds: {args.eval_seeds}.',
             'Paired costs and reset seeds; the trained model is frozen.', '',
             *comparison_tables(summary, by_scenario), '']
    args.output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


def train(args, base):
    _, report = train_model(base, state_mode=args.state_mode, seed=args.seed, episodes=args.episodes,
                            min_updates=args.min_updates, agent_id=args.agent_id, output=args.output)
    config = evaluation_config(base, checkpoint=args.output, seeds=args.eval_seeds, agent_id=args.agent_id)
    eval_path = args.output.with_suffix('.eval.yaml')
    eval_path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding='utf-8')
    print(f"Training survival: {report['aggregate']['survived']:.1%}. Native evaluation: {eval_path}", flush=True)


def benchmark(args, base):
    if len(set(args.train_seeds)) != len(args.train_seeds) or any(seed < 0 for seed in args.train_seeds):
        raise ValueError('Training repeats need distinct nonnegative seed starts')
    blocks = [set(range(seed, seed + args.episodes)) for seed in args.train_seeds]
    if any(left & right for i, left in enumerate(blocks) for right in blocks[i + 1:]):
        raise ValueError('Independent training seed blocks must not overlap')
    if set(args.eval_seeds) & set().union(*blocks):
        raise ValueError('Training/evaluation seed overlap')
    if len(set(args.variants)) != len(args.variants):
        raise ValueError('Duplicate state variants')
    records, training = [], []
    for repeat, seed in enumerate(args.train_seeds):
        models = {}
        for variant in args.variants:
            path = args.output / f'repeat_{repeat}' / f'{variant}.json'
            _, report = train_model(base, state_mode=variant, seed=seed, episodes=args.episodes,
                                    min_updates=args.min_updates, agent_id=args.agent_id, output=path)
            training.append({'repeat': repeat, **report})
            models[variant] = TverdokhlebAgent.load(path)
        evaluated = evaluate(base, models=models, seeds=args.eval_seeds, agent_id=args.agent_id)
        records.extend({'repeat': repeat, **row} for row in evaluated)
        print(f'Completed evaluation repeat {repeat + 1}/{len(args.train_seeds)}', flush=True)
    summary, by_scenario = comparison_summary(records)
    paired = []
    if {'context', 'history'} <= set(args.variants):
        for repeat in range(len(args.train_seeds)):
            by_policy = {name: {(tuple(row['condition']), row['seed']): row['return'] for row in records
                                if row['repeat'] == repeat and row['policy'] == name}
                         for name in ('context', 'history')}
            paired.append(mean(by_policy['history'][key] - value for key, value in by_policy['context'].items()))
    report = {'config': config_to_dict(base), 'train_seeds': args.train_seeds, 'eval_seeds': args.eval_seeds,
              'variants': args.variants, 'aggregate': summary, 'by_scenario': by_scenario,
              'training': training, 'episodes': records,
              'history_minus_context_by_training_repeat': paired,
              'note': 'Evaluation seeds/conditions are paired; repeats share evaluation cases. No independence or significance claim across all episodes.'}
    write_json(args.output / 'comparison.json', report)
    lines = ['# TverdokhlebAgent: comparison', '',
             f'Training seed starts: {args.train_seeds}; evaluation seeds: {args.eval_seeds}.',
             'One learner per variant/repeat; new costs each training episode. Evaluation is frozen.', '',
             *comparison_tables(summary, by_scenario)]
    lines.extend(['', f'Paired history − context mean returns by training repeat: {paired}.',
                  'Known states means a Q row exists, not that every action is well estimated.',
                  'These are individual-agent experiments, not the team mutual-adaptation cycles.', ''])
    (args.output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'benchmark', 'evaluate'):
        sub = commands.add_parser(name)
        sub.add_argument('--config', type=Path, default=Path('experiments/tverdokhleb_agent.yaml'))
        sub.add_argument('--agent-id', type=int, default=0)
        sub.add_argument('--horizon', type=positive)
        sub.add_argument('--eval-seeds', nargs='+', type=int, default=[100000, 100001, 100002])
        if name != 'evaluate':
            sub.add_argument('--episodes', type=positive, default=300)
            sub.add_argument('--min-updates', type=positive, default=10000)
        if name == 'train':
            sub.add_argument('--seed', type=int, default=42)
            sub.add_argument('--state-mode', choices=('basic', 'context', 'history'), default='history')
            sub.add_argument('--output', type=Path, default=Path('runs/tverdokhleb_agent/policy.json'))
        elif name == 'benchmark':
            sub.add_argument('--train-seeds', nargs='+', type=int, default=[42, 1042, 2042])
            sub.add_argument('--variants', nargs='+', choices=('basic', 'context', 'history'), default=['context', 'history'])
            sub.add_argument('--output', type=Path, default=Path('runs/tverdokhleb_agent/benchmark'))
        else:
            sub.add_argument('--checkpoint', type=Path, default=Path('runs/tverdokhleb_agent/policy.json'))
            sub.add_argument('--suite', choices=('base', 'matrix'), default='base',
                             help='base preserves the YAML; matrix evaluates all 32 controlled conditions')
            sub.add_argument('--output', type=Path, default=Path('runs/tverdokhleb_agent/evaluation.json'))
    args = parser.parse_args(argv)
    base = load_config(args.config)
    if args.horizon is not None:
        base.horizon = args.horizon
    validate_eval_seeds([], args.eval_seeds)
    # Fail before training or writing a checkpoint if requested evaluation overlaps it.
    if args.command == 'train' and set(args.eval_seeds) & set(range(args.seed, args.seed + args.episodes)):
        raise ValueError('Training/evaluation seed overlap')
    {'train': train, 'benchmark': benchmark, 'evaluate': evaluate_checkpoint}[args.command](args, base)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
