"""Train real Q-tables, compare paired policies, and emit native-runner YAML.

Only our files and the dedicated experiments/zhmurenko_agent.yaml are source
outputs. Every training/evaluation record is written outside the repository.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

import yaml

from social_market_env.config import ComponentSpec, config_from_dict, config_to_dict, load_config
from social_market_env.runner.experiment import build_environment, episode_info, episode_seed
from social_market_env.runner.loader import instantiate

from .agent import ACTIONS, JOIN, PACKAGE, ZhmurenkoVladimirAgent, output_path
from .references import activate, prepare

IMPL = 'zhmurenko_agent:ZhmurenkoVladimirAgent'
LOCAL = 'zhmurenko_agent.'
REPOSITORY = next(p for p in PACKAGE.parents if (p / '.git').exists())
BASE_CONFIG = REPOSITORY / 'experiments' / 'zhmurenko_agent.yaml'
TRAIN_SCENARIOS = ('mixed', 'competition', 'random', 'pairs', 'wave', 'wave2')
TOPOLOGIES = ('complete', 'star', 'path', 'inner_coalitions')
VALIDATION_SEEDS = tuple(range(5000, 5006))
FINAL_SEEDS = tuple(range(70000, 70012))
EVAL_SCENARIOS = ('mixed', 'pairs', 'wave', 'wave2', 'askarov_trained', 'tverdokhleb_trained', 'rodina')


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenario_config(base, *, scenario, index, focal, root=None, fast=False, position_index=None):
    raw = config_to_dict(base)
    raw.update(mode='eval', strict=True, episodes_per_seed=1)
    count = raw['agents']['count']
    position = (5 * (index if position_index is None else position_index)) % count
    raw['agents']['overrides'] = {i: spec for i, spec in raw['agents']['overrides'].items() if spec['impl'] != IMPL}
    if scenario == 'cost_shift':
        raw['agents'].update(cost_low=10, cost_high=35)
    raw['network'].update(topology=TOPOLOGIES[(index // len(TRAIN_SCENARIOS)) % 4],
                          mode=('static', 'evolving')[(index // 24) % 2])
    if fast:
        # A documented public network parameter, identical for both Q variants.
        # History/utility-driven edge evolution and normal coalition rules remain.
        raw['network']['w_coop'] = 0
    if scenario not in ('mixed', 'cost_shift'):
        raw['agents']['overrides'] = {}
        default = {'impl': 'social_market_env.baselines.agents:CartelAgent'}
        raw['coalitions']['policy'] = {'impl': 'social_market_env.baselines.coalitions:NoCoalitions'}
        raw['arbiter']['policy'] = {'impl': 'social_market_env.baselines.arbiters:AlwaysObserve'}
        if scenario == 'competition':
            default = {'impl': 'social_market_env.baselines.agents:BestResponseAgent'}
        elif scenario == 'random':
            default = {'impl': 'social_market_env.baselines.agents:RandomAgent', 'kwargs': {'low': 20, 'high': 60}}
        elif scenario == 'pairs':
            raw['coalitions']['policy'] = {'impl': LOCAL + 'components:PairOffers'}
        elif scenario in ('wave', 'wave2'):
            default = {'impl': LOCAL + 'components:PriceWave', 'kwargs': {'rising': scenario == 'wave', 'phase': (index*7)%20}}
        elif scenario.startswith(('askarov_', 'tverdokhleb_')):
            name, status = scenario.split('_')
            default = {'impl': LOCAL + f'references:{name}'}
            if status == 'trained':
                path = Path(root) / 'peer-models' / f'{name}.json'
                if not path.is_file():
                    raise FileNotFoundError(f'Missing trained peer model: {path}')
                default['kwargs'] = {'model_path': str(path)}
        elif scenario == 'rodina':
            default = {'impl': 'rodina_agent:RodinaAgent'}
        else:
            raise ValueError(f'Unknown scenario {scenario}')
        raw['agents']['default'] = default
    raw['agents']['overrides'][position] = focal
    raw['seeds'] = [0]
    return config_from_dict(raw), position


def value(mapping, key, default=0):
    return mapping.get(key, mapping.get(str(key), default))


def play(agent, config, *, position, seed, training, directory=None):
    env = build_environment(config=config, seed=seed)
    # The same public wiring pattern is used by the peers' experiment modules.
    env.agents[position] = agent
    reset_seed = episode_seed(seed=seed, episode=0)
    info = episode_info(env=env, episode_seed=reset_seed)
    before = agent.updates if hasattr(agent, 'updates') else 0
    env.reset(seed=reset_seed, training=training)
    records, td = [], []
    while not env.done:
        record = env.step().record
        if record['events']:
            raise RuntimeError(f"Component contract error: {record['events']}")
        records.append(record)
        if training and hasattr(agent, 'last_learning'):
            td.append(abs(agent.last_learning['td_error']))
    cost = info['costs'][position]
    prices = [value(r['prices'], position, None) for r in records]
    prices = [p for p in prices if p is not None]
    reward = sum(value(r['agent_rewards'], position) for r in records)
    row = dict(seed=seed, position=position, cost=cost, steps=len(records), total_reward=reward,
               mean_price=mean(prices) if prices else None, mean_margin=mean(p-cost for p in prices) if prices else None,
               mean_quantity=mean(value(r['quantities'], position) for r in records),
               gross_profit=sum(value(r['gross_profits'], position) for r in records),
               coalition_payout=sum(value(r['received'], position)-value(r['gross_profits'], position) for r in records),
               coalition_steps=sum(any(position in p['compliers'] for p in r['plans']) for r in records),
               fines=sum(value(r['fines'], position) for r in records),
               survived=bool(value(records[-1]['active'], position)),
               updates=(agent.updates-before) if hasattr(agent, 'updates') else 0,
               td_abs_mean=mean(td) if td else None, topology=config.network.topology, network_mode=config.network.mode,
               network_w_coop=config.network.w_coop)
    row.update(getattr(agent, 'stats', {}))
    for name, attribute in (('decisions', 'decisions'), ('known_states', 'known_state_decisions')):
        if name not in row and hasattr(agent, attribute):
            row[name] = getattr(agent, attribute)
    row['initial_conditions'] = hashlib.sha256(json.dumps(dict(costs=info['costs'], seed=reset_seed,
        network=asdict(config.network), market=asdict(config.market), finance=asdict(config.finance)), sort_keys=True).encode()).hexdigest()
    if directory:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory/'config.json', config_to_dict(config))
        write_json(directory/'episode.json', info)
        write_json(directory/'metrics.json', row)
        with gzip.open(directory/'steps.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        write_json(directory/'completed.json', dict(steps=len(records)))
    return row


def aggregate(rows):
    if not rows:
        return {}
    decisions = sum(r.get('decisions', 0) for r in rows)
    return dict(episodes=len(rows), reward=mean(r['total_reward'] for r in rows),
                reward_std=pstdev(r['total_reward'] for r in rows),
                price=mean([r['mean_price'] for r in rows if r['mean_price'] is not None]) if any(r['mean_price'] is not None for r in rows) else None,
                quantity=mean(r['mean_quantity'] for r in rows), margin=mean([r['mean_margin'] for r in rows if r['mean_margin'] is not None]) if any(r['mean_margin'] is not None for r in rows) else None,
                coalition_steps=sum(r['coalition_steps'] for r in rows), fines=sum(r['fines'] for r in rows),
                survival=mean(r['survived'] for r in rows),
                known_fraction=sum(r.get('known_states', 0) for r in rows)/decisions if decisions else None,
                learned_fraction=sum(r.get('learned_decisions', 0) for r in rows)/decisions if decisions else None,
                changed_from_cold=sum(r.get('differs_from_cold', 0) for r in rows)/decisions if decisions else None)


def policy_signature(agent):
    result = {}
    for base in agent.q:
        legal = list(range(8)) + ([8, 9] if base[0] else []) + ([JOIN] if base[1] not in (0, 7) else []) + [11]
        for direction in (0, 1, 2):
            values, _ = agent._values(base, (*base, direction))
            result['|'.join(map(str, (*base, direction)))] = max(legal, key=lambda i: (values[i], i == 2, -i))
    return result


def validate(agent, base, *, root=None):
    frozen = copy.deepcopy((agent.q, agent.visits, agent.context_q, agent.context_visits, agent.updates))
    rows = []
    weights = {}
    for index, seed in enumerate(VALIDATION_SEEDS):
        for scenario in TRAIN_SCENARIOS:
            config, pos = scenario_config(base, scenario=scenario, index=index*6, position_index=index,
                                           focal={'impl': IMPL, 'kwargs': {'fresh': True}}, fast=True)
            row = play(agent, config, position=pos, seed=seed, training=False)
            rows.append(dict(row, scenario=scenario))
            for key, count in agent.decision_counts.items():
                weights[key] = weights.get(key, 0) + count
    if frozen != (agent.q, agent.visits, agent.context_q, agent.context_visits, agent.updates):
        raise AssertionError('Validation changed Q values')
    return aggregate(rows) | {'state_weights': weights,
        'per_scenario': {s: aggregate([r for r in rows if r['scenario']==s]) for s in TRAIN_SCENARIOS}}


def train_pair(args, base):
    destination = args.root / f'repeat_{args.repeat}'
    start = 0
    if args.resume:
        common = set.intersection(*(set(p.name for p in (destination/name).glob('block_*.json')) for name in args.variants))
        if not common:
            raise ValueError('No complete common checkpoint block to resume')
        block = sorted(common)[-1]
        agents = {name:ZhmurenkoVladimirAgent(model_path=destination/name/block, market=asdict(base.market)) for name in args.variants}
        starts = {a.metadata['episodes'] for a in agents.values()}
        if len(starts) != 1:
            raise ValueError('Paired checkpoint budgets differ')
        start = starts.pop()
        if start >= args.max_episodes:
            raise ValueError('Resume requires a larger maximum episode count')
        episodes = {name:json.loads((destination/name/'training.json').read_text(encoding='utf-8'))[:start] for name in agents}
        validation = [r for r in json.loads((destination/'validation.json').read_text(encoding='utf-8')) if r['trained_episodes']<=start]
        previous = {name:(a.metadata['last_validation'],a.metadata['policy_signature']) for name,a in agents.items()}
        stable = {name:a.metadata['convergence_windows'] for name,a in agents.items()}
        for name in agents:
            for path in (destination/name/'episodes').iterdir():
                if path.is_dir() and int(path.name) >= start:
                    target = destination/'interrupted-blocks'/name/path.name
                    if not path.resolve().is_relative_to(args.root.resolve()) or not target.resolve().is_relative_to(args.root.resolve()):
                        raise ValueError('Unsafe checkpoint recovery path')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():
                        raise FileExistsError(target)
                    path.rename(target)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        agents = {name: ZhmurenkoVladimirAgent(fresh=True, market=asdict(base.market),
                  settings={'use_regime_context': name == 'regime'}) for name in args.variants}
        episodes = {name: [] for name in agents}
        validation = []
        previous, stable = {}, {name: 0 for name in agents}
    for episode in range(start,args.max_episodes):
        seed = 10000 + args.repeat*10000 + episode
        scenario = TRAIN_SCENARIOS[episode % len(TRAIN_SCENARIOS)]
        config, pos = scenario_config(base, scenario=scenario, index=episode+args.repeat,
            focal={'impl': IMPL, 'kwargs': {'fresh': True}}, fast=episode % 10 != 0)
        for name, agent in agents.items():
            row = play(agent, config, position=pos, seed=seed, training=True,
                       directory=destination/name/'episodes'/f'{episode:04d}')
            episodes[name].append(dict(row, episode=episode, scenario=scenario))
        if (episode+1) % args.block != 0:
            continue
        for name, agent in agents.items():
            metrics = validate(agent, base)
            signature = policy_signature(agent)
            old = previous.get(name)
            drift = (sum(weight * (signature.get(key) != old[1].get(key)) for key,weight in metrics['state_weights'].items())
                     / max(sum(metrics['state_weights'].values()),1)) if old else 1.0
            reward_change = abs(metrics['reward']-old[0]['reward'])/max(abs(old[0]['reward']),1) if old else 1.0
            coalition_samples = sum(v[JOIN] for v in agent.visits.values())
            ready = (episode+1 >= args.min_episodes and drift <= .03 and reward_change <= .02
                     and metrics['known_fraction'] >= .90 and coalition_samples >= 100)
            stable[name] = stable[name]+1 if ready else 0
            previous[name] = metrics, signature
            agent.metadata = dict(training_seeds=list(range(10000+args.repeat*10000, seed+1)),
                repeat=args.repeat, episodes=episode+1, selection='last_equal_budget_block_after_joint_stability',
                mechanism='confidence_gated_regime_Q' if name=='regime' else 'shared_context_Q',
                coalition_samples=coalition_samples, convergence_windows=stable[name],
                last_validation=metrics, policy_signature=signature)
            agent.save(destination/name/'model.json')
            agent.save(destination/name/f'block_{episode+1:04d}.json')
            write_json(destination/name/'training.json', episodes[name])
            validation.append(dict(model=name, trained_episodes=episode+1, reward_change=reward_change,
                                   policy_drift=drift, coalition_samples=coalition_samples, **metrics))
            print(f"repeat={args.repeat} {name}: {episode+1} episodes, {agent.updates} updates, states={len(agent.q)}/{len(agent.context_q)}, validation={metrics['reward']:.2f}, drift={drift:.3f}, stable={stable[name]}", flush=True)
        write_json(destination/'validation.json', validation)
        if min(stable.values()) >= args.patience:
            break
    write_json(destination/'summary.json', dict(episodes=episode+1, stability_reached=min(stable.values())>=args.patience,
        models={name:dict(training=aggregate(episodes[name]), validation=previous[name][0], updates=a.updates,
                         states=len(a.q), regime_states=len(a.context_q), coalition_samples=a.metadata['coalition_samples'])
                for name,a in agents.items()}))


def train_peer(args, base):
    activate(args.root)
    from .references import askarov, tverdokhleb
    agent = (askarov if args.peer == 'askarov' else tverdokhleb)(market=asdict(base.market))
    rows = []
    for episode in range(args.max_episodes):
        config, pos = scenario_config(base, scenario=TRAIN_SCENARIOS[episode % 6], index=episode,
            focal={'impl': IMPL, 'kwargs': {'fresh': True}}, fast=episode % 10 != 0)
        row = play(agent, config, position=pos, seed=40000+episode, training=True)
        rows.append(dict(row, episode=episode))
        if (episode+1) % args.block == 0:
            agent.save(args.root/'peer-models'/f'{args.peer}.json')
            write_json(args.root/'peer-models'/f'{args.peer}.training.json', rows)
            print(f'{args.peer}: {episode+1} episodes, {agent.updates} actual updates', flush=True)


def model_variants(root):
    result = [('cold', {'impl': IMPL, 'kwargs': {'fresh': True}}),
              ('economic', {'impl': LOCAL+'components:MarkupStrategy'})]
    for rep in range(3):
        for name in ('flat', 'regime'):
            path = root/f'repeat_{rep}'/name/'model.json'
            result.append((f'{name}_{rep}', {'impl': IMPL, 'kwargs': {'model_path': str(path)}}))
    return result


def evaluate(args, base):
    activate(args.root, v4=False)
    import sys
    if 'torch' in sys.modules:
        raise RuntimeError('The reduced Q-only evaluation must not load PyTorch')
    frozen = {}
    for name, spec in model_variants(args.root):
        path = spec.get('kwargs', {}).get('model_path') or spec.get('kwargs', {}).get('checkpoint_path')
        if path:
            frozen[name] = dict(path=path, sha256=digest(path))
    opponents = {name:dict(path=str(args.root/'peer-models'/f'{name}.json'),
                          sha256=digest(args.root/'peer-models'/f'{name}.json')) for name in ('askarov','tverdokhleb')}
    lock = dict(models=frozen, opponents=opponents, seeds=FINAL_SEEDS, scenarios=EVAL_SCENARIOS,
                source_sha256={p.relative_to(PACKAGE).as_posix():digest(p) for p in PACKAGE.rglob('*.py')
                               if 'tests' not in p.parts and p.name!='analysis.py'})
    lock_path = args.root/'final-lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text(encoding='utf-8')) != json.loads(json.dumps(lock)):
        raise ValueError('Final source/model lock changed')
    write_json(lock_path, lock)
    if args.freeze_only:
        return
    rows = []
    for name, spec in model_variants(args.root):
        if args.models and name not in args.models:
            continue
        for index, seed in enumerate(FINAL_SEEDS):
            for scenario in EVAL_SCENARIOS:
                directory = args.root/'final'/name/scenario/f'seed_{seed}'
                if (directory/'completed.json').exists():
                    row = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
                else:
                    config, pos = scenario_config(base, scenario=scenario, index=index*6, position_index=index,
                                                  focal=spec, root=args.root, fast=False)
                    config.network.mode = ('static','evolving')[(index//4+index%2)%2]
                    agent = instantiate(spec=ComponentSpec(**spec), defaults={'market': asdict(config.market)})
                    row = play(agent, config, position=pos, seed=seed, training=False, directory=directory)
                rows.append(dict(row, policy=name, scenario=scenario))
            if (index+1)%5==0:
                print(f'final {name}: {index+1}/{len(FINAL_SEEDS)} seeds', flush=True)
        for info in (*frozen.values(), *opponents.values()):
            if digest(info['path']) != info['sha256']:
                raise AssertionError('Evaluation modified a saved model')
    suffix = '' if not args.models else '_'+ '_'.join(args.models)
    write_json(args.root/'final'/f'episodes{suffix}.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'train-peer', 'evaluate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=BASE_CONFIG)
    parser.add_argument('--repeat', type=int, choices=(0,1,2), default=0)
    parser.add_argument('--peer', choices=('askarov','tverdokhleb'), default='askarov')
    parser.add_argument('--min-episodes', type=int, default=1000)
    parser.add_argument('--max-episodes', type=int, default=2500)
    parser.add_argument('--block', type=int, default=250)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--variants', nargs='+', choices=('flat','regime'), default=['flat','regime'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--v4-archive', type=Path)
    args = parser.parse_args()
    args.root = output_path(args.root)
    if args.block < 1 or not 1 <= args.min_episodes <= args.max_episodes or args.max_episodes % args.block or args.patience < 1:
        raise ValueError('Invalid training/stability budget')
    base = load_config(args.config)
    if args.stage == 'prepare':
        args.root.mkdir(parents=True, exist_ok=True)
        prepare(args.root, v4_archive=args.v4_archive)
        write_json(args.root/'protocol.json', dict(python=platform.python_version(), scenarios=TRAIN_SCENARIOS,
            validation_seeds=VALIDATION_SEEDS, final_seeds=FINAL_SEEDS, min_episodes=args.min_episodes,
            max_episodes=args.max_episodes, block=args.block, patience=args.patience,
            fast_training='9/10 episodes w_coop=0; every tenth uses canonical w_coop=1; final always canonical'))
    elif args.stage == 'train':
        train_pair(args, base)
    elif args.stage == 'train-peer':
        train_peer(args, base)
    else:
        evaluate(args, base)


if __name__ == '__main__':
    main()
