"""Post-run individual metrics from the officially returned step records."""

import json
import hashlib
import platform
import subprocess
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from social_market_env.runner.experiment import episode_seed

from .settings import PACKAGE_ROOT


def experiment_metadata():
    sources={path.relative_to(PACKAGE_ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
             for path in PACKAGE_ROOT.rglob('*') if path.is_file() and path.suffix in ('.py','.yaml')}
    repository=next(parent for parent in PACKAGE_ROOT.parents if (parent/'.git').exists())
    return {'python':platform.python_version(),'dependencies':{name:version(name) for name in ('torch','numpy','PyYAML','networkx','pytest')},
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip(),
            'source_hashes':sources,'source_digest':hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()}


def agent_value(values: dict, agent_id: int, default=0.0):
    return values.get(agent_id, values.get(str(agent_id), default))


def episode_metrics(records: list[dict], *, agent_id: int, cost: float | None = None) -> dict[str, Any]:
    prices = [float(agent_value(record['prices'], agent_id)) for record in records
              if agent_id in record['prices'] or str(agent_id) in record['prices']]
    offers = sum(any(agent_id in plan['members'] for plan in record['plans']) for record in records)
    joins = sum(any(agent_id in plan['compliers'] + plan['defectors'] for plan in record['plans']) for record in records)
    fines = [float(agent_value(record['fines'], agent_id)) for record in records]
    if cost is None:
        # Post-run algebra on the officially returned OWN realized outcome, never a model input.
        for record in records:
            quantity=agent_value(record['quantities'],agent_id)
            if quantity>1e-12:
                cost=agent_value(record['prices'],agent_id)-agent_value(record['gross_profits'],agent_id)/quantity
                break
    margins=[price-cost for price in prices] if cost is not None else []
    return {'steps': len(records), 'total_reward': sum(float(agent_value(r['agent_rewards'], agent_id)) for r in records),
            'fine_count': sum(fine > 0 for fine in fines), 'total_fines': sum(fines),
            'link_fees': sum(float(agent_value(r['link_fees'], agent_id)) for r in records),
            'offers': offers, 'joins': joins, 'join_rate': joins / offers if offers else 0.0,
            'participation_rate': len(prices) / len(records) if records else 0.0,
            'mean_price': mean(prices) if prices else None, 'price_std': pstdev(prices) if prices else None,
            'min_price': min(prices) if prices else None, 'max_price': max(prices) if prices else None,
            'cost':cost,'mean_margin':mean(margins) if margins else None,
            'zero_margin_steps':sum(abs(margin)<1e-7 for margin in margins),
            'total_gross_profit':sum(float(agent_value(record['gross_profits'],agent_id)) for record in records),
            'coalition_payout':sum(float(agent_value(record['received'],agent_id)-agent_value(record['gross_profits'],agent_id)) for record in records),
            'eliminated': bool(records and not agent_value(records[-1]['active'], agent_id, True)),
            'events': sum(len(record['events']) for record in records)}


def summarize(episodes: list[dict]) -> dict[str, Any]:
    by_seed = defaultdict(list)
    for episode in episodes:
        by_seed[episode['seed']].append(episode['total_reward'])
    rewards = [episode['total_reward'] for episode in episodes]
    per_seed = {str(seed): mean(values) for seed, values in sorted(by_seed.items())}
    offers = sum(e['offers'] for e in episodes)
    return {'episodes': len(episodes), 'mean_reward': mean(rewards), 'median_reward': median(rewards),
            'reward_std': pstdev(rewards), 'mean_reward_by_seed': per_seed,
            'seed_mean_std': pstdev(per_seed.values()),
            'fine_count': sum(e['fine_count'] for e in episodes),
            'total_fines': sum(e['total_fines'] for e in episodes),
            'link_fees': sum(e['link_fees'] for e in episodes),
            'join_rate': sum(e['joins'] for e in episodes) / offers if offers else 0.0,
            'mean_participation_rate': mean(e['participation_rate'] for e in episodes),
            'mean_price': mean(e['mean_price'] for e in episodes if e['mean_price'] is not None)
            if any(e['mean_price'] is not None for e in episodes) else None,
            'events': sum(e['events'] for e in episodes), 'eliminations': sum(e['eliminated'] for e in episodes)}


def play_episode(env, *, seed: int, episode: int, training: bool, directory: Path, learner=None):
    """Use reset/done/step only; never pass records or hidden state to the policy."""
    directory.mkdir(parents=True, exist_ok=False)
    reset_seed = episode_seed(seed=seed, episode=episode)
    env.reset(seed=reset_seed, training=training)
    records, learning = [], []
    with (directory / 'steps.jsonl').open('w', encoding='utf-8') as stream:
        while not env.done:
            record = env.step().record
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            if learner is not None:
                learning.extend(dict(row,seed=seed,episode=episode) for row in learner.drain_learning_metrics())
    info={'seed':seed,'episode':episode,'reset_seed':reset_seed,'training':training}
    if learner is not None:
        info.update(learner.episode_info)
    (directory/'episode.json').write_text(json.dumps(info,indent=2,allow_nan=False),encoding='utf-8')
    return records, learning, reset_seed
