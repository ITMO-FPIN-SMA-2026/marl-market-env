"""Обязательные эксперименты ТЗ для коалиции: топология, структура, эволюция сети, арбитраж.

Этап 5 плана. Каждый эпизод сохраняется строкой в CSV, сводка печатается со средним
и 95 % бутстреп-интервалом по seeds. Seeds от 3000 не использовались ни в настройке,
ни в обучении бандита.

    python -m coalitions_badretdinov.research.tz_experiments topology --seeds 10
    python -m coalitions_badretdinov.research.tz_experiments structure --seeds 10
    python -m coalitions_badretdinov.research.tz_experiments evolution --seeds 10
    python -m coalitions_badretdinov.research.tz_experiments arbitrage --seeds 10

Арбитры preventive и adaptive берутся из пакетов preventive_arbiter и arbiter_lab,
агент askarov — из askarov_agent: их нужно положить в PYTHONPATH.
"""

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any

import networkx as nx
import numpy as np

from social_market_env.config import config_from_dict
from social_market_env.runner.experiment import build_environment

HORIZON = 64
COUNT = 12
FIRST_SEED = 3000
OURS = 'coalitions_badretdinov:AdaptiveCoalitionPolicy'
LAB = 'coalitions_badretdinov.research.agents:'

POPULATIONS: dict[str, dict[str, Any]] = {
    'patient90': {'default': {'impl': LAB + 'PatientAgent', 'kwargs': {'patience': 0.9}}},
    'askarov': {'default': {'impl': 'askarov_agent:AskarovAgent'}},
}
ARBITERS: dict[str, dict[str, Any]] = {
    'observe': {'impl': 'social_market_env.baselines.arbiters:AlwaysObserve'},
    'threshold': {'impl': 'social_market_env.baselines.arbiters:ThresholdArbiter',
                  'kwargs': {'jump_threshold': 4, 'fine': 50}},
    'preventive': {'impl': 'preventive_arbiter:PreventiveArbiter'},
    # Учится внутри эпизода: сравнивает отклик рынка на штраф и на профилактику.
    # Это не MARL, а промежуточная замена: обученному PPO-арбитру нужен PyTorch.
    'adaptive': {'impl': 'arbiter_lab:AdaptiveArbiter'},
}
TOPOLOGIES: tuple[str, ...] = ('complete', 'star', 'path', 'inner_coalitions')
H_BINS: tuple[float, ...] = (-0.2, 0.2, 0.6)


def policy(structure: str, mode: str) -> dict[str, Any]:
    # Режим сети нужен политике явно: в наблюдении его нет, а в статичной сети связи не рвутся.
    return {'impl': OURS, 'kwargs': {'horizon': HORIZON, 'structure': structure, 'network_mode': mode}}


def grid(experiment: str) -> list[dict[str, str]]:
    """Вернуть сочетания условий эксперимента."""
    if experiment == 'topology':
        # Статичная сеть: влияние именно стартовой формы, без её изменения.
        return [dict(topology=t, mode='static', structure=s, arbiter='observe', agents='patient90')
                for t in TOPOLOGIES for s in ('antagonism', 'adaptive')]
    if experiment == 'structure':
        return [dict(topology=t, mode='static', structure=s, arbiter='observe', agents=a)
                for t in ('complete', 'inner_coalitions') for a in POPULATIONS
                for s in ('antagonism', 'partial', 'grand', 'adaptive')]
    if experiment == 'evolution':
        return [dict(topology=t, mode=m, structure='adaptive', arbiter='observe', agents='patient90')
                for t in (*TOPOLOGIES, 'random') for m in ('static', 'evolving')]
    if experiment == 'arbitrage':
        return [dict(topology='random', mode='evolving', structure=s, arbiter=r, agents=a)
                for a in POPULATIONS for r in ARBITERS for s in ('antagonism', 'adaptive')]
    raise ValueError(f'unknown experiment {experiment}')


def play(condition: dict[str, str], seed: int) -> tuple[list[dict], nx.Graph]:
    """Сыграть эпизод и вернуть записи шагов и стартовый граф."""
    arbiter = dict(ARBITERS[condition['arbiter']])
    if condition['arbiter'] in ('preventive', 'adaptive'):
        arbiter['kwargs'] = {**arbiter.get('kwargs', {}), 'network_mode': condition['mode']}
    config = config_from_dict({
        'horizon': HORIZON,
        'strict': True,
        'agents': {'count': COUNT, **POPULATIONS[condition['agents']]},
        'coalitions': {'policy': policy(condition['structure'], condition['mode'])},
        'arbiter': {'policy': arbiter},
        'network': {'topology': condition['topology'], 'mode': condition['mode']},
    })
    env = build_environment(config=config, seed=seed)
    env.reset(seed=seed, training=False)
    records = []
    while not env.done:
        records.append(env.step().record)
    start = nx.Graph()
    start.add_nodes_from(range(COUNT))
    start.add_edges_from(tuple(map(int, key.split('-'))) for key in records[0]['edges'])
    return records, start


def episode_metrics(records: list[dict], start: nx.Graph) -> dict[str, float]:
    """Вернуть показатели эпизода."""
    totals = defaultdict(float)
    for record in records:
        for agent, reward in record['agent_rewards'].items():
            totals[int(agent)] += reward
    rewards = np.array([totals[agent] for agent in range(COUNT)])
    plans = [plan for record in records for plan in record['plans']]
    offered = sum(len(plan['members']) for plan in plans)
    hub = max(range(COUNT), key=start.degree)
    last = records[-10:]
    return {
        'profit': rewards.sum() / (COUNT * len(records)),
        'gap': fmean(1 - r['social_welfare'] / r['target_social_welfare'] for r in records),
        'harm': fmean(any(p['harmful'] for p in r['plans']) for r in records),
        'kept': sum(len(p['compliers']) for p in plans) / offered if offered else 0.0,
        'cover': offered / (COUNT * len(records)),
        'plan_size': offered / len(plans) if plans else 0.0,
        'lifetime': _mean_lifetime(records),
        'gini': _gini(rewards),
        'hub_ratio': rewards[hub] / rewards.mean() if rewards.mean() else 0.0,
        'edges_start': start.number_of_edges(),
        'edges_end': fmean(len(r['edges']) for r in last),
        'h_end': fmean(fmean(r['edges'].values()) if r['edges'] else 0.0 for r in last),
    }


def _gini(values: np.ndarray) -> float:
    """Неравенство наград: 0 — у всех поровну, к 1 — всё у одного."""
    ordered = np.sort(values)
    if ordered.sum() <= 0:
        return 0.0
    ranks = np.arange(1, ordered.size + 1)
    return float((2 * ranks - ordered.size - 1) @ ordered / (ordered.size * ordered.sum()))


def _mean_lifetime(records: list[dict]) -> float:
    """Сколько шагов подряд в среднем живёт один состав плана."""
    lengths: list[int] = []
    running: dict[frozenset[int], int] = {}
    for record in records:
        current = {frozenset(p['members']) for p in record['plans']}
        lengths += [length for group, length in running.items() if group not in current]
        running = {group: running.get(group, 0) + 1 for group in current}
    lengths += list(running.values())
    return fmean(lengths) if lengths else 0.0


def pair_transitions(records: list[dict]) -> dict[str, list[int]]:
    """Посчитать, как часто пара оказывается в одном соблюдённом плане на следующем шаге.

    Ключ — состояние пары сейчас: есть ли связь, корзина памяти H, были ли вместе, дорогая ли связь.
    Значение — [сколько раз было такое состояние, сколько раз пара потом соблюдала план вместе].
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    together = [_together(record) for record in records]
    for step in range(len(records) - 1):
        record = records[step]
        for first, second in combinations(range(COUNT), 2):
            key = f'{first}-{second}'
            if key not in record['edges']:
                state = 'no_edge'
            else:
                h_bin = sum(record['edges'][key] >= edge for edge in H_BINS)
                costly = record['link_costs'].get(key, 0.0) > 0
                state = f'h{h_bin}|{"with" if (first, second) in together[step] else "apart"}|' \
                        f'{"costly" if costly else "free"}'
            counts[state][0] += 1
            counts[state][1] += int((first, second) in together[step + 1])
    return dict(counts)


def _together(record: dict) -> set[tuple[int, int]]:
    return {pair for plan in record['plans'] for pair in combinations(sorted(plan['compliers']), 2)}


def bootstrap(values: list[float], *, rounds: int = 2000) -> tuple[float, float, float]:
    """Вернуть среднее и 95 % интервал по seeds."""
    data = np.array(values)
    rng = np.random.default_rng(0)
    means = rng.choice(data, size=(rounds, data.size)).mean(axis=1)
    return float(data.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    """Прогнать эксперимент, сохранить эпизоды и напечатать сводку."""
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['topology', 'structure', 'evolution', 'arbitrage'])
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--out', type=Path, default=Path('runs/tz'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    transitions: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    traces: dict[str, list[list[float]]] = {}
    for condition in grid(args.experiment):
        name = '|'.join(condition.values())
        trace = []
        for seed in range(FIRST_SEED, FIRST_SEED + args.seeds):
            records, start = play(condition, seed)
            rows.append({**condition, 'seed': seed, **episode_metrics(records, start)})
            for state, (seen, joined) in pair_transitions(records).items():
                transitions[state][0] += seen
                transitions[state][1] += joined
            trace.append([len(r['edges']) for r in records])
        traces[name] = np.mean(trace, axis=0).round(3).tolist()
        print(f'done {name}', flush=True)
    with (args.out / f'{args.experiment}.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / f'{args.experiment}_pairs.json').write_text(json.dumps(transitions, indent=1), encoding='utf-8')
    (args.out / f'{args.experiment}_edges.json').write_text(json.dumps(traces), encoding='utf-8')
    summarize(rows)


def summarize(rows: list[dict[str, Any]]) -> None:
    """Напечатать таблицу: среднее [95 % интервал] по каждому условию."""
    metrics = [key for key in rows[0] if key not in {'topology', 'mode', 'structure', 'arbiter', 'agents', 'seed'}]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row['topology'], row['mode'], row['structure'], row['arbiter'], row['agents'])].append(row)
    for key, group in groups.items():
        cells = []
        for metric in metrics:
            mean, low, high = bootstrap([row[metric] for row in group])
            cells.append(f'{metric}={mean:.2f}[{low:.2f};{high:.2f}]')
        print(' | '.join(key), '::', ' '.join(cells), flush=True)


if __name__ == '__main__':
    main()
