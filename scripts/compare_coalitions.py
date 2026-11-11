"""Сравнить политики коалиций на разных агентах и арбитрах: критерии готовности этапов плана.

Запуск из корня репозитория:

    python scripts/compare_coalitions.py --seeds 5 --arbiters observe --populations cartel patient90

Арбитр preventive и популяции askarov требуют, чтобы пакеты preventive_arbiter и askarov_agent
лежали в PYTHONPATH: их нет в этой ветке. --warmup N обучает агентов N эпизодов перед оценкой.
"""

import argparse
import time
from statistics import fmean
from typing import Any

from social_market_env.config import config_from_dict
from social_market_env.runner.experiment import build_environment

BASELINE = 'social_market_env.baselines.coalitions:'
OURS = 'coalition_policy:AdaptiveCoalitionPolicy'
HORIZON = 64
COUNT = 12


def ours(**kwargs: Any) -> dict[str, Any]:
    return {'impl': OURS, 'kwargs': {'horizon': HORIZON, 'diagnostics': True, **kwargs}}


POLICIES: dict[str, dict[str, Any]] = {
    'none': {'impl': BASELINE + 'NoCoalitions'},
    'components': {'impl': BASELINE + 'ConnectedComponentsCoalitions', 'kwargs': {'payout': 'equal'}},
    'stage1': ours(pricing='fixed', price_level='moderate', payout='myerson', memory=False),
    'adaptive': ours(),
    'no-memory': ours(memory=False),
    'fixed-bold': ours(pricing='fixed', price_level='bold', payout='myerson', memory=False),
    # Контрольные версии бандита: каждый его вариант, зафиксированный на весь эпизод.
    'cap-cautious': ours(price_level='cautious', payout='myerson'),
    'cap-moderate': ours(price_level='moderate', payout='myerson'),
    'cap-bold': ours(price_level='bold', payout='myerson'),
    'cap-cartel': ours(price_level='cartel', payout='myerson'),
    'bandit': ours(learning='bandit', payout='myerson', checkpoint_path='experiments/coalition_bandit.json'),
    'equal': ours(payout='equal'),
    'shapley': ours(payout='shapley'),
    'myerson': ours(payout='myerson'),
    'owen': ours(payout='owen'),
}
ARBITERS: dict[str, dict[str, Any]] = {
    'observe': {'impl': 'social_market_env.baselines.arbiters:AlwaysObserve'},
    'threshold': {'impl': 'social_market_env.baselines.arbiters:ThresholdArbiter',
                  'kwargs': {'jump_threshold': 4, 'fine': 50}},
    'preventive': {'impl': 'preventive_arbiter:PreventiveArbiter'},
    # Тот же арбитр с другим порогом: проверка, что политика не подогнана под 8 %.
    'preventive4': {'impl': 'preventive_arbiter:PreventiveArbiter', 'kwargs': {'welfare_gap_threshold': 0.04}},
    # Этого арбитра нет в обучении бандита: на нём проверяется обобщение.
    'preventive6': {'impl': 'preventive_arbiter:PreventiveArbiter', 'kwargs': {'welfare_gap_threshold': 0.06}},
}
LAB = 'coalition_lab.agents:'
FLAKY = {'impl': LAB + 'FlakyAgent', 'kwargs': {'betray_rate': 0.3}}
POPULATIONS: dict[str, dict[str, Any]] = {
    'cartel': {'default': {'impl': 'social_market_env.baselines.agents:CartelAgent'}},
    'patient90': {'default': {'impl': LAB + 'PatientAgent', 'kwargs': {'patience': 0.9}}},
    'patient50': {'default': {'impl': LAB + 'PatientAgent', 'kwargs': {'patience': 0.5}}},
    'myopic': {'default': {'impl': LAB + 'PatientAgent', 'kwargs': {'patience': 0.0, 'refuse': True}}},
    'flaky3': {'default': {'impl': LAB + 'PatientAgent', 'kwargs': {'patience': 0.9}},
               'overrides': {1: FLAKY, 5: FLAKY, 9: FLAKY}},
    'askarov': {'default': {'impl': 'askarov_agent:AskarovAgent'}},
}
TOPOLOGIES: tuple[str, ...] = ('random', 'complete', 'star', 'path', 'inner_coalitions')


def play(
    *, policy: dict[str, Any], arbiter: dict[str, Any], population: dict[str, Any], topology: str, seed: int,
    warmup: int,
) -> dict[str, float]:
    """Сыграть эпизод оценки и вернуть его показатели."""
    config = config_from_dict({
        'horizon': HORIZON,
        'strict': True,
        'agents': {'count': COUNT, **population},
        'coalitions': {'policy': policy},
        'arbiter': {'policy': arbiter},
        'network': {'topology': topology, 'mode': 'evolving'},
    })
    env = build_environment(config=config, seed=seed)
    for episode in range(warmup):
        env.reset(seed=1000 + 100 * seed + episode, training=True)
        while not env.done:
            env.step()
    timings: list[float] = []
    gaps: list[float] = []
    original = env.coalition_policy.act

    def timed(observation: Any) -> Any:
        started = time.perf_counter()
        result = original(observation)
        timings.append(time.perf_counter() - started)
        report = getattr(env.coalition_policy, 'last_report', {})
        if 'core_gap' in report:
            gaps.append(report['core_gap'])
        return result

    env.coalition_policy.act = timed
    env.reset(seed=seed, training=False)
    records = []
    while not env.done:
        records.append(env.step().record)
    plans = [plan for record in records for plan in record['plans']]
    offered = sum(len(plan['members']) for plan in plans)
    return {
        'profit': sum(sum(record['agent_rewards'].values()) for record in records) / (COUNT * len(records)),
        'gap': fmean(1 - record['social_welfare'] / record['target_social_welfare'] for record in records),
        'harm': fmean(any(plan['harmful'] for plan in record['plans']) for record in records),
        'kept': sum(len(plan['compliers']) for plan in plans) / offered if offered else 0.0,
        'coverage': offered / (COUNT * len(records)),
        'edges': fmean(len(record['edges']) for record in records[-10:]),
        'core': fmean(gaps) if gaps else 0.0,
        'rejected': sum(len(record['rejected_plans']) for record in records),
        'act_ms': 1000 * fmean(timings),
        'act_max_ms': 1000 * max(timings),
    }


def main() -> None:
    """Напечатать таблицу по всем сочетаниям."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--arbiters', nargs='+', default=['observe'], choices=sorted(ARBITERS))
    parser.add_argument('--populations', nargs='+', default=['cartel'], choices=sorted(POPULATIONS))
    parser.add_argument('--policies', nargs='+', default=list(POLICIES), choices=sorted(POLICIES))
    parser.add_argument('--topologies', nargs='+', default=['random'], choices=TOPOLOGIES)
    args = parser.parse_args()
    print(f'{"arbiter":10} {"agents":10} {"topology":10} {"policy":11} {"profit":>7} {"gap%":>6} {"harm%":>6} '
          f'{"kept%":>6} {"cover%":>6} {"edges":>6} {"core%":>6} {"rej":>4} {"ms":>6} {"max":>6}', flush=True)
    for arbiter in args.arbiters:
        for population in args.populations:
            for topology in args.topologies:
                for name in args.policies:
                    runs = [play(policy=POLICIES[name], arbiter=ARBITERS[arbiter], topology=topology, seed=seed,
                                 population=POPULATIONS[population], warmup=args.warmup)
                            for seed in range(args.first_seed, args.first_seed + args.seeds)]
                    mean = {key: fmean(run[key] for run in runs) for key in runs[0]}
                    print(f'{arbiter:10} {population:10} {topology[:10]:10} {name:11} {mean["profit"]:7.1f} '
                          f'{100 * mean["gap"]:6.1f} {100 * mean["harm"]:6.1f} {100 * mean["kept"]:6.1f} '
                          f'{100 * mean["coverage"]:6.1f} {mean["edges"]:6.1f} {100 * mean["core"]:6.2f} '
                          f'{sum(run["rejected"] for run in runs):4.0f} {mean["act_ms"]:6.1f} '
                          f'{max(run["act_max_ms"] for run in runs):6.0f}', flush=True)


if __name__ == '__main__':
    main()
