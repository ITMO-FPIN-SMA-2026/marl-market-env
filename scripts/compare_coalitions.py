"""Сравнить политики коалиций: проверка критерия готовности этапа 1 плана.

Запуск из корня репозитория:

    python scripts/compare_coalitions.py --seeds 10

Чтобы добавить профилактического арбитра, положите его пакет в PYTHONPATH и передайте --preventive.
"""

import argparse
import time
from statistics import fmean
from typing import Any

from social_market_env.config import config_from_dict
from social_market_env.runner.experiment import build_environment

BASELINE = 'social_market_env.baselines.coalitions:'
OURS = 'coalition_policy:AdaptiveCoalitionPolicy'

POLICIES: dict[str, dict[str, Any]] = {
    'none': {'impl': BASELINE + 'NoCoalitions'},
    'components/equal': {'impl': BASELINE + 'ConnectedComponentsCoalitions', 'kwargs': {'payout': 'equal'}},
    'adaptive/cautious': {'impl': OURS, 'kwargs': {'price_level': 'cautious'}},
    'adaptive/moderate': {'impl': OURS, 'kwargs': {'price_level': 'moderate'}},
    'adaptive/bold': {'impl': OURS, 'kwargs': {'price_level': 'bold'}},
    'adaptive/cartel': {'impl': OURS, 'kwargs': {'price_level': 'cartel'}},
    'partial/cartel': {'impl': OURS, 'kwargs': {'structure': 'partial', 'price_level': 'cartel'}},
}
ARBITERS: dict[str, dict[str, Any]] = {
    'observe': {'impl': 'social_market_env.baselines.arbiters:AlwaysObserve'},
    'threshold': {'impl': 'social_market_env.baselines.arbiters:ThresholdArbiter',
                  'kwargs': {'jump_threshold': 4, 'fine': 50}},
}
PREVENTIVE: dict[str, Any] = {'impl': 'preventive_arbiter:PreventiveArbiter'}
TOPOLOGIES: tuple[str, ...] = ('random', 'complete', 'star', 'path', 'inner_coalitions')


def play(*, policy: dict[str, Any], arbiter: dict[str, Any], topology: str, seed: int) -> dict[str, float]:
    """Сыграть один эпизод и вернуть его показатели."""
    config = config_from_dict({
        'horizon': 64,
        'strict': True,
        'agents': {'count': 12, 'default': {'impl': 'social_market_env.baselines.agents:CartelAgent'}},
        'coalitions': {'policy': policy},
        'arbiter': {'policy': arbiter},
        'network': {'topology': topology, 'mode': 'evolving'},
    })
    env = build_environment(config=config, seed=seed)
    timings: list[float] = []
    original = env.coalition_policy.act

    def timed(observation: Any) -> Any:
        started = time.perf_counter()
        result = original(observation)
        timings.append(time.perf_counter() - started)
        return result

    env.coalition_policy.act = timed
    env.reset(seed=seed, training=False)
    records = []
    while not env.done:
        records.append(env.step().record)
    return {
        'profit': sum(sum(record['agent_rewards'].values()) for record in records) / (12 * len(records)),
        'gap': fmean(1 - record['social_welfare'] / record['target_social_welfare'] for record in records),
        'harm': fmean(any(plan['harmful'] for plan in record['plans']) for record in records),
        'edges': fmean(len(record['edges']) for record in records[-10:]),
        'rejected': sum(len(record['rejected_plans']) for record in records),
        'act_ms': 1000 * fmean(timings),
        'act_max_ms': 1000 * max(timings),
    }


def main() -> None:
    """Напечатать таблицу по всем сочетаниям арбитра, топологии и политики."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--preventive', action='store_true')
    parser.add_argument('--arbiter', help='оставить только этого арбитра')
    parser.add_argument('--topologies', nargs='+', default=TOPOLOGIES)
    args = parser.parse_args()
    arbiters = {**ARBITERS, **({'preventive': PREVENTIVE} if args.preventive else {})}
    if args.arbiter:
        arbiters = {args.arbiter: arbiters[args.arbiter]}
    print(f'{"arbiter":10} {"topology":17} {"policy":18} {"profit":>7} {"gap%":>6} {"harm%":>6} '
          f'{"edges":>6} {"rej":>4} {"act ms":>7} {"max ms":>7}')
    for arbiter_name, arbiter in arbiters.items():
        for topology in args.topologies:
            for name, policy in POLICIES.items():
                runs = [play(policy=policy, arbiter=arbiter, topology=topology, seed=seed)
                        for seed in range(args.seeds)]
                mean = {key: fmean(run[key] for run in runs) for key in runs[0]}
                print(f'{arbiter_name:10} {topology:17} {name:18} {mean["profit"]:7.1f} {100 * mean["gap"]:6.1f} '
                      f'{100 * mean["harm"]:6.1f} {mean["edges"]:6.1f} {sum(run["rejected"] for run in runs):4.0f} '
                      f'{mean["act_ms"]:7.1f} {max(run["act_max_ms"] for run in runs):7.1f}', flush=True)


if __name__ == '__main__':
    main()
