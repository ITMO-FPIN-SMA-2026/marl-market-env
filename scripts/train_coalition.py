"""Обучить контекстного бандита политики коалиций на смеси арбитров и агентов.

Этап 3 плана. Смесь нужна потому, что ТЗ запрещает настраиваться на одного противника.
Порог 6 % и seeds от 2000 сюда не входят: на них идёт оценка в compare_coalitions.py.

    python scripts/train_coalition.py --episodes 120 --out experiments/coalition_bandit.json

Арбитр preventive берётся из PYTHONPATH, как и в compare_coalitions.py.
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from compare_coalitions import ARBITERS, COUNT, HORIZON, POPULATIONS

from coalition_policy import AdaptiveCoalitionPolicy
from social_market_env.config import config_from_dict
from social_market_env.runner.experiment import build_environment

TRAIN_ARBITERS: tuple[str, ...] = ('observe', 'threshold', 'preventive', 'preventive4')
TRAIN_POPULATIONS: tuple[str, ...] = ('cartel', 'patient90', 'patient50', 'flaky3')
TOPOLOGIES: tuple[str, ...] = ('random', 'complete', 'path', 'star')
FIRST_TRAIN_SEED: int = 100


def main() -> None:
    """Обучить бандита и сохранить его."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=120)
    parser.add_argument('--out', type=Path, default=Path('experiments/coalition_bandit.json'))
    args = parser.parse_args()
    policy = AdaptiveCoalitionPolicy(horizon=HORIZON, learning='bandit', payout='myerson')
    scenarios = list(product(TRAIN_ARBITERS, TRAIN_POPULATIONS, TOPOLOGIES))
    np.random.default_rng(0).shuffle(scenarios)
    for episode in range(args.episodes):
        arbiter, population, topology = scenarios[episode % len(scenarios)]
        seed = FIRST_TRAIN_SEED + episode
        config = config_from_dict({
            'horizon': HORIZON,
            'strict': True,
            'agents': {'count': COUNT, **POPULATIONS[population]},
            'arbiter': {'policy': ARBITERS[arbiter]},
            'network': {'topology': topology, 'mode': 'evolving'},
        })
        env = build_environment(config=config, seed=seed)
        # Одна политика на все эпизоды: иначе бандит начинал бы обучение заново.
        env.coalition_policy = policy
        env.reset(seed=seed, training=True)
        while not env.done:
            env.step()
        if (episode + 1) % 20 == 0:
            print(f'episode {episode + 1}: updates per arm {policy.learner.updates.tolist()}', flush=True)
    policy.learner.save(args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
