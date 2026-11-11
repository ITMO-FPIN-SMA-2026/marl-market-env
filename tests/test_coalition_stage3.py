"""Тесты этапа 3 политики коалиций: контекстный бандит и его встраивание."""

import numpy as np
import pytest

from coalition_policy import AdaptiveCoalitionPolicy, PolicySettings
from coalition_policy.learner import ARMS, FEATURES, LinearThompson
from social_market_env.config import config_from_dict
from social_market_env.runner.experiment import build_environment

POLICY = 'coalition_policy:AdaptiveCoalitionPolicy'


def context(*values: float) -> np.ndarray:
    return np.array([*values, *([0.0] * (len(FEATURES) - len(values)))])


def test_bandit_learns_best_arm_for_each_context() -> None:
    learner = LinearThompson()
    first, second = context(1.0, 1.0), context(1.0, 0.0)
    for _ in range(50):
        learner.update(arm=0, context=first, reward=1.0)
        learner.update(arm=1, context=first, reward=0.0)
        learner.update(arm=0, context=second, reward=0.0)
        learner.update(arm=1, context=second, reward=1.0)

    assert learner.choose(first, explore=False) == 0
    assert learner.choose(second, explore=False) == 1


def test_exploration_tries_several_arms_and_greedy_is_stable() -> None:
    learner = LinearThompson()
    state = context(1.0)

    assert len({learner.choose(state, explore=True) for _ in range(50)}) > 1
    assert len({learner.choose(state, explore=False) for _ in range(50)}) == 1


def test_checkpoint_round_trip_and_version_check(tmp_path) -> None:
    learner = LinearThompson()
    learner.update(arm=2, context=context(1.0, 0.5), reward=3.0)
    path = tmp_path / 'bandit.json'
    learner.save(path)

    restored = LinearThompson()
    restored.load(path)

    assert restored.precision == pytest.approx(learner.precision)
    assert restored.moment == pytest.approx(learner.moment)
    assert restored.updates.tolist() == learner.updates.tolist()
    path.write_text('{"version": 0}', encoding='utf-8')
    with pytest.raises(ValueError):
        restored.load(path)


def test_bandit_needs_adaptive_pricing() -> None:
    with pytest.raises(ValueError):
        PolicySettings(learning='bandit', pricing='fixed')


def play(*, policy=None, training=True, kwargs=None, horizon=24) -> list[dict]:
    config = config_from_dict({
        'horizon': horizon,
        'strict': True,
        'agents': {'count': 6, 'default': {'impl': 'social_market_env.baselines.agents:CartelAgent'}},
        'coalitions': {'policy': {'impl': POLICY, 'kwargs': {'horizon': horizon, **(kwargs or {})}}},
        'network': {'topology': 'complete', 'mode': 'evolving'},
    })
    env = build_environment(config=config, seed=0)
    if policy is not None:
        env.coalition_policy = policy
    env.reset(seed=0, training=training)
    records = []
    while not env.done:
        records.append(env.step().record)
    return records


def test_policy_updates_bandit_once_per_block_when_training() -> None:
    policy = AdaptiveCoalitionPolicy(horizon=24, learning='bandit', block_steps=8)

    play(policy=policy)

    assert policy.learner.updates.sum() == 3


def test_policy_does_not_learn_during_evaluation() -> None:
    policy = AdaptiveCoalitionPolicy(horizon=24, learning='bandit')

    play(policy=policy, training=False)

    assert policy.learner.updates.sum() == 0


def test_bandit_survives_episode_reset() -> None:
    policy = AdaptiveCoalitionPolicy(horizon=24, learning='bandit')

    play(policy=policy)
    play(policy=policy)

    assert policy.learner.updates.sum() == 6


def test_arm_cap_limits_price() -> None:
    policy = AdaptiveCoalitionPolicy(horizon=40, learning='bandit', start_gap=0.2)
    # Обученная модель всегда выбирает осторожный вариант.
    policy.learner.moment[ARMS.index('cautious'), 0] = 100.0

    records = play(policy=policy, training=False, horizon=40)

    gaps = [1 - record['social_welfare'] / record['target_social_welfare'] for record in records]
    assert max(gaps) <= 0.05 + 1e-6


def test_learning_off_ignores_bandit() -> None:
    assert play(kwargs={'learning': 'off'}) == play(kwargs={})
