"""Turn an experiment config into episodes and logs."""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from social_market_env.config import ExperimentConfig
from social_market_env.core.guard import CallGuard
from social_market_env.core.seeding import spawn_seeds
from social_market_env.environment import SocialMarketEnvironment
from social_market_env.recording.summary import summarize_episode
from social_market_env.recording.writer import EpisodeWriter, RunRecorder
from social_market_env.runner.loader import instantiate

logger = logging.getLogger(__name__)

COST_STREAM: int = 0
EPISODE_STREAM: int = 1


def draw_costs(*, count: int, low: float, high: float, seed: int) -> dict[int, float]:
    """Draw unit costs uniformly for agents 0..count-1."""
    rng = np.random.default_rng(seed)
    return {agent_id: round(float(cost), 2) for agent_id, cost in enumerate(rng.uniform(low, high, size=count))}


def episode_seed(*, seed: int, episode: int) -> int:
    """Return the env.reset seed of one episode of an experiment seed."""
    episodes_root = spawn_seeds(seed=seed, count=EPISODE_STREAM + 1)[EPISODE_STREAM]
    return spawn_seeds(seed=episodes_root, count=episode + 1)[episode]


def build_environment(*, config: ExperimentConfig, seed: int) -> SocialMarketEnvironment:
    """Instantiate all components from the config and wire them into an environment."""
    agents_config = config.agents
    cost_seed = spawn_seeds(seed=seed, count=COST_STREAM + 1)[COST_STREAM]
    costs = draw_costs(
        count=agents_config.count, low=agents_config.cost_low, high=agents_config.cost_high, seed=cost_seed
    )
    # Baselines that model the market must see the experiment's market, not MarketConfig() defaults.
    defaults = {'market': asdict(config.market)}
    agents = {
        agent_id: instantiate(spec=agents_config.overrides.get(agent_id, agents_config.default), defaults=defaults)
        for agent_id in costs
    }
    return SocialMarketEnvironment(
        agents=agents,
        costs=costs,
        coalition_policy=instantiate(spec=config.coalitions.policy, defaults=defaults),
        arbiter=instantiate(spec=config.arbiter.policy, defaults=defaults),
        config=config,
        guard=CallGuard(strict=config.strict, time_limit=config.act_seconds_limit),
    )


def run_episode(
    *, env: SocialMarketEnvironment, seed: int, training: bool, writer: EpisodeWriter
) -> list[dict[str, Any]]:
    """Play one episode, streaming every step record to the writer."""
    env.reset(seed=seed, training=training)
    records: list[dict[str, Any]] = []
    while not env.done:
        record = env.step().record
        writer.write_step(record)
        records.append(record)
    return records


def episode_info(*, env: SocialMarketEnvironment, episode_seed: int) -> dict[str, Any]:
    """Return the costs, the reset seed and the component implementations of an episode."""
    config = env.config
    agents_config = config.agents
    agent_impls = {
        agent_id: agents_config.overrides.get(agent_id, agents_config.default).impl for agent_id in sorted(env.costs)
    }
    return {
        'costs': dict(sorted(env.costs.items())),
        'episode_seed': episode_seed,
        'components': {
            'agents': agent_impls,
            'coalition_policy': config.coalitions.policy.impl,
            'arbiter': config.arbiter.policy.impl,
        },
    }


def run_experiment(*, config: ExperimentConfig, runs_root: Path) -> Path:
    """Run every seed and episode of the config and return the run directory."""
    recorder = RunRecorder(runs_root=runs_root, config=config)
    for seed in config.seeds:
        env = build_environment(config=config, seed=seed)
        for episode in range(config.episodes_per_seed):
            _record_episode(env=env, recorder=recorder, seed=seed, episode=episode, training=config.mode == 'train')
    return recorder.run_dir


def _record_episode(
    *, env: SocialMarketEnvironment, recorder: RunRecorder, seed: int, episode: int, training: bool
) -> None:
    writer = recorder.open_episode(seed=seed, episode=episode)
    reset_seed = episode_seed(seed=seed, episode=episode)
    env.guard.slow_calls.clear()
    try:
        writer.write_episode_info(episode_info(env=env, episode_seed=reset_seed))
        records = run_episode(env=env, seed=reset_seed, training=training, writer=writer)
        writer.close(summary=summarize_episode(records), slow_calls=dict(env.guard.slow_calls))
    finally:
        writer.abort()
    logger.info('Finished seed %s episode %s after %s steps', seed, episode, len(records))
