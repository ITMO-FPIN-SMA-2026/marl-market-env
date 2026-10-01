import json
from pathlib import Path

import pytest
import yaml

from social_market_env.config import ConfigError, config_from_dict
from social_market_env.contracts import AgentAction, AgentObservation, MarketAgentAPI
from social_market_env.run import main
from social_market_env.runner.experiment import build_environment, episode_seed, run_experiment

SMOKE = {
    'name': 'smoke',
    'horizon': 6,
    'seeds': [0, 1],
    'agents': {
        'count': 5,
        'default': {'impl': 'social_market_env.baselines.agents:CartelAgent'},
        'overrides': {4: {'impl': 'social_market_env.baselines.agents:RandomAgent'}},
    },
    'coalitions': {'policy': {'impl': 'social_market_env.baselines.coalitions:ConnectedComponentsCoalitions'}},
    'arbiter': {'policy': {'impl': 'social_market_env.baselines.arbiters:ThresholdArbiter'}},
    'network': {'topology': 'random', 'initial_density': 0.5},
}


def test_run_experiment_writes_meta_steps_and_summary(tmp_path: Path) -> None:
    run_dir = run_experiment(config=config_from_dict(SMOKE), runs_root=tmp_path)
    episode_dir = run_dir / 'seed_1' / 'episode_0'

    meta = json.loads((run_dir / 'meta.json').read_text(encoding='utf-8'))
    steps = (episode_dir / 'steps.jsonl').read_text(encoding='utf-8').splitlines()
    summary = json.loads((episode_dir / 'summary.json').read_text(encoding='utf-8'))

    assert meta['config']['name'] == 'smoke'
    assert len(steps) == 6
    assert json.loads(steps[0])['step'] == 0
    assert summary['steps'] == 6


def test_same_config_and_seed_give_identical_logs(tmp_path: Path) -> None:
    first = run_experiment(config=config_from_dict(SMOKE), runs_root=tmp_path / 'a')
    second = run_experiment(config=config_from_dict(SMOKE), runs_root=tmp_path / 'b')

    for seed in (0, 1):
        path = Path(f'seed_{seed}') / 'episode_0' / 'steps.jsonl'
        assert (first / path).read_bytes() == (second / path).read_bytes()


def test_cli_runs_a_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / 'smoke.yaml'
    config_path.write_text(yaml.safe_dump(SMOKE), encoding='utf-8')

    exit_code = main([str(config_path), '--runs-dir', str(tmp_path / 'runs')])

    assert exit_code == 0
    assert len(list((tmp_path / 'runs').iterdir())) == 1


class FailingAgent(MarketAgentAPI):
    """Test agent that raises on every act call."""

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        """Nothing to reset."""

    def act(self, observation: AgentObservation) -> AgentAction:
        """Always fail."""
        raise RuntimeError('boom')


def test_strict_failure_leaves_closed_steps_file_and_no_summary(tmp_path: Path) -> None:
    agents = {'count': 3, 'default': {'impl': f'{__name__}:FailingAgent'}}
    raw = {**SMOKE, 'strict': True, 'seeds': [0], 'agents': agents}

    with pytest.raises(RuntimeError, match='boom'):
        run_experiment(config=config_from_dict(raw), runs_root=tmp_path)

    episode_dir = next(tmp_path.iterdir()) / 'seed_0' / 'episode_0'
    assert (episode_dir / 'steps.jsonl').exists()
    assert not (episode_dir / 'summary.json').exists()


def test_timings_cover_only_their_own_episode(tmp_path: Path) -> None:
    raw = {**SMOKE, 'seeds': [0], 'episodes_per_seed': 2, 'act_seconds_limit': 0.0}

    run_dir = run_experiment(config=config_from_dict(raw), runs_root=tmp_path)

    first, second = (
        json.loads((run_dir / 'seed_0' / f'episode_{episode}' / 'timings.json').read_text(encoding='utf-8'))
        for episode in (0, 1)
    )
    assert first['slow_calls']
    assert first == second


def test_baselines_get_the_experiment_market() -> None:
    raw = {**SMOKE, 'market': {'tau': 10.0}}

    env = build_environment(config=config_from_dict(raw), seed=0)

    assert env.agents[0].market.tau == 10.0
    assert env.coalition_policy.market.tau == 10.0


def test_explicit_market_kwargs_win_over_the_experiment_market() -> None:
    default = {'impl': 'social_market_env.baselines.agents:BestResponseAgent', 'kwargs': {'market': {'tau': 3.0}}}
    raw = {**SMOKE, 'market': {'tau': 10.0}, 'agents': {'count': 3, 'default': default}}

    env = build_environment(config=config_from_dict(raw), seed=0)

    assert env.agents[0].market.tau == 3.0


def test_episode_info_and_reference_prices_are_logged(tmp_path: Path) -> None:
    run_dir = run_experiment(config=config_from_dict(SMOKE), runs_root=tmp_path)
    episode_dir = run_dir / 'seed_1' / 'episode_0'

    info = json.loads((episode_dir / 'episode.json').read_text(encoding='utf-8'))
    first_step = json.loads((episode_dir / 'steps.jsonl').read_text(encoding='utf-8').splitlines()[0])

    assert sorted(info['costs']) == ['0', '1', '2', '3', '4']
    assert info['episode_seed'] == episode_seed(seed=1, episode=0)
    assert info['components']['agents']['4'] == 'social_market_env.baselines.agents:RandomAgent'
    assert info['components']['arbiter'] == 'social_market_env.baselines.arbiters:ThresholdArbiter'
    assert first_step['cartel_average_price'] > first_step['nash_average_price'] > 0


@pytest.mark.parametrize(
    ('impl', 'message'),
    [
        ('no_such_package.agents:Agent', 'Cannot import module .no_such_package.agents.'),
        ('social_market_env.baselines.agents:NoSuchAgent', 'has no attribute .NoSuchAgent.'),
    ],
)
def test_bad_impl_raises_config_error(impl: str, message: str) -> None:
    raw = {**SMOKE, 'agents': {'count': 3, 'default': {'impl': impl}}}

    with pytest.raises(ConfigError, match=message):
        build_environment(config=config_from_dict(raw), seed=0)
