"""Native PPO + GRU component and checkpoint continuity."""

from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path

import numpy as np

import pytest

torch = pytest.importorskip('torch')

from arbiter_miroshnik import PPOGRUArbiter, Settings
from arbiter_miroshnik.checkpoint import market_payload
from arbiter_miroshnik.features import PublicMemory
from arbiter_miroshnik.ppo import PPOGRUArbiterV2
from arbiter_miroshnik.reference import PublicReference
from arbiter_miroshnik.research.cli import play
from social_market_env.config import MarketConfig, config_from_dict, load_config
from social_market_env.contracts import ArbiterAction, ArbiterObservation
from social_market_env.recording.summary import summarize_episode
from social_market_env.runner.experiment import build_environment


def test_native_component_trains_and_resumes_complete_checkpoint(tmp_path):
    torch.set_num_threads(1)
    settings = Settings(screening_mode='learned', min_history=3,
                        history_window=5, batch_episodes=2, epochs=1)
    market = MarketConfig()
    reference = PublicReference(market_payload(market), 25., 35., .8, 3, 5,
                                ('reference-world',), ('calibration-world',))
    reference_path = tmp_path / 'reference.json'
    reference.save(reference_path)
    initial = PPOGRUArbiterV2(market=asdict(market), settings=asdict(settings),
                             reference_path=reference_path, model_seed=17)
    checkpoint = initial.save_checkpoint(tmp_path / 'initial.pt')
    state = tmp_path / 'state'
    config = config_from_dict({
        'name': 'arbiter_miroshnik_test', 'horizon': 6, 'strict': True,
        'agents': {'count': 4, 'default': {
            'impl': 'social_market_env.baselines.agents:BestResponseAgent'}},
        'coalitions': {'policy': {
            'impl': 'social_market_env.baselines.coalitions:NoCoalitions'}},
        'arbiter': {'policy': {'impl': 'arbiter_miroshnik:PPOGRUArbiter',
            'kwargs': {'checkpoint_path': str(checkpoint), 'device': 'cpu',
                       'state_dir': str(state)}}},
    })
    env = build_environment(config=config, seed=4100)
    for seed in (4200, 4201):
        env.reset(seed=seed, training=True)
        while not env.done:
            result = env.step().record
            assert not result['events'] and not result['rejected_plans']
    assert env.arbiter.policy.episodes_seen == 2
    assert env.arbiter.policy.updates == 1
    assert len(json.loads((state / 'index.json').read_text())) == 3
    before = env.arbiter.policy
    resumed = PPOGRUArbiter(market=asdict(MarketConfig()),
                            checkpoint_path=checkpoint, state_dir=state, device='cpu')
    assert resumed.policy.episodes_seen == 2 and resumed.policy.updates == 1
    for network in ('actor', 'critic'):
        for key, expected in getattr(before, network).state_dict().items():
            assert torch.equal(getattr(resumed.policy, network).state_dict()[key], expected)
        saved = getattr(before, f'{network}_optimizer').state_dict()
        loaded = getattr(resumed.policy, f'{network}_optimizer').state_dict()
        assert saved['param_groups'] == loaded['param_groups']
        for parameter, values in saved['state'].items():
            for key, expected in values.items():
                assert torch.equal(loaded['state'][parameter][key], expected)


def test_bundled_checkpoint_uses_native_runner_and_summary(tmp_path, monkeypatch):
    package = Path(__file__).resolve().parents[2]
    config = load_config(package / 'arbiter_miroshnik.yaml')
    config.arbiter.policy.kwargs['checkpoint_path'] = str(package / 'checkpoints/initial_model_0.pt')
    env = build_environment(config=config, seed=17)
    original = env.arbiter.act
    observations = []

    def act(observation):
        assert isinstance(observation, ArbiterObservation)
        observations.append(observation)
        action = original(observation)
        assert isinstance(action, ArbiterAction)
        return action

    monkeypatch.setattr(env.arbiter, 'act', act)
    log = tmp_path / 'episode.jsonl.gz'
    metrics = play(env, 71, training=False, log=log)
    with gzip.open(log, 'rt') as stream:
        records = [json.loads(line) for line in stream]
    summary = summarize_episode(records)
    assert metrics['steps'] == len(observations) == config.horizon
    assert metrics['welfare'] == summary['mean_social_welfare']
    assert metrics['reward'] == summary['total_arbiter_reward']
    assert metrics['false_positives'] == summary['false_positives']
    assert type(env.arbiter.policy.market) is MarketConfig

    policy = env.arbiter.policy
    memories = [PublicMemory(policy.market, policy.settings, policy.reference) for _ in range(2)]
    for observation in observations:
        # Extra native fields must not become neural inputs after removing the adapter.
        altered = replace(observation, target_social_welfare=float('nan'),
                          last_action=ArbiterAction(), fines_history=())
        frames = [memory.update(obs) for memory, obs in zip(memories, (observation, altered))]
        for field in ('global_features', 'nodes', 'pairs', 'eligible', 'risks'):
            np.testing.assert_array_equal(getattr(frames[0], field), getattr(frames[1], field))
