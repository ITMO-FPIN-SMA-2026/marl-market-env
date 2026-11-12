"""Public component, legacy paths, and checkpoint continuity after packaging."""

from dataclasses import asdict
import importlib
import json

import pytest

torch = pytest.importorskip('torch')

from arbiter_miroshnik import PPOGRUArbiter, Settings
from arbiter_miroshnik.contracts import public_market
from arbiter_miroshnik.ppo import PPOGRUArbiterV2
from arbiter_miroshnik.reference import PublicReference
from social_market_env.config import MarketConfig, config_from_dict
from social_market_env.runner.experiment import build_environment


def test_legacy_imports_are_the_same_modules_and_component():
    from transferred_ppo_gru import TeamPPOGRU

    assert TeamPPOGRU is PPOGRUArbiter
    modules = {'api': 'policy', 'policy': 'ppo', 'model': 'model',
               'features': 'features', 'settings': 'settings',
               'contracts': 'contracts', 'reference': 'reference',
               'cli': 'research.cli', 'convert': 'research.convert',
               'calibrate': 'research.calibrate', 'sources': 'research.sources',
               'seller_state': 'research.seller_state'}
    for old, new in modules.items():
        assert importlib.import_module(f'transferred_ppo_gru.{old}') is importlib.import_module(f'arbiter_miroshnik.{new}')


def test_native_component_trains_and_resumes_complete_checkpoint(tmp_path):
    torch.set_num_threads(1)
    settings = Settings(screening_mode='learned', min_history=3,
                        history_window=5, batch_episodes=2, epochs=1)
    market = public_market(MarketConfig())
    reference = PublicReference(asdict(market), 25., 35., .8, 3, 5,
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
