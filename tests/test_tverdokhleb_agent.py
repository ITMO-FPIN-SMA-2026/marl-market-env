import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tverdokhleb_agent import TverdokhlebAgent, TverdokhlebConfig
from social_market_env.config import MarketConfig, config_from_dict, load_config
from social_market_env.contracts import AgentAction, AgentObservation, CoalitionOffer, MarketInfo
from social_market_env.core.sanitize import sanitize_agent_action
from social_market_env.runner.experiment import build_environment, run_experiment
from tverdokhleb_agent.experiment import (
    evaluate, evaluation_config, main, scenario_config, train_model, validate_eval_seeds,
)

ROOT = Path(__file__).resolve().parents[1]


def observation(**overrides):
    values = dict(agent_id=0, step=0, cost=20.0, balance=1000.0, last_profit=0.0,
                  last_price=None, last_quantity=0.0, restricted=False,
                  market=MarketInfo(last_prices={}, average_price=None, total_quantity=None),
                  neighbors={}, link_costs={}, offer=None, active=True)
    return AgentObservation(**dict(values, **overrides))


def learner(**options):
    agent = TverdokhlebAgent(config={'allow_loss_leading': True, **options})
    agent.reset(agent_id=0, seed=42, training=True)
    return agent


def transition(agent, obs, action, reward, following=None, done=True):
    agent.observe_transition(observation=obs, action=action, reward=reward, next_observation=following, done=done)


def test_td_target_and_terminal_transition():
    agent = learner(state_mode='basic', learning_rate=0.5, gamma=0.5, reward_scale=2)
    current = observation()
    following = observation(step=1, market=MarketInfo({}, 40, 80))
    state, next_state = agent.encode_state(current), agent.encode_state(following)
    agent.q[state] = [2.0] * agent.n_actions
    agent.q[next_state] = [6.0] * agent.n_actions
    action = AgentAction(agent.prices[0])
    transition(agent, current, action, 4, following, False)
    assert agent.q[state][0] == 3.5
    agent.q[state][0] = 2
    transition(agent, current, action, 4, following, True)
    assert agent.q[state][0] == 2


def test_bootstrap_masks_missing_offer_and_prices_below_restricted_cost():
    agent = learner(state_mode='basic', learning_rate=1, gamma=0.5, reward_scale=1)
    current = observation()
    following = observation(step=1, cost=80, restricted=True, balance=-1)
    values = [1000.0] * agent.n_actions
    for index in agent.available_actions(following):
        values[index] = 6
    agent.q[agent.encode_state(following)] = values
    transition(agent, current, AgentAction(agent.prices[0]), -1, following, False)
    assert agent.q[agent.encode_state(current)][0] == 2


@pytest.mark.parametrize('training', [True, False])
def test_restricted_actions_survive_sanitization_unchanged(training):
    agent = learner(epsilon_start=1, epsilon_end=1)
    agent.reset(agent_id=0, seed=6, training=training)
    obs = observation(cost=41.1, restricted=True, balance=-10,
                      offer=CoalitionOffer(7, (0, 1), 60.3, 0.5))
    assert agent.accept_index not in agent.available_actions(obs)
    for _ in range(100):
        action = agent.act(obs)
        assert action == sanitize_agent_action(action=action, cost=obs.cost, restricted=True, market=MarketConfig())
        assert not action.participate or action.price >= obs.cost
        assert action.coalition_choice is None


def test_follow_offer_uses_exact_non_grid_price_and_its_own_q_action():
    agent = learner(learning_rate=1, reward_scale=1)
    obs = observation(offer=CoalitionOffer(7, (0, 1), 41.123, 0.5))
    state = agent.encode_state(obs)
    agent.q[state] = [0.0] * agent.n_actions
    agent.q[state][agent.accept_index] = 100
    agent.reset(agent_id=0, seed=1, training=False)
    action = agent.act(obs)
    assert action == AgentAction(price=41.123, coalition_choice=7)
    agent.reset(agent_id=0, seed=1, training=True)
    transition(agent, obs, action, 42)
    assert agent.q[state][agent.accept_index] == 42
    assert all(value == 0 for index, value in enumerate(agent.q[state]) if index != agent.accept_index)


def test_absent_offer_cannot_be_chosen_even_with_high_q_value():
    agent = learner()
    obs = observation()
    agent.q[agent.encode_state(obs)] = [0] * agent.n_actions
    agent.q[agent.encode_state(obs)][agent.accept_index] = 1000
    agent.reset(agent_id=0, seed=1, training=False)
    assert agent.act(obs).coalition_choice is None


def test_abstaining_is_a_learned_action():
    agent = learner(learning_rate=1, reward_scale=1)
    obs = observation()
    state = agent.encode_state(obs)
    transition(agent, obs, agent.available_actions(obs)[agent.abstain_index], 8)
    assert agent.q[state][agent.abstain_index] == 8


def test_no_affordable_grid_action_uses_abstention():
    agent = learner()
    obs = observation(cost=101, restricted=True, balance=-1)
    assert list(agent.available_actions(obs)) == [agent.abstain_index]
    assert agent.act(obs).participate is False


def test_individual_elimination_is_terminal():
    agent = learner(state_mode='basic', learning_rate=1, reward_scale=1)
    current = observation()
    following = observation(step=1, active=False, market=MarketInfo({}, 50, 70))
    agent.q[agent.encode_state(following)] = [1000.0] * agent.n_actions
    transition(agent, current, AgentAction(agent.prices[0]), -10, following, False)
    assert agent.q[agent.encode_state(current)][0] == -10


def test_reset_keeps_learning_and_switches_mode_but_clears_memory():
    agent = learner()
    obs = observation(market=MarketInfo({}, 30, 90))
    transition(agent, obs, agent.act(obs), 5)
    learned, updates = copy.deepcopy(agent.q), agent.updates
    agent.reset(agent_id=7, seed=6, training=False)
    assert agent.q == learned and agent.updates == updates
    assert agent.epsilon == 0 and not agent.training
    assert not agent.recent_rewards and agent.previous_market_price is None
    agent.reset(agent_id=7, seed=7, training=True)
    assert agent.epsilon > 0 and agent.updates == updates


def test_evaluation_freezes_model_but_advances_memory():
    agent = TverdokhlebAgent()
    agent.reset(agent_id=0, seed=0, training=False)
    snapshot = copy.deepcopy(agent.metadata)
    obs = observation(market=MarketInfo({}, 25, 95))
    following = observation(step=1, market=MarketInfo({}, 80, 40))
    transition(agent, obs, agent.act(obs), -50, following, False)
    assert agent.q == {} and agent.updates == 0 and agent.metadata == snapshot
    assert agent.encode_state(following)[6:8] == (1, -1)


def test_next_state_memory_matches_the_next_act():
    agent = learner(learning_rate=1, gamma=0.5, reward_scale=100)
    current = observation(market=MarketInfo({}, 20, 100))
    following = observation(step=1, market=MarketInfo({}, 80, 40))
    expected = TverdokhlebAgent(config=agent.config)
    action = AgentAction(agent.prices[2])
    transition(expected, current, action, -50, following, False)
    next_state = expected.encode_state(following)
    agent.q[next_state] = [10.0] * agent.n_actions
    state = agent.encode_state(current)
    transition(agent, current, action, -50, following, False)
    assert agent.q[state][2] == 4.5
    assert agent.encode_state(following) == next_state


def test_learns_rewarding_action():
    agent = learner(state_mode='basic', price_levels=3, gamma=0, reward_scale=1, epsilon_decay=0.99)
    obs = observation()
    for _ in range(400):
        action = agent.act(obs)
        reward = 10 if action.price == agent.prices[1] and action.participate else -1
        transition(agent, obs, action, reward, obs, False)
    agent.reset(agent_id=0, seed=4, training=False)
    assert agent.act(obs) == AgentAction(agent.prices[1])


def test_inactive_agent_does_not_trade_or_learn():
    agent = learner()
    obs = observation(active=False)
    action = agent.act(obs)
    assert not action.participate
    transition(agent, obs, action, 0)
    assert agent.updates == 0


def test_save_load_and_yaml_constructor(tmp_path):
    agent = learner()
    obs = observation()
    transition(agent, obs, agent.act(obs), 200)
    path = tmp_path / 'agent.json'
    agent.save(path)
    loaded = TverdokhlebAgent(model_path=path, market=agent.market)
    assert loaded.q == agent.q and loaded.updates == agent.updates
    assert loaded.metadata == agent.metadata and not loaded.training
    agent.reset(agent_id=2, seed=50, training=False)
    loaded.reset(agent_id=2, seed=50, training=False)
    assert [agent.act(obs) for _ in range(10)] == [loaded.act(obs) for _ in range(10)]
    with pytest.raises(ValueError, match='market differs'):
        TverdokhlebAgent(model_path=path, market={'tau': 10})


def test_terminal_checkpoint_and_eval_does_not_write(tmp_path):
    template = str(tmp_path / 'agent_{agent_id}.json')
    agent = TverdokhlebAgent(checkpoint_path=template)
    agent.reset(agent_id=4, seed=9, training=True)
    obs = observation()
    transition(agent, obs, agent.act(obs), 20)
    saved = tmp_path / 'agent_4.json'
    assert TverdokhlebAgent.load(saved).updates == 1
    contents = saved.read_bytes()
    agent.reset(agent_id=4, seed=10, training=False)
    transition(agent, obs, agent.act(obs), 99)
    assert saved.read_bytes() == contents


def test_rejects_old_or_corrupt_models(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'format_version': 1}))
    with pytest.raises(ValueError, match='MVP'):
        TverdokhlebAgent.load(path)
    agent = learner()
    obs = observation()
    transition(agent, obs, agent.act(obs), 1)
    agent.save(path)
    payload = json.loads(path.read_text())
    payload['q_table'][0]['values'][0] = float('nan')
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='invalid Q'):
        TverdokhlebAgent.load(path)


def test_rejects_unrecognized_executed_action_instead_of_nearest_grid():
    agent = learner()
    with pytest.raises(ValueError, match='executed action'):
        transition(agent, observation(), AgentAction(price=42.42), 5)


@pytest.mark.parametrize('changes', [{'gamma': 2}, {'price_levels': 1}, {'price_levels': 3.5},
                                   {'reward_scale': 0}, {'epsilon_decay': 0}, {'max_price': float('nan')}])
def test_invalid_hyperparameters(changes):
    with pytest.raises(ValueError):
        replace(TverdokhlebConfig(), **changes)


def test_standalone_yaml_loads_public_agent_without_checkpoint():
    config = load_config(ROOT / 'experiments/tverdokhleb_agent.yaml')
    config.horizon = 4
    env = build_environment(config=config, seed=0)
    agent = env.agents[0]
    assert isinstance(agent, TverdokhlebAgent)
    assert agent.market == asdict(config.market)
    env.reset(seed=0, training=False)
    while not env.done:
        assert not env.step().record['events']
    assert agent.updates == 0 and agent.q == {}


@pytest.mark.parametrize('topology', ['complete', 'star', 'path', 'inner_coalitions'])
@pytest.mark.parametrize('mode', ['static', 'evolving'])
@pytest.mark.parametrize('share_model', ['logit', 'tz_linear'])
@pytest.mark.parametrize('state_mode', ['basic', 'context', 'history'])
def test_public_plugin_trains_and_evaluates_across_environment_modes(topology, mode, share_model, state_mode):
    config = config_from_dict({
        'horizon': 6, 'strict': True,
        'market': {'share_model': share_model, 'p_min': 10.0, 'p_max': 80.0},
        'finance': {'initial_balance': 100000.0},
        'agents': {'count': 4, 'default': {'impl': 'social_market_env.baselines.agents:CartelAgent'},
                   'overrides': {0: {'impl': 'tverdokhleb_agent:TverdokhlebAgent',
                                     'kwargs': {'config': {'state_mode': state_mode}}}}},
        'network': {'topology': topology, 'mode': mode},
        'coalitions': {'policy': {'impl': 'social_market_env.baselines.coalitions:ConnectedComponentsCoalitions'}},
        'arbiter': {'policy': {'impl': 'social_market_env.baselines.arbiters:ThresholdArbiter'}},
    })
    env = build_environment(config=config, seed=0)
    agent = env.agents[0]
    assert isinstance(agent, TverdokhlebAgent)
    assert agent.market == asdict(config.market)
    env.reset(seed=10, training=True)
    while not env.done:
        assert not env.step().record['events']
    assert agent.updates == config.horizon
    frozen = copy.deepcopy((agent.q, agent.metadata)), agent.updates
    env.reset(seed=11, training=False)
    while not env.done:
        assert not env.step().record['events']
    assert ((agent.q, agent.metadata), agent.updates) == frozen


def test_environment_delivers_next_observation_with_the_next_offer():
    class RecordingAgent(TverdokhlebAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.observations = []
            self.transitions = []

        def act(self, obs):
            self.observations.append(obs)
            return super().act(obs)

        def observe_transition(self, **transition):
            self.transitions.append(transition)
            super().observe_transition(**transition)

    config = load_config(ROOT / 'experiments/tverdokhleb_agent.yaml')
    config.horizon = 5
    config.finance.initial_balance = 100000.0
    env = build_environment(config=config, seed=0)
    agent = RecordingAgent(market=asdict(config.market))
    env.agents[0] = agent
    env.reset(seed=10, training=True)
    while not env.done:
        assert not env.step().record['events']
    assert len(agent.observations) == len(agent.transitions) == config.horizon
    for transition, following in zip(agent.transitions[:-1], agent.observations[1:]):
        assert transition['next_observation'] == following
    assert agent.transitions[-1]['done'] and agent.transitions[-1]['next_observation'] is None


@pytest.mark.parametrize('allow_loss_leading', [False, True])
@pytest.mark.parametrize('price_levels,cost,expected', [(3, 55.0, 100.0), (33, 120.0, None)])
def test_unseen_state_fallback_never_rounds_below_cost(allow_loss_leading, price_levels, cost, expected):
    agent = TverdokhlebAgent(config={'price_levels': price_levels, 'allow_loss_leading': allow_loss_leading})
    agent.reset(agent_id=0, seed=42, training=False)
    action = agent.act(observation(cost=cost))
    if expected is None:
        assert not action.participate
    else:
        assert action.participate and action.price == expected and action.price >= cost


@pytest.mark.parametrize('training', [True, False])
def test_default_policy_protects_margin_with_positive_reserve(training):
    agent = TverdokhlebAgent()
    agent.reset(agent_id=0, seed=4, training=training)
    obs = observation(cost=41.1, offer=CoalitionOffer(2, (0, 1), 40.0, 0.5))
    state = agent.encode_state(obs)
    agent.q[state] = [1000.0] * agent.n_actions
    assert agent.accept_index not in agent.available_actions(obs)
    for _ in range(100):
        action = agent.act(obs)
        assert not action.participate or action.price >= obs.cost
        assert action.coalition_choice is None


def test_loss_leading_is_opt_in_and_restrictions_still_apply():
    agent = TverdokhlebAgent(config={'allow_loss_leading': True})
    obs = observation()
    assert agent.available_actions(obs)[0].price < obs.cost
    assert 0 not in agent.available_actions(replace(obs, restricted=True, balance=-1))


def test_non_loss_mask_also_applies_to_the_bootstrap_target():
    agent = TverdokhlebAgent(config={'state_mode': 'basic', 'learning_rate': 1, 'gamma': 0.5, 'reward_scale': 1})
    agent.reset(agent_id=0, seed=0, training=True)
    current = observation(cost=20)
    following = observation(cost=55, offer=CoalitionOffer(2, (0, 1), 40.0, 0.5))
    values = [1000.0] * agent.n_actions
    for index in agent.available_actions(following):
        values[index] = 6.0
    agent.q[agent.encode_state(following)] = values
    action = AgentAction(price=agent.prices[10])
    transition(agent, current, action, 1.0, following, False)
    assert agent.q[agent.encode_state(current)][10] == 4.0


def test_previous_v2_checkpoint_requires_explicit_retraining(tmp_path):
    path = tmp_path / 'previous.json'
    path.write_text(json.dumps({'format_version': 2, 'contracts_version': '2.0.0'}))
    with pytest.raises(ValueError, match='require retraining'):
        TverdokhlebAgent.load(path)


def test_context_ablation_changes_only_the_two_history_features():
    obs = observation(cost=35, balance=300, offer=CoalitionOffer(2, (0, 1), 42.0, 0.7),
                      market=MarketInfo({}, 90.0, 30.0))
    history = TverdokhlebAgent(config={'state_mode': 'history'})
    context = TverdokhlebAgent(config={'state_mode': 'context'})
    for agent in (history, context):
        agent.previous_market_price = 20.0
        agent.recent_rewards.append(-200.0)
    full, current = history.encode_state(obs), context.encode_state(obs)
    assert full[6:8] == (1, -1) and current[6:8] == (0, 0)
    assert full[:6] + full[8:] == current[:6] + current[8:]
    assert history.available_actions(obs) == context.available_actions(obs)


def test_fresh_training_rotates_costs_and_retains_the_learner(tmp_path):
    base = load_config(ROOT / 'experiments/tverdokhleb_agent.yaml')
    base.horizon = 5
    original = copy.deepcopy(base)
    first, report = train_model(base, state_mode='history', seed=42, episodes=4,
                                min_updates=15, agent_id=0, output=tmp_path / 'first.json')
    second, repeated = train_model(base, state_mode='history', seed=42, episodes=4,
                                   min_updates=15, agent_id=0, output=tmp_path / 'second.json')
    assert base == original
    assert report == repeated and first.q == second.q
    assert first.updates == sum(row['transitions'] for row in report['episodes'])
    assert report['budget_met'] and first.metadata['training_root_seeds'] == [42, 43, 44, 45]
    assert len({row['cost'] for row in report['episodes']}) > 1
    assert len({tuple(row['condition']) for row in report['episodes']}) > 1
    assert TverdokhlebAgent.load(tmp_path / 'first.json').q == first.q


def test_training_cli_emits_native_eval_and_evaluation_is_frozen(tmp_path):
    model = tmp_path / 'policy.json'
    assert main(['train', '--episodes', '3', '--horizon', '5', '--min-updates', '10',
                 '--output', str(model), '--eval-seeds', '1000']) == 0
    evaluation = load_config(model.with_suffix('.eval.yaml'))
    assert evaluation.mode == 'eval' and evaluation.seeds == [1000] and evaluation.strict
    saved = model.read_bytes()
    result = run_experiment(config=evaluation, runs_root=tmp_path / 'native')
    assert model.read_bytes() == saved
    for path in result.glob('seed_*/episode_*/steps.jsonl'):
        assert all(not json.loads(line)['events'] for line in path.read_text().splitlines())
    loaded = TverdokhlebAgent.load(model)
    frozen = copy.deepcopy((loaded.q, loaded.metadata)), loaded.updates
    rows = evaluate(evaluation, models={'history': loaded}, seeds=[1001], agent_id=0,
                    conditions=[('mixed', 'path', 'static')])
    assert ((loaded.q, loaded.metadata), loaded.updates) == frozen
    assert {row['policy'] for row in rows} == {'history', 'untrained', 'best_response'}
    assert len({row['cost'] for row in rows}) == 1


def test_checkpoint_evaluation_rejects_training_seeds_and_market_changes(tmp_path):
    agent = TverdokhlebAgent()
    agent.metadata['training_root_seeds'] = [42, 43]
    path = tmp_path / 'agent.json'
    agent.save(path)
    base = load_config(ROOT / 'experiments/tverdokhleb_agent.yaml')
    with pytest.raises(ValueError, match='overlap'):
        evaluation_config(base, checkpoint=path, seeds=[43], agent_id=0)
    with pytest.raises(ValueError, match='distinct'):
        validate_eval_seeds([agent], [1000, 1000])
    base.market.tau = 10
    with pytest.raises(ValueError, match='market differs'):
        evaluate(base, models={'history': agent}, seeds=[1000], agent_id=0)


def test_benchmark_has_independent_training_repeats_and_paired_memory_comparison(tmp_path):
    output = tmp_path / 'benchmark'
    assert main(['benchmark', '--episodes', '2', '--horizon', '3', '--min-updates', '4',
                 '--train-seeds', '42', '52', '--eval-seeds', '1000', '--output', str(output)]) == 0
    report = json.loads((output / 'comparison.json').read_text())
    assert set(report['aggregate']) == {'context', 'history', 'untrained', 'best_response'}
    assert len(report['training']) == 4
    assert len(report['history_minus_context_by_training_repeat']) == 2
    assert {tuple(row['training_root_seeds']) for row in report['training']} == {(42, 43), (52, 53)}
    for name in ('comparison.md', 'repeat_0/history.json', 'repeat_1/context.json'):
        assert (output / name).is_file()


@pytest.mark.parametrize('options,match', [
    (['--train-seeds', '42', '43'], 'must not overlap'),
    (['--eval-seeds', '42'], 'seed overlap'),
    (['--variants', 'history', 'history'], 'Duplicate'),
])
def test_invalid_benchmark_plan_fails_before_writing(tmp_path, options, match):
    output = tmp_path / 'unused'
    with pytest.raises(ValueError, match=match):
        main(['benchmark', '--episodes', '3', '--horizon', '3', '--min-updates', '1',
              '--output', str(output), *options])
    assert not output.exists()


def test_unreachable_training_budget_is_rejected(tmp_path):
    base = load_config(ROOT / 'experiments/tverdokhleb_agent.yaml')
    base.horizon = 2
    with pytest.raises(ValueError, match='reachable'):
        train_model(base, state_mode='history', seed=42, episodes=2, min_updates=5,
                    agent_id=0, output=tmp_path / 'unused.json')
    assert not (tmp_path / 'unused.json').exists()


def test_actual_budget_failure_is_reported_after_early_elimination(tmp_path, monkeypatch):
    import tverdokhleb_agent.experiment as experiment

    monkeypatch.setattr(experiment, 'CONDITIONS', (('mixed', 'complete', 'static'),))
    base = load_config(ROOT / 'experiments/tverdokhleb_agent.yaml')
    base.horizon = 5
    base.finance.max_negative_steps = 0
    base.arbiter.max_fine = 100000
    base.arbiter.policy.impl = 'social_market_env.baselines.arbiters:PunishAll'
    base.arbiter.policy.kwargs = {'fine': 100000}
    output = tmp_path / 'failed.json'
    with pytest.raises(ValueError, match='required 8'):
        train_model(base, state_mode='history', seed=42, episodes=2, min_updates=8,
                    agent_id=0, output=output)
    report = json.loads(output.with_suffix('.training.json').read_text())
    assert not report['budget_met'] and report['updates'] == 2
    assert TverdokhlebAgent.load(output).updates == 2
