"""Behavioral and integration checks for the independent seller implementation."""

import copy
import json
from dataclasses import asdict, replace

import pytest

from askarov_agent import AskarovAgent
from askarov_agent.experiment import main, scenario_config
from social_market_env.config import config_from_dict, config_to_dict
from social_market_env.contracts import (
    AgentEvaluation, AgentObservation, CoalitionOffer, LinkActionKind, MarketInfo,
)
from social_market_env.core.sanitize import sanitize_agent_action
from social_market_env.runner.experiment import build_environment


def observation(**changes) -> AgentObservation:
    base = AgentObservation(
        agent_id=0, step=1, cost=18.0, balance=1000.0, active=True,
        restricted=False, last_price=30.0, last_quantity=10.0, last_profit=120.0,
        market=MarketInfo(last_prices={0: 30.0, 1: 30.0, 2: 30.0}, average_price=30.0, total_quantity=90.0),
        neighbors={}, link_costs={1: 0.0, 2: 0.0}, offer=None,
    )
    return replace(base, **changes)


def fine_evaluation() -> AgentEvaluation:
    return AgentEvaluation(
        step=0, price=30.0, quantity=10.0, gross_profit=120.0, coalition_payout=120.0,
        fine=100.0, link_fee=0.0, reward=20.0, profit_error=None,
        reward_error=None, joined_plan_id=None,
    )


def test_economic_initialization_best_responds_to_expensive_rivals() -> None:
    obs = observation(market=MarketInfo({1: 60.0, 2: 60.0}, 60.0, 60.0))
    action = AskarovAgent().act(obs)
    assert action.participate
    assert 18.0 < action.price < 60.0
    assert action.coalition_choice is None


@pytest.mark.parametrize('share_model', ['logit', 'tz_linear'])
def test_high_share_offer_is_accepted_but_zero_share_is_refused(share_model) -> None:
    agent = AskarovAgent(market={'share_model': share_model})
    price = 30.0 if share_model == 'logit' else 80.0
    generous = CoalitionOffer(7, (0, 1, 2), recommended_price=price, payout_weight=0.9)
    accepted = agent.act(observation(offer=generous))
    refused = agent.act(observation(offer=replace(generous, payout_weight=0.0)))
    assert accepted.coalition_choice == 7
    assert accepted.price == price
    assert refused.coalition_choice is None


@pytest.mark.parametrize('restricted', [False, True])
def test_exploration_always_respects_price_bounds_and_restrictions(restricted) -> None:
    agent = AskarovAgent(market={'p_min': 10.0, 'p_max': 40.0}, epsilon=1.0, epsilon_min=1.0)
    agent.reset(agent_id=0, seed=9, training=True)
    obs = observation(
        cost=38.0, restricted=restricted, last_price=99.0,
        offer=CoalitionOffer(0, (0, 1), 12.0, 0.5),
    )
    for _ in range(100):
        action = agent.act(obs)
        assert 38.0 <= action.price <= 40.0
        assert action.coalition_choice is None  # Cannot comply below own cost.
        assert sanitize_agent_action(action=action, cost=obs.cost, restricted=restricted, market=agent.market) == action


def test_restricted_agent_refuses_even_a_profitable_offer() -> None:
    action = AskarovAgent().act(observation(
        restricted=True, balance=-5.0,
        offer=CoalitionOffer(2, (0, 1, 2), 30.0, 1.0),
    ))
    assert action.coalition_choice is None
    assert action.link_action is None
    assert action.price >= 18.0


@pytest.mark.parametrize('changes', [{'active': False}, {'cost': 120.0}])
def test_inactive_or_unprofitable_seller_sits_out(changes) -> None:
    action = AskarovAgent().act(observation(**changes))
    assert not action.participate
    assert action.coalition_choice is None
    assert action.link_action is None


def test_monopoly_is_priced_for_demand_instead_of_cost_plus_five() -> None:
    action = AskarovAgent().act(observation(link_costs={}, market=MarketInfo({}, None, None)))
    assert action.price == pytest.approx(69.0, abs=0.02)


def test_q_update_uses_net_reward_and_does_not_bootstrap_terminal_step() -> None:
    agent = AskarovAgent(learning_rate=1.0, gamma=0.5, reward_scale=10.0, epsilon=0.0, epsilon_min=0.0)
    agent.reset(agent_id=0, seed=0, training=True)
    obs = observation()
    action = agent.act(obs)
    agent.observe_transition(observation=obs, action=action, reward=-25.0, next_observation=obs, done=True)
    row = agent.q_table[agent._state_key(obs)]
    assert row['best_response'] == pytest.approx(-2.5)
    assert agent.updates == 1


def test_q_bootstrap_masks_unavailable_coalition_action() -> None:
    agent = AskarovAgent(learning_rate=1.0, gamma=0.5, reward_scale=10.0, epsilon=0.0, epsilon_min=0.0)
    agent.reset(agent_id=0, seed=0, training=True)
    obs = observation()
    action = agent.act(obs)
    agent.q_table[agent._state_key(obs)] = {
        'best_response': 3.0, 'cautious': 2.0, 'bold': 1.0,
        'hold': 2.0, 'wait': 0.0, 'coalition': 1000.0,
    }
    agent.observe_transition(observation=obs, action=action, reward=10.0, next_observation=obs, done=False)
    assert agent.q_table[agent._state_key(obs)]['best_response'] == pytest.approx(2.5)


def test_agent_learns_to_refuse_a_repeatedly_loss_making_offer() -> None:
    agent = AskarovAgent(learning_rate=1.0, epsilon=0.0, epsilon_min=0.0)
    agent.reset(agent_id=0, seed=0, training=True)
    obs = observation(offer=CoalitionOffer(2, (0, 1, 2), 30.0, 0.9))
    action = agent.act(obs)
    assert action.coalition_choice == 2
    agent.observe_transition(observation=obs, action=action, reward=-100.0, next_observation=None, done=True)
    agent.reset(agent_id=0, seed=1, training=False)
    assert agent.act(obs).coalition_choice is None


def test_evaluation_is_frozen_even_for_unseen_states_and_high_epsilon() -> None:
    agent = AskarovAgent(epsilon=1.0, epsilon_min=1.0)
    agent.reset(agent_id=0, seed=0, training=True)
    obs = observation()
    action = agent.act(obs)
    agent.observe_transition(observation=obs, action=action, reward=25.0, next_observation=None, done=True)
    before = copy.deepcopy(agent.q_table), agent.updates
    agent.reset(agent_id=0, seed=99, training=False)
    assert agent.exploration_rate == 0.0
    for step in range(10):
        unseen = observation(step=step, balance=10.0)
        chosen = agent.act(unseen)
        assert chosen == agent.act(unseen)
        agent.observe_transition(observation=unseen, action=chosen, reward=-10.0, next_observation=obs, done=False)
    assert (agent.q_table, agent.updates) == before


def test_reset_keeps_learning_and_seed_reproduces_exploration() -> None:
    agent = AskarovAgent(epsilon=1.0, epsilon_min=1.0)
    obs = observation(offer=CoalitionOffer(0, (0, 1), 35.0, 0.5))
    sequences = []
    for _ in range(2):
        agent.reset(agent_id=0, seed=123, training=True)
        sequences.append([agent.act(obs) for _ in range(30)])
    assert sequences[0] == sequences[1]
    action = agent.act(obs)
    agent.observe_transition(observation=obs, action=action, reward=5.0, next_observation=None, done=True)
    before = copy.deepcopy(agent.q_table), agent.updates
    agent.reset(agent_id=0, seed=124, training=True)
    assert (agent.q_table, agent.updates) == before


def test_recent_fine_changes_local_context_and_the_untrained_decision() -> None:
    offer = CoalitionOffer(2, (0, 1, 2), 30.0, 0.4)
    fined = observation(offer=offer, last_evaluation=fine_evaluation())
    assert AskarovAgent(use_memory=False).act(fined).coalition_choice == 2
    assert AskarovAgent(use_memory=True).act(fined).coalition_choice is None


def test_links_keep_coalition_partner_and_drop_hostile_partner() -> None:
    agent = AskarovAgent()
    obs = observation(neighbors={1: 0.8, 2: 0.2}, offer=CoalitionOffer(2, (0, 1, 2), 30.0, 0.9))
    link = agent.act(obs).link_action
    assert link is not None
    assert (link.kind, link.target) == (LinkActionKind.maintain, 1)
    link = agent.act(replace(obs, neighbors={1: -0.5, 2: 0.2})).link_action
    assert link is not None
    assert (link.kind, link.target) == (LinkActionKind.remove, 1)


def test_new_link_respects_current_fee_and_reserve_budget() -> None:
    agent = AskarovAgent(max_link_fee=5.0)
    obs = observation(balance=100.0, link_costs={1: 4.0, 2: 1.0})
    link = agent.act(obs).link_action
    assert link is not None
    assert (link.kind, link.target) == (LinkActionKind.add, 2)
    assert agent.act(replace(obs, link_costs={1: 4.0, 2: 2.0})).link_action is None


def test_checkpoint_round_trip_and_market_mismatch(tmp_path) -> None:
    agent = AskarovAgent()
    agent.reset(agent_id=0, seed=4, training=True)
    obs = observation()
    action = agent.act(obs)
    agent.observe_transition(observation=obs, action=action, reward=15.0, next_observation=None, done=True)
    path = tmp_path / 'policy.json'
    agent.save(path)
    restored = AskarovAgent.from_checkpoint(path)
    assert restored.q_table == agent.q_table
    assert restored.updates == agent.updates
    agent.reset(agent_id=0, seed=10, training=False)
    restored.reset(agent_id=0, seed=11, training=False)
    assert restored.act(obs) == agent.act(obs)
    with pytest.raises(ValueError, match='market/settings'):
        AskarovAgent(market={'share_model': 'tz_linear'}, checkpoint_path=str(path))
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['q_table']['corrupt'] = {'best_response': float('nan')}
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='Q values'):
        AskarovAgent.from_checkpoint(path)


@pytest.mark.parametrize('topology', ['complete', 'star', 'path', 'inner_coalitions'])
@pytest.mark.parametrize('mode,share_model', [('static', 'logit'), ('evolving', 'tz_linear')])
def test_runs_through_environment_without_fallbacks(topology, mode, share_model) -> None:
    config = config_from_dict({
        'horizon': 6, 'strict': True,
        'agents': {'count': 4, 'default': {'impl': 'social_market_env.baselines.agents:CartelAgent'},
                   'overrides': {0: {'impl': 'askarov_agent:AskarovAgent'}}},
        'market': {'share_model': share_model},
        'network': {'topology': topology, 'mode': mode},
        'coalitions': {'policy': {'impl': 'social_market_env.baselines.coalitions:ConnectedComponentsCoalitions'}},
        'arbiter': {'policy': {'impl': 'social_market_env.baselines.arbiters:ThresholdArbiter'}},
    })
    env = build_environment(config=config, seed=0)
    learner = env.agents[0]
    assert isinstance(learner, AskarovAgent)
    assert asdict(learner.market) == asdict(config.market)
    env.reset(seed=10, training=True)
    while not env.done:
        record = env.step().record
        assert not record['events']
    assert learner.updates == config.horizon
    before = copy.deepcopy(learner.q_table), learner.updates
    env.reset(seed=11, training=False)
    while not env.done:
        env.step()
    assert (learner.q_table, learner.updates) == before


def test_training_and_paired_evaluation_cli(tmp_path) -> None:
    import yaml

    config = config_from_dict({
        'horizon': 3, 'seeds': [0], 'agents': {
            'count': 3, 'overrides': {0: {'impl': 'askarov_agent:AskarovAgent'}},
        },
    })
    before = config_to_dict(config)
    scenario_config(config, scenario='cooperation', agent_id=0)
    assert config_to_dict(config) == before
    config_path = tmp_path / 'input.yaml'
    config_path.write_text(yaml.safe_dump(before), encoding='utf-8')
    checkpoint = tmp_path / 'policy.json'
    common = ['--config', str(config_path), '--topologies', 'path', '--network-modes', 'static',
              '--scenarios', 'competition', 'cooperation']
    assert main(['train', *common, '--episodes', '2', '--output', str(checkpoint)]) == 0
    saved_bytes = checkpoint.read_bytes()
    report = tmp_path / 'comparison.json'
    assert main(['benchmark', *common, '--checkpoint', str(checkpoint), '--seeds', '100',
                 '--output', str(report)]) == 0
    assert checkpoint.read_bytes() == saved_bytes
    metrics = json.loads(report.read_text(encoding='utf-8'))
    assert metrics['seed_overlap'] == []
    assert len(metrics['episodes']) == 8
    assert set(metrics['aggregate']) == {'best_response', 'cartel', 'hybrid', 'q_learning'}
    evaluation = config_from_dict(yaml.safe_load(checkpoint.with_suffix('.eval.yaml').read_text(encoding='utf-8')))
    assert evaluation.mode == 'eval'
    assert evaluation.seeds == [100, 101, 102]
    restored = build_environment(config=evaluation, seed=100).agents[0]
    assert isinstance(restored, AskarovAgent)
    assert restored.updates == 6
