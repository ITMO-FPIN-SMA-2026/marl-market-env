"""Behavioral and integration checks for RodinaAgent."""

from dataclasses import asdict

import pytest

from rodina_agent import RodinaAgent
from social_market_env.config import config_from_dict
from social_market_env.contracts import AgentObservation, CoalitionOffer, MarketInfo
from social_market_env.runner.experiment import build_environment


def _obs(*, offer: CoalitionOffer | None, last_prices: dict[int, float], cost: float = 18.0) -> AgentObservation:
    return AgentObservation(
        agent_id=0,
        step=1,
        cost=cost,
        balance=1000.0,
        active=True,
        restricted=False,
        last_price=last_prices.get(0),
        last_quantity=0.0,
        last_profit=0.0,
        market=MarketInfo(last_prices=last_prices, average_price=None, total_quantity=None),
        neighbors={},
        link_costs={},
        offer=offer,
    )


def test_reset_and_opening_price() -> None:
    agent = RodinaAgent(damping=0.0)
    agent.reset(agent_id=0, seed=1, training=False)
    action = agent.act(_obs(offer=None, last_prices={}))
    assert action.participate is True
    assert action.price == 23.0  # cost + opening_markup


def test_joins_near_best_response_offer() -> None:
    agent = RodinaAgent(damping=0.0, max_cartel_premium=0.2)
    agent.reset(agent_id=0, seed=0, training=False)
    solo_obs = _obs(offer=None, last_prices={0: 35.0, 1: 40.0, 2: 40.0})
    solo_price = RodinaAgent(damping=0.0)._solo_price(solo_obs)
    offer = CoalitionOffer(
        plan_id=2, members=(0, 1), recommended_price=solo_price * 1.05, payout_weight=0.5
    )
    action = agent.act(_obs(offer=offer, last_prices={0: 35.0, 1: 40.0, 2: 40.0}))
    assert action.coalition_choice == 2
    assert action.price == pytest.approx(solo_price * 1.05)


def test_defects_from_overpriced_cartel() -> None:
    agent = RodinaAgent(damping=0.0, max_cartel_premium=0.12)
    agent.reset(agent_id=0, seed=0, training=False)
    offer = CoalitionOffer(plan_id=2, members=(0, 1), recommended_price=90.0, payout_weight=0.5)
    action = agent.act(_obs(offer=offer, last_prices={0: 40.0, 1: 40.0, 2: 40.0}))
    assert action.coalition_choice is None
    assert action.price < 90.0


def test_rejects_tiny_payout_weight() -> None:
    agent = RodinaAgent(damping=0.0)
    agent.reset(agent_id=0, seed=0, training=False)
    offer = CoalitionOffer(plan_id=1, members=(0, 1), recommended_price=40.0, payout_weight=0.01)
    action = agent.act(_obs(offer=offer, last_prices={0: 40.0, 1: 40.0, 2: 40.0}))
    assert action.coalition_choice is None


def test_restricted_agent_refuses_offer() -> None:
    agent = RodinaAgent(damping=0.0)
    agent.reset(agent_id=0, seed=0, training=False)
    offer = CoalitionOffer(plan_id=2, members=(0, 1), recommended_price=40.0, payout_weight=0.5)
    obs = _obs(offer=offer, last_prices={0: 40.0, 1: 40.0, 2: 40.0})
    obs = AgentObservation(
        agent_id=obs.agent_id,
        step=obs.step,
        cost=obs.cost,
        balance=obs.balance,
        active=True,
        restricted=True,
        last_price=obs.last_price,
        last_quantity=obs.last_quantity,
        last_profit=obs.last_profit,
        market=obs.market,
        neighbors=obs.neighbors,
        link_costs=obs.link_costs,
        offer=offer,
    )
    action = agent.act(obs)
    assert action.coalition_choice is None


def test_observe_transition_updates_when_training() -> None:
    agent = RodinaAgent(damping=0.5, learn_rate=0.2)
    agent.reset(agent_id=0, seed=0, training=True)
    before = agent.damping
    obs = _obs(offer=None, last_prices={0: 40.0, 1: 40.0})
    action = agent.act(obs)
    agent.observe_transition(observation=obs, action=action, reward=10.0, next_observation=obs, done=False)
    assert agent.damping != before or agent._ema_reward != 0.0


def test_observe_transition_noop_when_not_training() -> None:
    agent = RodinaAgent(damping=0.5, learn_rate=0.2)
    agent.reset(agent_id=0, seed=0, training=False)
    before_damping = agent.damping
    before_ema = agent._ema_reward
    obs = _obs(offer=None, last_prices={0: 40.0, 1: 40.0})
    action = agent.act(obs)
    agent.observe_transition(observation=obs, action=action, reward=10.0, next_observation=obs, done=False)
    assert agent.damping == before_damping
    assert agent._ema_reward == before_ema


@pytest.mark.parametrize('topology', ['complete', 'star', 'path'])
@pytest.mark.parametrize('mode,share_model', [('static', 'logit'), ('evolving', 'tz_linear')])
def test_runs_through_environment_without_fallbacks(topology, mode, share_model) -> None:
    config = config_from_dict({
        'horizon': 6, 'strict': True,
        'agents': {
            'count': 4,
            'default': {'impl': 'social_market_env.baselines.agents:CartelAgent'},
            'overrides': {0: {'impl': 'rodina_agent:RodinaAgent'}},
        },
        'market': {'share_model': share_model},
        'network': {'topology': topology, 'mode': mode},
        'coalitions': {'policy': {'impl': 'social_market_env.baselines.coalitions:ConnectedComponentsCoalitions'}},
        'arbiter': {'policy': {'impl': 'social_market_env.baselines.arbiters:ThresholdArbiter'}},
    })
    env = build_environment(config=config, seed=0)
    learner = env.agents[0]
    assert isinstance(learner, RodinaAgent)
    assert asdict(learner.market) == asdict(config.market)
    env.reset(seed=10, training=True)
    while not env.done:
        record = env.step().record
        assert not record['events']
    env.reset(seed=11, training=False)
    while not env.done:
        env.step()
