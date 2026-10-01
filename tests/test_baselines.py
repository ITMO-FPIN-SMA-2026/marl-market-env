import pytest

from social_market_env.contracts import AgentObservation, CoalitionObservation, CoalitionOffer, MarketInfo
from social_market_env.baselines.agents import BestResponseAgent, CartelAgent
from social_market_env.baselines.coalitions import ConnectedComponentsCoalitions


def agent_observation(*, offer: CoalitionOffer | None, last_prices: dict[int, float]) -> AgentObservation:
    return AgentObservation(
        agent_id=0,
        step=1,
        cost=18.0,
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


def test_cartel_agent_keeps_recommended_price() -> None:
    offer = CoalitionOffer(plan_id=2, members=(0, 1), recommended_price=69.0, payout_weight=0.5)

    action = CartelAgent().act(agent_observation(offer=offer, last_prices={}))

    assert (action.price, action.coalition_choice) == (69.0, 2)


def test_best_response_agent_undercuts_a_high_priced_market() -> None:
    last_prices = {0: 60.0, 1: 60.0, 2: 60.0}

    action = BestResponseAgent(damping=0.0).act(agent_observation(offer=None, last_prices=last_prices))

    assert 18.0 < action.price < 60.0


def test_connected_components_propose_monopoly_price_on_connected_graph() -> None:
    observation = CoalitionObservation(
        step=0,
        active_agents=(0, 1, 2),
        restricted=(),
        costs={0: 18.0, 1: 18.0, 2: 18.0},
        balances={0: 1000.0, 1: 1000.0, 2: 1000.0},
        edges={(0, 1): 0.0, (1, 2): 0.0},
        link_costs={},
        last_prices={},
        last_profits={},
        compliance_history={},
    )

    plan = ConnectedComponentsCoalitions().act(observation).plans[0]

    assert plan.members == (0, 1, 2)
    assert plan.recommended_prices[0] == pytest.approx(69.0, abs=0.05)
    assert sum(plan.payout_weights.values()) == pytest.approx(1.0)


def test_connected_components_prices_follow_cost_changes_on_a_reused_instance() -> None:
    policy = ConnectedComponentsCoalitions()

    def plan_price(cost: float) -> float:
        observation = CoalitionObservation(
            step=0,
            active_agents=(0, 1, 2),
            restricted=(),
            costs={0: cost, 1: cost, 2: cost},
            balances={0: 1000.0, 1: 1000.0, 2: 1000.0},
            edges={(0, 1): 0.0, (1, 2): 0.0},
            link_costs={},
            last_prices={},
            last_profits={},
            compliance_history={},
        )
        return policy.act(observation).plans[0].recommended_prices[0]

    cheap, expensive = plan_price(18.0), plan_price(40.0)

    assert cheap == pytest.approx(69.0, abs=0.05)
    assert expensive == pytest.approx(80.0, abs=0.05)
