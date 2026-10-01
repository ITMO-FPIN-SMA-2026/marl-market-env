"""Observation builders: what each role is allowed to see."""

from social_market_env.contracts import (
    AgentObservation,
    ArbiterObservation,
    CoalitionObservation,
    CoalitionOffer,
    MarketInfo,
)
from social_market_env.core.state import EnvState, StepContext
from social_market_env.gametheory.graphs import as_pair


def build_agent_observation(
    *, state: EnvState, agent_id: int, cost: float, offer: CoalitionOffer | None
) -> AgentObservation:
    """Return the agent's private data, public market data and its own neighborhood."""
    account = state.accounts[agent_id]
    active = set(state.active_ids())
    return AgentObservation(
        agent_id=agent_id,
        step=state.step,
        cost=cost,
        balance=account.balance,
        active=account.active,
        restricted=account.restricted,
        last_price=state.last_prices.get(agent_id),
        last_quantity=state.last_quantities.get(agent_id, 0.0),
        last_profit=state.last_rewards.get(agent_id, 0.0),
        market=MarketInfo(
            last_prices=dict(state.last_prices),
            average_price=state.last_average_price,
            total_quantity=state.last_total_quantity,
        ),
        neighbors=state.network.neighbor_weights(agent_id=agent_id, active=active),
        link_costs={
            other: state.network.link_costs[as_pair((agent_id, other))] for other in sorted(active - {agent_id})
        },
        offer=offer,
    )


def build_coalition_observation(*, state: EnvState, costs: dict[int, float]) -> CoalitionObservation:
    """Return the network and market state the coalition policy works with."""
    active = state.active_ids()
    return CoalitionObservation(
        step=state.step,
        active_agents=tuple(active),
        restricted=tuple(agent_id for agent_id in active if state.accounts[agent_id].restricted),
        costs={agent_id: costs[agent_id] for agent_id in active},
        balances={agent_id: state.accounts[agent_id].balance for agent_id in active},
        edges=state.network.edge_weights(active=set(active)),
        link_costs=dict(state.network.link_costs),
        last_prices=dict(state.last_prices),
        last_profits={agent_id: state.last_rewards.get(agent_id, 0.0) for agent_id in active},
        compliance_history={agent_id: tuple(records) for agent_id, records in state.compliance_history.items()},
    )


def build_arbiter_observation(*, state: EnvState, context: StepContext) -> ArbiterObservation:
    """Return market data of this step only; nothing about the network or coalitions."""
    outcome = context.outcome
    previous_price = outcome.average_price if state.last_average_price is None else state.last_average_price
    previous_welfare = context.welfare if state.last_welfare is None else state.last_welfare
    return ArbiterObservation(
        step=state.step,
        active_agents=tuple(state.active_ids()),
        prices=dict(outcome.prices),
        quantities=dict(outcome.quantities),
        total_quantity=outcome.total_quantity,
        average_price=outcome.average_price,
        social_welfare=context.welfare,
        target_social_welfare=context.target_welfare,
        delta_average_price=outcome.average_price - previous_price,
        delta_social_welfare=context.welfare - previous_welfare,
        last_action=state.last_arbiter_action,
        fines_history=tuple(state.fines_history),
    )
