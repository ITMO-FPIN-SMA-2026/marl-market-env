"""Mutable episode state and the per-step scratchpad."""

from dataclasses import dataclass, field

import numpy as np

from social_market_env.contracts import (
    AgentAction,
    AgentEvaluation,
    AgentExpectations,
    AgentObservation,
    ArbiterAction,
    ArbiterObservation,
    CoalitionDecision,
    CoalitionObservation,
    CoalitionPlan,
    ComplianceRecord,
    FineRecord,
    Pair,
)
from social_market_env.core.coalitions import PlanCompliance, PlanRejection
from social_market_env.core.finance import Account
from social_market_env.core.market import MarketOutcome
from social_market_env.core.network import NetworkState
from social_market_env.core.scoring import ArbiterVerdict, HarmFootprint


@dataclass
class AgentState:
    """Complete per-episode record available to the environment and centralized training."""

    observations: list[AgentObservation] = field(default_factory=list)
    expectations: AgentExpectations = field(default_factory=AgentExpectations)
    expectation_history: list[AgentExpectations] = field(default_factory=list)
    evaluations: list[AgentEvaluation] = field(default_factory=list)
    coalition_plan_ids: tuple[int, ...] = ()
    coalition_join_vector: tuple[int, ...] = ()


@dataclass
class EnvState:
    """Mutable state of one episode."""

    step: int
    """Index of the step being played."""
    rng: np.random.Generator
    """Environment randomness (network)."""
    training: bool
    """Whether components are allowed to learn."""
    accounts: dict[int, Account]
    """Reserve and status per agent."""
    agent_state: dict[int, AgentState]
    """Per-agent observations, forecasts, evaluations and coalition choices."""
    network: NetworkState
    """Hidden network."""
    last_prices: dict[int, float] = field(default_factory=dict)
    """Participants' prices on the previous step."""
    last_quantities: dict[int, float] = field(default_factory=dict)
    """Participants' sales on the previous step."""
    last_rewards: dict[int, float] = field(default_factory=dict)
    """What each agent received on the previous step."""
    last_average_price: float | None = None
    """Average price on the previous step."""
    last_total_quantity: float | None = None
    """Total demand on the previous step."""
    last_welfare: float | None = None
    """Welfare on the previous step."""
    last_arbiter_action: ArbiterAction | None = None
    """Arbiter action on the previous step."""
    compliance_history: dict[int, list[ComplianceRecord]] = field(default_factory=dict)
    """How each agent treated past plans."""
    fines_history: list[FineRecord] = field(default_factory=list)
    """All sanctions so far."""

    def active_ids(self) -> list[int]:
        """Return ids of agents still in the game."""
        return sorted(agent_id for agent_id, account in self.accounts.items() if account.active)


@dataclass
class StepContext:
    """Intermediate results of one step, filled phase by phase."""

    edges_snapshot: dict[Pair, float] = field(default_factory=dict)
    """G_t between active agents, as used by coalitions on this step."""
    coalition_observation: CoalitionObservation | None = None
    """What the coalition policy saw."""
    decision: CoalitionDecision = field(default_factory=CoalitionDecision)
    """What it proposed."""
    plans: tuple[CoalitionPlan, ...] = ()
    """Accepted plans."""
    rejections: tuple[PlanRejection, ...] = ()
    """Rejected plans."""
    agent_observations: dict[int, AgentObservation] = field(default_factory=dict)
    """What each active agent saw."""
    actions: dict[int, AgentAction] = field(default_factory=dict)
    """Sanitized actions of active agents."""
    outcome: MarketOutcome = field(default_factory=MarketOutcome.empty)
    """Market clearing result."""
    compliance: list[PlanCompliance] = field(default_factory=list)
    """Compliance per accepted plan, in plan order."""
    received: dict[int, float] = field(default_factory=dict)
    """Market income after pooling."""
    welfare: float = 0.0
    """SW on this step."""
    target_welfare: float = 0.0
    """SW* for the current active set."""
    nash_average_price: float = 0.0
    """Mean price of the Nash equilibrium for the current active set."""
    cartel_average_price: float = 0.0
    """Mean price when all active agents act as one cartel."""
    harm: HarmFootprint = field(default_factory=HarmFootprint.none)
    """Ground-truth harmful plans."""
    arbiter_observation: ArbiterObservation | None = None
    """What the arbiter saw."""
    arbiter_action: ArbiterAction = field(default_factory=ArbiterAction)
    """Sanitized arbiter action."""
    verdict: ArbiterVerdict = field(default_factory=ArbiterVerdict.clean)
    """Environment's judgement of the arbiter action."""
    arbiter_reward: float = 0.0
    """Arbiter reward."""
    fines: dict[int, float] = field(default_factory=dict)
    """Fines applied on this step."""
    link_fees: dict[int, float] = field(default_factory=dict)
    """Fees for newly formed requested links."""
    agent_rewards: dict[int, float] = field(default_factory=dict)
    """Final reward of every agent that acted."""
    coalition_reward: float = 0.0
    """Pooled income minus fines of complying members."""


def remember_step(*, state: EnvState, context: StepContext) -> None:
    """Store this step's results as 'previous step' data for the next observations."""
    state.last_prices = dict(context.outcome.prices)
    state.last_quantities = dict(context.outcome.quantities)
    state.last_rewards = dict(context.agent_rewards)
    state.last_average_price = context.outcome.average_price if context.outcome.prices else None
    state.last_total_quantity = context.outcome.total_quantity
    state.last_welfare = context.welfare
    state.last_arbiter_action = context.arbiter_action
    for record in context.compliance:
        _remember_compliance(state=state, record=record)
    state.fines_history.extend(
        FineRecord(step=state.step, target=target, fine=fine) for target, fine in sorted(context.fines.items())
    )
    _remember_agents(state=state, context=context)


def _remember_agents(*, state: EnvState, context: StepContext) -> None:
    plan_ids = tuple(plan.plan_id for plan in context.plans)
    for agent_id, observation in context.agent_observations.items():
        memory = state.agent_state[agent_id]
        action = context.actions[agent_id]
        gross_profit = context.outcome.gross_profits.get(agent_id, 0.0)
        payout = context.received.get(agent_id, 0.0) - gross_profit
        reward = context.agent_rewards[agent_id]
        chosen = next(
            (plan.plan_id for plan in context.plans
             if agent_id in plan.members and action.participate and action.coalition_choice == plan.plan_id),
            None,
        )
        expected = observation.expectations
        memory.coalition_plan_ids = plan_ids
        memory.coalition_join_vector = tuple(int(plan_id == chosen) for plan_id in plan_ids)
        memory.evaluations.append(AgentEvaluation(
            step=state.step,
            price=context.outcome.prices.get(agent_id),
            quantity=context.outcome.quantities.get(agent_id, 0.0),
            gross_profit=gross_profit,
            coalition_payout=payout,
            fine=context.fines.get(agent_id, 0.0),
            link_fee=context.link_fees.get(agent_id, 0.0),
            reward=reward,
            profit_error=None if expected.gross_profit is None else gross_profit - expected.gross_profit,
            reward_error=None if expected.reward is None else reward - expected.reward,
            joined_plan_id=chosen,
        ))
        memory.expectations = AgentExpectations(
            average_price=context.outcome.average_price,
            quantity=context.outcome.quantities.get(agent_id, 0.0),
            gross_profit=gross_profit,
            coalition_payout=payout,
            reward=reward,
        )
        memory.expectation_history.append(memory.expectations)


def _remember_compliance(*, state: EnvState, record: PlanCompliance) -> None:
    for agent_id in record.compliers + record.defectors + record.refusers:
        entry = ComplianceRecord(step=state.step, plan_id=record.plan_id, complied=agent_id in record.compliers)
        state.compliance_history.setdefault(agent_id, []).append(entry)
