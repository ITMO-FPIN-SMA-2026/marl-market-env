"""Mutable episode state and the per-step scratchpad."""

from dataclasses import dataclass, field

import numpy as np

from social_market_env.contracts import (
    AgentAction,
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


def _remember_compliance(*, state: EnvState, record: PlanCompliance) -> None:
    for agent_id in record.compliers + record.defectors + record.refusers:
        entry = ComplianceRecord(step=state.step, plan_id=record.plan_id, complied=agent_id in record.compliers)
        state.compliance_history.setdefault(agent_id, []).append(entry)
