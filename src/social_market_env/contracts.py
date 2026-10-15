"""Public contracts between the environment and team packages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

CONTRACTS_VERSION: str = '2.0.0'

Pair = tuple[int, int]


class LinkActionKind(StrEnum):
    """What an agent wants to do with a link."""

    add = 'add'
    """Ask for a new link; the requester pays c_ij if it forms."""
    remove = 'remove'
    """Drop an existing link for the next step."""
    maintain = 'maintain'
    """Ask to keep an existing link."""


@dataclass(frozen=True)
class LinkAction:
    """Agent's request about one link."""

    kind: LinkActionKind
    """Requested operation."""
    target: int
    """Agent on the other end of the link."""


@dataclass(frozen=True)
class MarketInfo:
    """Public market data from the previous step."""

    last_prices: dict[int, float]
    """Prices of all participants on the previous step."""
    average_price: float | None
    """Average price on the previous step, None on the first step."""
    total_quantity: float | None
    """Total demand on the previous step, None on the first step."""


@dataclass(frozen=True)
class CoalitionOffer:
    """Coalition plan as seen by one of its members."""

    plan_id: int
    """Put this into AgentAction.coalition_choice to join."""
    members: tuple[int, ...]
    """All members of the plan."""
    recommended_price: float
    """Price this member is asked to set."""
    payout_weight: float
    """This member's share of the coalition pool."""


@dataclass(frozen=True)
class AgentExpectations:
    """Agent-local one-step forecasts, initially unknown.

    The environment supplies a persistence baseline from the last realized step;
    learning agents may maintain their own richer forecasts internally.
    """

    average_price: float | None = None
    quantity: float | None = None
    gross_profit: float | None = None
    coalition_payout: float | None = None
    reward: float | None = None


@dataclass(frozen=True)
class AgentEvaluation:
    """Realized outcome and forecast errors for one agent action."""

    step: int
    price: float | None
    quantity: float
    gross_profit: float
    coalition_payout: float
    fine: float
    link_fee: float
    reward: float
    profit_error: float | None
    reward_error: float | None
    joined_plan_id: int | None


@dataclass(frozen=True)
class AgentObservation:
    """Everything a market agent sees before choosing an action."""

    agent_id: int
    """Agent identifier."""
    step: int
    """Current step, starting from 0."""
    cost: float
    """Own unit cost."""
    balance: float
    """Own financial reserve."""
    active: bool
    """False once the agent is eliminated."""
    restricted: bool
    """True while the reserve is negative: no pricing below cost, no coalitions."""
    last_price: float | None
    """Own price on the previous step."""
    last_quantity: float
    """Own sales on the previous step."""
    last_profit: float
    """What the agent received on the previous step after pooling, fines and link fees."""
    market: MarketInfo
    """Public market data."""
    neighbors: dict[int, float]
    """Active neighbors in the network with link weights."""
    link_costs: dict[int, float]
    """Current cost of forming a link to every other active agent."""
    offer: CoalitionOffer | None
    """Coalition plan offered to this agent on this step."""
    expectations: AgentExpectations = field(default_factory=AgentExpectations)
    """Forecast based on the preceding realized step."""
    last_evaluation: AgentEvaluation | None = None
    """Evaluation of this agent's preceding action."""
    last_coalition_plan_ids: tuple[int, ...] = ()
    """Plan IDs indexing last_coalition_join_vector."""
    last_coalition_join_vector: tuple[int, ...] = ()
    """One-hot choice among accepted plans on the preceding step."""


@dataclass(frozen=True)
class AgentAction:
    """Market agent's decision for one step."""

    price: float
    """Price to set; clipped to [p_min, p_max]."""
    participate: bool = True
    """Whether to sell on this step."""
    coalition_choice: int | None = None
    """Plan id to join, or None."""
    link_action: LinkAction | None = None
    """Optional request about a link."""


@dataclass(frozen=True)
class ComplianceRecord:
    """How one agent treated one coalition plan."""

    step: int
    """Step of the plan."""
    plan_id: int
    """Plan identifier."""
    complied: bool
    """True if the agent joined and kept the recommended price."""


@dataclass(frozen=True)
class CoalitionObservation:
    """Everything the coalition policy sees before proposing plans."""

    step: int
    """Current step."""
    active_agents: tuple[int, ...]
    """Agents still in the game."""
    restricted: tuple[int, ...]
    """Active agents with negative reserve; they cannot be plan members."""
    costs: dict[int, float]
    """Unit costs of active agents."""
    balances: dict[int, float]
    """Reserves of active agents."""
    edges: dict[Pair, float]
    """Links between active agents with their weights."""
    link_costs: dict[Pair, float]
    """Cost of forming each link."""
    last_prices: dict[int, float]
    """Prices of participants on the previous step."""
    last_profits: dict[int, float]
    """What each active agent received on the previous step."""
    compliance_history: dict[int, tuple[ComplianceRecord, ...]]
    """Per agent, how it treated past plans."""


@dataclass(frozen=True)
class CoalitionPlan:
    """One coalition proposal."""

    plan_id: int
    """Unique id within one decision."""
    members: tuple[int, ...]
    """Members; must be connected in the current network."""
    recommended_prices: dict[int, float]
    """Price for every member."""
    payout_weights: dict[int, float]
    """Share of the pool for every member; non-negative, sums to 1."""


@dataclass(frozen=True)
class CoalitionDecision:
    """Coalition policy's decision for one step."""

    plans: tuple[CoalitionPlan, ...] = ()
    """Proposed plans; invalid ones are rejected and logged."""


class ArbiterActionKind(StrEnum):
    """Arbiter's action classes from the assignment."""

    observe = 'observe'
    """Do nothing."""
    change_link_cost = 'change_link_cost'
    """Change the cost of forming selected links."""
    sanction = 'sanction'
    """Fine one agent."""


@dataclass(frozen=True)
class ArbiterAction:
    """Arbiter's decision for one step."""

    kind: ArbiterActionKind = ArbiterActionKind.observe
    """Action class."""
    link_cost_changes: dict[Pair, float] = field(default_factory=dict)
    """For change_link_cost: value added to c_ij of each listed pair."""
    target: int | None = None
    """For sanction: agent to fine."""
    fine: float = 0.0
    """For sanction: fine amount, clipped to [0, max_fine]."""


@dataclass(frozen=True)
class FineRecord:
    """One sanction applied by the arbiter."""

    step: int
    """Step of the sanction."""
    target: int
    """Fined agent."""
    fine: float
    """Applied amount."""


@dataclass(frozen=True)
class ArbiterObservation:
    """Market data the arbiter sees; never the network or coalitions."""

    step: int
    """Current step."""
    active_agents: tuple[int, ...]
    """Agents still in the game."""
    prices: dict[int, float]
    """Prices of participants on this step."""
    quantities: dict[int, float]
    """Sales of participants on this step."""
    total_quantity: float
    """Total demand on this step."""
    average_price: float
    """Average price on this step."""
    social_welfare: float
    """Consumer surplus on this step."""
    target_social_welfare: float
    """Consumer surplus at the competitive (Nash) equilibrium."""
    delta_average_price: float
    """Change of the average price since the previous step."""
    delta_social_welfare: float
    """Change of the welfare since the previous step."""
    last_action: ArbiterAction | None
    """Arbiter's own action on the previous step."""
    fines_history: tuple[FineRecord, ...]
    """All sanctions applied so far."""


class MarketAgentAPI(ABC):
    """Contract every market agent implementation follows."""

    @abstractmethod
    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        """Prepare for a new episode."""

    @abstractmethod
    def act(self, observation: AgentObservation) -> AgentAction:
        """Choose an action for the current step."""

    def observe_transition(
        self,
        *,
        observation: AgentObservation,
        action: AgentAction,
        reward: float,
        next_observation: AgentObservation | None,
        done: bool,
    ) -> None:
        """Receive the outcome of an action; learning agents update here."""


class CoalitionPolicyAPI(ABC):
    """Contract every coalition policy implementation follows."""

    @abstractmethod
    def reset(self, *, seed: int, training: bool) -> None:
        """Prepare for a new episode."""

    @abstractmethod
    def act(self, observation: CoalitionObservation) -> CoalitionDecision:
        """Propose coalition plans for the current step."""

    def observe_transition(
        self,
        *,
        observation: CoalitionObservation,
        action: CoalitionDecision,
        reward: float,
        next_observation: CoalitionObservation | None,
        done: bool,
    ) -> None:
        """Receive the pooled income minus fines of complying members."""


class ArbiterAPI(ABC):
    """Contract every arbiter implementation follows."""

    @abstractmethod
    def reset(self, *, seed: int, training: bool) -> None:
        """Prepare for a new episode."""

    @abstractmethod
    def act(self, observation: ArbiterObservation) -> ArbiterAction:
        """Choose an intervention for the current step."""

    def observe_transition(
        self,
        *,
        observation: ArbiterObservation,
        action: ArbiterAction,
        reward: float,
        next_observation: ArbiterObservation | None,
        done: bool,
    ) -> None:
        """Receive the arbiter reward for the last action."""
