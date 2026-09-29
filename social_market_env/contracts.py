from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AgentObservation:
    agent_id: int
    step: int
    cost: float
    balance: float
    last_profit: float
    last_price: Optional[float]
    market_average_price: Optional[float]
    local_neighbors: Tuple[int, ...]
    active: bool


@dataclass(frozen=True)
class AgentAction:
    price: float
    participate: bool = True
    coalition_choice: Optional[int] = None
    link_action: Optional[Tuple[str, int]] = None  # ('add'|'remove'|'maintain', target_id)


@dataclass(frozen=True)
class CoalitionObservation:
    step: int
    active_agents: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    recent_profits: Dict[int, float]
    recent_prices: Dict[int, Optional[float]]


@dataclass(frozen=True)
class CoalitionDecision:
    coalitions: Tuple[Tuple[int, ...], ...] = ()


class ArbiterActionType(str, Enum):
    OBSERVE = "observe"
    CHANGE_LINK_COST = "change_link_cost"
    SANCTION = "sanction"


@dataclass(frozen=True)
class ArbiterObservation:
    step: int
    active_agents: Tuple[int, ...]
    prices: Dict[int, float]
    quantities: Dict[int, float]
    total_quantity: float
    average_price: float
    social_welfare: float
    delta_average_price: float
    delta_social_welfare: float
    # Intentionally NO true graph and NO true coalitions.


@dataclass(frozen=True)
class ArbiterAction:
    kind: ArbiterActionType = ArbiterActionType.OBSERVE
    target_agent: Optional[int] = None
    fine: float = 0.0
    link_cost_delta: float = 0.0


class MarketAgentAPI(ABC):
    """Public API contract for a market-agent SDK/repository."""

    @abstractmethod
    def reset(self, agent_id: int, seed: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def act(self, observation: AgentObservation) -> AgentAction:
        ...

    def observe_transition(
        self,
        observation: AgentObservation,
        action: AgentAction,
        reward: float,
        next_observation: Optional[AgentObservation],
        done: bool,
    ) -> None:
        """Optional hook for online learning. A non-learning model may ignore it."""
        return None


class CoalitionPolicyAPI(ABC):
    """Public API contract for the coalition/network SDK/repository."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def propose(self, observation: CoalitionObservation) -> CoalitionDecision:
        ...


class ArbiterAPI(ABC):
    """Public API contract for the arbiter SDK/repository."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def act(self, observation: ArbiterObservation) -> ArbiterAction:
        ...

    def observe_transition(
        self,
        observation: ArbiterObservation,
        action: ArbiterAction,
        reward: float,
        next_observation: Optional[ArbiterObservation],
        done: bool,
    ) -> None:
        return None
