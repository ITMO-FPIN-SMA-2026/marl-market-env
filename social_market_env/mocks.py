from __future__ import annotations

import random
from typing import Optional

from .contracts import (
    AgentAction,
    AgentObservation,
    ArbiterAction,
    ArbiterActionType,
    ArbiterAPI,
    ArbiterObservation,
    CoalitionDecision,
    CoalitionObservation,
    CoalitionPolicyAPI,
    MarketAgentAPI,
)


class RandomMarketAgent(MarketAgentAPI):
    """A tiny mock that lets the environment run before RL models exist."""

    def __init__(self, min_price: float = 10.0, max_price: float = 90.0):
        self.min_price = min_price
        self.max_price = max_price
        self.rng = random.Random()

    def reset(self, agent_id: int, seed: Optional[int] = None) -> None:
        self.rng.seed(None if seed is None else seed + agent_id)

    def act(self, observation: AgentObservation) -> AgentAction:
        if not observation.active:
            return AgentAction(price=self.min_price, participate=False)
        return AgentAction(price=self.rng.uniform(self.min_price, self.max_price))


class ConnectedComponentsCoalitionPolicy(CoalitionPolicyAPI):
    """Mock coalition policy: connected components of size >=2 become coalitions."""

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def propose(self, observation: CoalitionObservation) -> CoalitionDecision:
        adjacency = {a: set() for a in observation.active_agents}
        for u, v in observation.edges:
            if u in adjacency and v in adjacency:
                adjacency[u].add(v)
                adjacency[v].add(u)

        seen = set()
        coalitions = []
        for node in observation.active_agents:
            if node in seen:
                continue
            stack = [node]
            comp = []
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.append(cur)
                stack.extend(adjacency[cur] - seen)
            if len(comp) >= 2:
                coalitions.append(tuple(sorted(comp)))
        return CoalitionDecision(coalitions=tuple(coalitions))


class ThresholdArbiter(ArbiterAPI):
    """Mock arbiter: sanctions the highest-price agent after a price jump."""

    def __init__(self, jump_threshold: float = 5.0, fine: float = 10.0):
        self.jump_threshold = jump_threshold
        self.fine = fine

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, observation: ArbiterObservation) -> ArbiterAction:
        if observation.delta_average_price > self.jump_threshold and observation.prices:
            target = max(observation.prices, key=observation.prices.get)
            return ArbiterAction(
                kind=ArbiterActionType.SANCTION,
                target_agent=target,
                fine=self.fine,
            )
        return ArbiterAction(kind=ArbiterActionType.OBSERVE)
