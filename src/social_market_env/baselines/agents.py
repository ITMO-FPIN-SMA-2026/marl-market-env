"""Reference market agents."""

from typing import Any

import numpy as np

from social_market_env.config import MarketConfig
from social_market_env.contracts import AgentAction, AgentObservation, MarketAgentAPI
from social_market_env.gametheory.equilibrium import best_response_price


class RandomAgent(MarketAgentAPI):
    """Sets a uniformly random price every step."""

    def __init__(self, *, low: float = 20.0, high: float = 80.0) -> None:
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(0)

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        """Reseed the price generator."""
        self.rng = np.random.default_rng(seed)

    def act(self, observation: AgentObservation) -> AgentAction:
        """Draw a price."""
        return AgentAction(price=float(self.rng.uniform(self.low, self.high)))


class BestResponseAgent(MarketAgentAPI):
    """Plays a damped best response to last step's prices, which drifts to the Nash equilibrium."""

    def __init__(
        self, *, market: dict[str, Any] | None = None, damping: float = 0.5, opening_markup: float = 5.0
    ) -> None:
        self.market = MarketConfig(**(market or {}))
        self.damping = damping
        self.opening_markup = opening_markup

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        """Nothing to reset."""

    def act(self, observation: AgentObservation) -> AgentAction:
        """Price as a best response to the others."""
        return AgentAction(price=self.best_price(observation))

    def best_price(self, observation: AgentObservation) -> float:
        """Return the damped best-response price."""
        others = np.array(
            [
                price
                for agent_id, price in sorted(observation.market.last_prices.items())
                if agent_id != observation.agent_id
            ]
        )
        if others.size == 0 or observation.last_price is None:
            return observation.cost + self.opening_markup
        response = best_response_price(cost=observation.cost, other_prices=others, market=self.market)
        return self.damping * observation.last_price + (1.0 - self.damping) * response


class CartelAgent(BestResponseAgent):
    """Joins every offered plan and keeps the recommended price; best-responds otherwise."""

    def act(self, observation: AgentObservation) -> AgentAction:
        """Follow the offer if there is one."""
        if observation.offer is None:
            return super().act(observation)
        return AgentAction(price=observation.offer.recommended_price, coalition_choice=observation.offer.plan_id)
