"""Rodina market agent: damped best response + selective coalition joining."""

from __future__ import annotations

from typing import Any

import numpy as np

from social_market_env.config import MarketConfig
from social_market_env.contracts import AgentAction, AgentObservation, MarketAgentAPI
from social_market_env.gametheory.equilibrium import best_response_price


class RodinaAgent(MarketAgentAPI):
    """Market agent that outperforms naive baselines via:

    * damped best-response pricing against last-step rivals;
    * selective coalition joining (follow offer when it beats or matches solo BR);
    * optional online adaptation of damping / join bias when ``training=True``.
    """

    def __init__(
        self,
        *,
        market: dict[str, Any] | None = None,
        damping: float = 0.35,
        opening_markup: float = 5.0,
        join_slack: float = 0.08,
        max_cartel_premium: float = 0.12,
        min_payout_weight: float = 0.05,
        learn_rate: float = 0.05,
    ) -> None:
        self.market = MarketConfig(**(market or {}))
        self.base_damping = float(np.clip(damping, 0.0, 0.95))
        self.opening_markup = opening_markup
        self.join_slack = join_slack
        self.max_cartel_premium = max_cartel_premium
        self.min_payout_weight = min_payout_weight
        self.learn_rate = learn_rate

        self.agent_id = 0
        self.training = False
        self.damping = self.base_damping
        self.join_bias = 0.0
        self._ema_reward = 0.0
        self._joined_last = False
        self._solo_last = False
        self.rng = np.random.default_rng(0)

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        """Prepare for a new episode; keep learned damping/join bias across episodes."""
        self.agent_id = agent_id
        self.training = training
        self.rng = np.random.default_rng(seed)
        self._joined_last = False
        self._solo_last = False
        # Mild exploration noise only while training.
        if training:
            self.damping = float(np.clip(self.base_damping + self.rng.normal(0.0, 0.02), 0.05, 0.9))
        else:
            self.damping = self.base_damping

    def act(self, observation: AgentObservation) -> AgentAction:
        """Choose price and optional coalition membership."""
        if not observation.active:
            return AgentAction(price=observation.cost + self.opening_markup, participate=False)

        solo = self._solo_price(observation)
        offer = observation.offer

        if offer is not None and not observation.restricted and self._should_join(observation, solo):
            self._joined_last = True
            self._solo_last = False
            return AgentAction(price=float(offer.recommended_price), coalition_choice=int(offer.plan_id))

        self._joined_last = False
        self._solo_last = True
        price = solo
        if self.training:
            # Small price jitter while learning; clipped by the environment sanitizer.
            price = float(price + self.rng.normal(0.0, 0.15))
        return AgentAction(price=float(price))

    def observe_transition(
        self,
        *,
        observation: AgentObservation,
        action: AgentAction,
        reward: float,
        next_observation: AgentObservation | None,
        done: bool,
    ) -> None:
        """Adapt damping and join bias from realized reward when training."""
        del observation, action, next_observation, done
        if not self.training:
            return

        previous = self._ema_reward
        self._ema_reward = (1.0 - self.learn_rate) * self._ema_reward + self.learn_rate * reward
        delta = self._ema_reward - previous

        # Falling rewards → react faster to rivals (lower damping).
        self.damping = float(np.clip(self.damping - self.learn_rate * np.sign(delta) * 0.05, 0.05, 0.9))
        self.base_damping = (1.0 - self.learn_rate) * self.base_damping + self.learn_rate * self.damping

        if self._joined_last:
            self.join_bias = float(np.clip(self.join_bias + self.learn_rate * np.sign(delta) * 0.1, -0.2, 0.25))
        elif self._solo_last:
            self.join_bias = float(np.clip(self.join_bias - self.learn_rate * np.sign(delta) * 0.05, -0.2, 0.25))

    def _solo_price(self, observation: AgentObservation) -> float:
        """Damped best response to last-step competitor prices."""
        others = np.array(
            [
                price
                for agent_id, price in sorted(observation.market.last_prices.items())
                if agent_id != observation.agent_id
            ],
            dtype=float,
        )
        if others.size == 0 or observation.last_price is None:
            return observation.cost + self.opening_markup
        response = best_response_price(cost=observation.cost, other_prices=others, market=self.market)
        return self.damping * observation.last_price + (1.0 - self.damping) * response

    def _should_join(self, observation: AgentObservation, solo_price: float) -> bool:
        """Join fair cartels; defect when the recommended price is an overpriced cartel.

        CartelAgent always complies. BestResponseAgent never joins. This policy sits in
        between: follow the plan when it is close to the solo best response, otherwise
        undercut by refusing the offer and pricing at the damped BR.
        """
        offer = observation.offer
        assert offer is not None
        if offer.payout_weight < self.min_payout_weight:
            return False
        if offer.recommended_price < observation.cost:
            return False

        relative = (offer.recommended_price - solo_price) / max(abs(solo_price), 1.0)
        lower = -self.join_slack + self.join_bias
        upper = self.max_cartel_premium + self.join_bias

        if observation.last_evaluation is not None and observation.last_evaluation.fine > 0:
            # After a fine, prefer solo play slightly more.
            upper -= 0.04
        if observation.last_profit < 0:
            lower += 0.02
            upper -= 0.02

        return lower <= relative <= upper


# Backwards-compatible alias from the original AdaptiveMarketAgent package.
AdaptiveMarketAgent = RodinaAgent
