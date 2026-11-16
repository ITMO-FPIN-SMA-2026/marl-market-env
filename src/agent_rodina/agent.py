"""Rodina market agent: damped best response + reward-based coalition joining."""

from __future__ import annotations

from typing import Any

import numpy as np

from social_market_env.config import MarketConfig
from social_market_env.contracts import AgentAction, AgentObservation, CoalitionOffer, MarketAgentAPI
from social_market_env.core.market import clear_market
from social_market_env.gametheory.equilibrium import best_response_price


class RodinaAgent(MarketAgentAPI):
    """Market agent that outperforms naive baselines via:

    * damped best-response pricing against last-step rivals;
    * preferential coalition joining by comparing expected solo vs coalition reward
      (including the risk that other members refuse and the cartel never forms);
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
        formation_prior: float = 0.7,
        learn_rate: float = 0.05,
    ) -> None:
        self.market = MarketConfig(**(market or {}))
        self.base_damping = float(np.clip(damping, 0.0, 0.95))
        self.opening_markup = opening_markup
        self.join_slack = join_slack
        # Kept for YAML / constructor compatibility; join now uses expected rewards.
        self.max_cartel_premium = max_cartel_premium
        self.min_payout_weight = min_payout_weight
        self.formation_prior = float(np.clip(formation_prior, 0.05, 0.95))
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
        """Join when expected coalition reward beats expected solo reward.

        Accepting an offer commits this agent to ``recommended_price``. If other
        members refuse, the environment still clears the market at that price and
        the agent keeps only its own gross profit (no fallback to the solo BR
        price). The observation does not expose a failure probability, so we mix
        success and failure payoffs with ``formation_prior`` (+ learned join bias).
        """
        offer = observation.offer
        assert offer is not None
        if offer.payout_weight < self.min_payout_weight:
            return False
        if offer.recommended_price < observation.cost:
            return False

        solo_reward = self._expected_own_profit(observation, my_price=solo_price)
        join_reward = self._expected_join_reward(observation, offer=offer)

        slack = self.join_slack * max(abs(solo_reward), 1.0)
        # join_bias also nudges the comparison directly (same units as relative slack).
        slack += self.join_bias * max(abs(solo_reward), 1.0)

        if observation.last_evaluation is not None and observation.last_evaluation.fine > 0:
            # After a fine, require a clearer advantage to join again.
            slack -= 0.04 * max(abs(solo_reward), 1.0)
        if observation.last_profit < 0:
            slack -= 0.02 * max(abs(solo_reward), 1.0)

        return join_reward + slack >= solo_reward

    def _expected_join_reward(self, observation: AgentObservation, *, offer: CoalitionOffer) -> float:
        """Mixture of formed-cartel share and alone-at-cartel-price fallback."""
        success = self._expected_coalition_share(observation, offer=offer)
        failure = self._expected_own_profit(observation, my_price=float(offer.recommended_price))
        p_form = float(np.clip(self.formation_prior + self.join_bias, 0.05, 0.95))
        if observation.last_evaluation is not None and observation.last_evaluation.fine > 0:
            p_form = float(np.clip(p_form - 0.1, 0.05, 0.95))
        return p_form * success + (1.0 - p_form) * failure

    def _expected_coalition_share(self, observation: AgentObservation, *, offer: CoalitionOffer) -> float:
        """Own pool share if every listed member complies at the recommended price.

        The offer only carries this agent's recommended price; we approximate that
        other members are asked to post the same cartel price. Unknown rival costs
        are proxied by this agent's cost (observation has no foreign costs).
        """
        cartel_price = float(offer.recommended_price)
        prices = self._price_snapshot(observation, my_price=cartel_price)
        for member in offer.members:
            prices[int(member)] = cartel_price
        outcome = clear_market(
            prices=prices,
            costs={agent_id: observation.cost for agent_id in prices},
            market=self.market,
        )
        pool = sum(outcome.gross_profits.get(int(member), 0.0) for member in offer.members)
        return float(pool * offer.payout_weight)

    def _expected_own_profit(self, observation: AgentObservation, *, my_price: float) -> float:
        """Gross own profit if rivals keep last-step prices and I post ``my_price``."""
        prices = self._price_snapshot(observation, my_price=my_price)
        outcome = clear_market(
            prices=prices,
            costs={agent_id: observation.cost for agent_id in prices},
            market=self.market,
        )
        return float(outcome.gross_profits.get(observation.agent_id, 0.0))

    def _price_snapshot(self, observation: AgentObservation, *, my_price: float) -> dict[int, float]:
        """Last-step prices with this agent's price overwritten."""
        prices = {int(agent_id): float(price) for agent_id, price in observation.market.last_prices.items()}
        prices[observation.agent_id] = float(my_price)
        if observation.agent_id not in prices:
            prices[observation.agent_id] = float(my_price)
        return prices


# Backwards-compatible alias from the original AdaptiveMarketAgent package.
AdaptiveMarketAgent = RodinaAgent
