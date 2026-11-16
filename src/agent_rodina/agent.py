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
    * online calibration of formation chance from realized outcomes;
    * optional adaptation of damping / join bias when ``training=True``.
    """

    def __init__(
        self,
        *,
        market: dict[str, Any] | None = None,
        damping: float = 0.35,
        opening_markup: float = 5.0,
        join_slack: float = 0.05,
        max_cartel_premium: float = 0.12,
        min_payout_weight: float = 0.05,
        formation_prior: float = 0.55,
        learn_rate: float = 0.05,
        calibrate_formation: bool = True,
    ) -> None:
        self.market = MarketConfig(**(market or {}))
        self.base_damping = float(np.clip(damping, 0.0, 0.95))
        self.opening_markup = opening_markup
        self.join_slack = join_slack
        # Kept for YAML / constructor compatibility; join now uses expected rewards.
        self.max_cartel_premium = max_cartel_premium
        self.min_payout_weight = min_payout_weight
        self.base_formation_prior = float(np.clip(formation_prior, 0.05, 0.95))
        self.formation_prior = self.base_formation_prior
        self.learn_rate = learn_rate
        self.calibrate_formation = calibrate_formation

        self.agent_id = 0
        self.training = False
        self.damping = self.base_damping
        self.join_bias = 0.0
        self.reward_residual = 0.0  # EMA of (realized - predicted); corrects forecast bias
        self._ema_reward = 0.0
        self._joined_last = False
        self._solo_last = False
        self._forecast: dict[str, float] | None = None
        self.rng = np.random.default_rng(0)

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        """Prepare for a new episode; keep calibrated priors / join bias across episodes."""
        self.agent_id = agent_id
        self.training = training
        self.rng = np.random.default_rng(seed)
        self._joined_last = False
        self._solo_last = False
        self._forecast = None
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
        self._forecast = None

        if offer is not None and not observation.restricted and self._should_join(observation, solo):
            self._joined_last = True
            self._solo_last = False
            return AgentAction(price=float(offer.recommended_price), coalition_choice=int(offer.plan_id))

        self._joined_last = False
        self._solo_last = True
        e_solo = self._expected_own_profit(observation, my_price=solo)
        self._forecast = {"kind": 0.0, "predicted": e_solo + self.reward_residual, "e_solo": e_solo}
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
        """Calibrate formation chance / forecast bias; adapt damping when training."""
        del next_observation, done
        forecast = self._forecast
        self._forecast = None

        if forecast is not None:
            predicted = float(forecast.get("predicted", 0.0))
            residual = reward - predicted
            rate = max(self.learn_rate, 0.05)
            self.reward_residual = (1.0 - rate) * self.reward_residual + rate * residual

            if self.calibrate_formation and forecast.get("kind", 0.0) >= 0.5:
                e_success = float(forecast["e_success"])
                e_fail = float(forecast["e_fail"])
                # Infer whether the cartel paid like success or like alone-at-cartel-price.
                formed = abs(reward - e_success) <= abs(reward - e_fail)
                target = 1.0 if formed else 0.0
                self.formation_prior = float(
                    np.clip((1.0 - rate) * self.formation_prior + rate * target, 0.05, 0.95)
                )
                self.base_formation_prior = (
                    (1.0 - 0.5 * rate) * self.base_formation_prior + (0.5 * rate) * self.formation_prior
                )

        if not self.training:
            return

        previous = self._ema_reward
        self._ema_reward = (1.0 - self.learn_rate) * self._ema_reward + self.learn_rate * reward
        delta = self._ema_reward - previous

        # Falling rewards → react faster to rivals (lower damping).
        self.damping = float(np.clip(self.damping - self.learn_rate * np.sign(delta) * 0.05, 0.05, 0.9))
        self.base_damping = (1.0 - self.learn_rate) * self.base_damping + self.learn_rate * self.damping

        if self._joined_last:
            # Also nudge join_bias from whether join beat the stashed solo forecast.
            if forecast is not None and "e_solo" in forecast:
                beat_solo = reward >= float(forecast["e_solo"])
                step = self.learn_rate * (0.1 if beat_solo else -0.1)
            else:
                step = self.learn_rate * np.sign(delta) * 0.1
            self.join_bias = float(np.clip(self.join_bias + step, -0.2, 0.25))
        elif self._solo_last:
            self.join_bias = float(np.clip(self.join_bias - self.learn_rate * np.sign(delta) * 0.05, -0.2, 0.25))

        del observation, action

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
        success and failure payoffs with a calibrated ``formation_prior``.
        """
        offer = observation.offer
        assert offer is not None
        if offer.payout_weight < self.min_payout_weight:
            return False
        if offer.recommended_price < observation.cost:
            return False

        solo_reward = self._expected_own_profit(observation, my_price=solo_price)
        success = self._expected_coalition_share(observation, offer=offer)
        failure = self._expected_own_profit(observation, my_price=float(offer.recommended_price))
        join_reward = self._mix_join_reward(observation, success=success, failure=failure)

        # Correct systematic forecast bias learned from past residuals.
        solo_adj = solo_reward + self.reward_residual
        join_adj = join_reward + self.reward_residual

        slack = self.join_slack * max(abs(solo_adj), 1.0)
        slack += self.join_bias * max(abs(solo_adj), 1.0)

        if observation.last_evaluation is not None and observation.last_evaluation.fine > 0:
            slack -= 0.04 * max(abs(solo_adj), 1.0)
        if observation.last_profit < 0:
            slack -= 0.02 * max(abs(solo_adj), 1.0)

        decide = join_adj + slack >= solo_adj
        self._forecast = {
            "kind": 1.0 if decide else 0.0,
            "predicted": (join_reward if decide else solo_reward) + self.reward_residual,
            "e_solo": solo_reward,
            "e_join": join_reward,
            "e_success": success,
            "e_fail": failure,
        }
        return decide

    def _expected_join_reward(self, observation: AgentObservation, *, offer: CoalitionOffer) -> float:
        """Mixture of formed-cartel share and alone-at-cartel-price fallback."""
        success = self._expected_coalition_share(observation, offer=offer)
        failure = self._expected_own_profit(observation, my_price=float(offer.recommended_price))
        return self._mix_join_reward(observation, success=success, failure=failure)

    def _mix_join_reward(self, observation: AgentObservation, *, success: float, failure: float) -> float:
        p_form = float(np.clip(self.formation_prior + self.join_bias, 0.05, 0.95))
        if observation.last_evaluation is not None and observation.last_evaluation.fine > 0:
            p_form = float(np.clip(p_form - 0.1, 0.05, 0.95))
        return p_form * success + (1.0 - p_form) * failure

    def _expected_coalition_share(self, observation: AgentObservation, *, offer: CoalitionOffer) -> float:
        """Own pool share if every listed member complies at the recommended price.

        The offer only carries this agent's recommended price; we approximate that
        other members are asked to post the same cartel price. Rival unit costs are
        inferred from last prices and this agent's own margin (observation has no
        foreign costs).
        """
        cartel_price = float(offer.recommended_price)
        prices = self._price_snapshot(observation, my_price=cartel_price)
        for member in offer.members:
            prices[int(member)] = cartel_price
        outcome = clear_market(
            prices=prices,
            costs=self._estimate_costs(observation, prices=prices),
            market=self.market,
        )
        pool = sum(outcome.gross_profits.get(int(member), 0.0) for member in offer.members)
        return float(pool * offer.payout_weight)

    def _expected_own_profit(self, observation: AgentObservation, *, my_price: float) -> float:
        """Gross own profit if rivals keep last-step prices and I post ``my_price``."""
        prices = self._price_snapshot(observation, my_price=my_price)
        outcome = clear_market(
            prices=prices,
            costs=self._estimate_costs(observation, prices=prices),
            market=self.market,
        )
        return float(outcome.gross_profits.get(observation.agent_id, 0.0))

    def _estimate_costs(self, observation: AgentObservation, *, prices: dict[int, float]) -> dict[int, float]:
        """Own cost exact; rivals ≈ last price minus own recent margin (clipped)."""
        own_id = observation.agent_id
        if observation.last_price is not None:
            markup = max(0.5, float(observation.last_price) - observation.cost)
        else:
            markup = max(0.5, self.opening_markup)
        costs: dict[int, float] = {}
        for agent_id, price in prices.items():
            if agent_id == own_id:
                costs[agent_id] = float(observation.cost)
                continue
            # Prefer last observed price of that rival when estimating their cost.
            ref = float(observation.market.last_prices.get(agent_id, price))
            costs[agent_id] = float(np.clip(ref - markup, self.market.p_min * 0.5, ref - 1e-3))
        return costs

    def _price_snapshot(self, observation: AgentObservation, *, my_price: float) -> dict[int, float]:
        """Last-step prices with this agent's price overwritten."""
        prices = {int(agent_id): float(price) for agent_id, price in observation.market.last_prices.items()}
        prices[observation.agent_id] = float(my_price)
        return prices


# Backwards-compatible alias from the original AdaptiveMarketAgent package.
AdaptiveMarketAgent = RodinaAgent
