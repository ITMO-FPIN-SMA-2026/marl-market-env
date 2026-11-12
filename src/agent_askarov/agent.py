"""Economic price proposals with independent tabular Q-learning."""

import json
import math
from bisect import bisect_right
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from social_market_env.config import MarketConfig
from social_market_env.contracts import (
    AgentAction,
    AgentObservation,
    LinkAction,
    LinkActionKind,
    MarketAgentAPI,
)
from social_market_env.core.market import share_vector
from social_market_env.gametheory.equilibrium import best_response_price

ACTIONS = ('best_response', 'cautious', 'bold', 'hold', 'coalition', 'wait')
CHECKPOINT_VERSION = 1


class AskarovAgent(MarketAgentAPI):
    """Learn pricing, participation and coalition acceptance from local rewards."""

    def __init__(
        self,
        *,
        market: dict[str, Any] | None = None,
        learning_rate: float = 0.15,
        gamma: float = 0.9,
        epsilon: float = 0.25,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        reward_scale: float = 100.0,
        opening_markup: float = 5.0,
        use_memory: bool = True,
        manage_links: bool = True,
        max_link_fee: float = 5.0,
        checkpoint_path: str | None = None,
    ) -> None:
        self.market = MarketConfig(**(market or {}))
        parameters = {
            'learning_rate': learning_rate, 'gamma': gamma, 'epsilon': epsilon,
            'epsilon_min': epsilon_min, 'epsilon_decay': epsilon_decay,
            'reward_scale': reward_scale, 'opening_markup': opening_markup,
            'max_link_fee': max_link_fee,
        }
        if not all(math.isfinite(value) for value in parameters.values()):
            raise ValueError('Agent parameters must be finite')
        if not 0 < learning_rate <= 1 or not 0 <= gamma < 1:
            raise ValueError('Require 0 < learning_rate <= 1 and 0 <= gamma < 1')
        if not 0 <= epsilon_min <= epsilon <= 1 or not 0 < epsilon_decay <= 1:
            raise ValueError('Invalid epsilon schedule')
        if reward_scale <= 0 or opening_markup < 0 or max_link_fee < 0:
            raise ValueError('Invalid reward scale, markup or link fee')
        if self.market.tau <= 0 or self.market.demand_b <= 0 or not 0 < self.market.p_min < self.market.p_max:
            raise ValueError('Invalid market parameters')
        if self.market.share_model not in {'logit', 'tz_linear'}:
            raise ValueError('Unknown share model')
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_scale = reward_scale
        self.opening_markup = opening_markup
        self.use_memory = use_memory
        self.manage_links = manage_links
        self.max_link_fee = max_link_fee
        self.q_table: dict[str, dict[str, float]] = {}
        self.updates = 0
        self.training = False
        self.rng = np.random.default_rng(0)
        self._pending: tuple[int, str, str, dict[str, float], AgentAction] | None = None
        if checkpoint_path is not None:
            self.load(checkpoint_path)

    @property
    def exploration_rate(self) -> float:
        """Exploration decays across training episodes and is off in evaluation."""
        if not self.training:
            return 0.0
        return max(self.epsilon_min, self.epsilon * self.epsilon_decay**self.updates)

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        """Reset episode state, preserving the learned table and update count."""
        self.training = training
        self.rng = np.random.default_rng(seed)
        self._pending = None

    def act(self, observation: AgentObservation) -> AgentAction:
        """Choose among legal proposals; the evaluation policy is deterministic."""
        candidates = self._candidates(observation)
        state = self._state_key(observation)
        priors = self._priors(observation, candidates)
        values = self.q_table.get(state, {})
        if self.training and self.rng.random() < self.exploration_rate:
            name = str(self.rng.choice(list(candidates)))
        else:
            name = max(candidates, key=lambda item: values.get(item, priors[item]))
        action = candidates[name]
        if self.manage_links and action.participate:
            action = replace(action, link_action=self._link_request(observation, action))
        if self.training:
            self._pending = (observation.step, state, name, priors, action)
        return action

    def observe_transition(
        self,
        *,
        observation: AgentObservation,
        action: AgentAction,
        reward: float,
        next_observation: AgentObservation | None,
        done: bool,
    ) -> None:
        if not self.training or self._pending is None:
            return
        step, state, name, priors, requested = self._pending
        self._pending = None
        if step != observation.step or not math.isfinite(reward):
            return
        if (action.price, action.participate, action.coalition_choice) != (
            requested.price, requested.participate, requested.coalition_choice
        ):
            return
        target = reward / self.reward_scale
        if not done and next_observation is not None and next_observation.active:
            candidates = self._candidates(next_observation)
            next_priors = self._priors(next_observation, candidates)
            next_values = self.q_table.get(self._state_key(next_observation), {})
            target += self.gamma * max(next_values.get(item, next_priors[item]) for item in candidates)
        row = self.q_table.setdefault(state, {})
        current = row.get(name, priors[name])
        row[name] = current + self.learning_rate * (target - current)
        self.updates += 1

    def _rivals(self, observation: AgentObservation) -> dict[int, float]:
        return {
            other: observation.market.last_prices.get(other, self._clip(observation.cost + self.opening_markup))
            for other in sorted(observation.link_costs)
            if other != observation.agent_id
        }

    def _clip(self, price: float) -> float:
        return float(np.clip(price, self.market.p_min, self.market.p_max))

    def _candidates(self, observation: AgentObservation) -> dict[str, AgentAction]:
        floor = max(self.market.p_min, observation.cost)
        if not observation.active or floor > self.market.p_max:
            return {'wait': AgentAction(price=self.market.p_max, participate=False)}
        response = max(floor, best_response_price(
            cost=observation.cost,
            other_prices=np.array(list(self._rivals(observation).values())),
            market=self.market,
        ))
        candidates = {
            'best_response': AgentAction(price=self._clip(response)),
            'cautious': AgentAction(price=self._clip(max(floor, observation.cost + 0.75 * (response - observation.cost)))),
            'bold': AgentAction(price=self._clip(max(floor, observation.cost + 1.25 * (response - observation.cost)))),
        }
        if observation.last_price is not None:
            candidates['hold'] = AgentAction(price=self._clip(max(floor, observation.last_price)))
        offer = observation.offer
        if (offer is not None and not observation.restricted
                and floor <= offer.recommended_price <= self.market.p_max):
            candidates['coalition'] = AgentAction(
                price=float(offer.recommended_price), coalition_choice=offer.plan_id
            )
        candidates['wait'] = AgentAction(price=self._clip(response), participate=False)
        return candidates

    def _priors(self, observation: AgentObservation, candidates: dict[str, AgentAction]) -> dict[str, float]:
        """Initialize unknown choices with an approximate economic Q value."""
        rivals = list(self._rivals(observation).values())
        result = {}
        for name, action in candidates.items():
            estimate = 0.0
            if action.participate:
                prices = np.array([action.price, *rivals])
                quantity = max(0.0, self.market.demand_a - self.market.demand_b * float(prices.mean()))
                estimate = (action.price - observation.cost) * quantity * float(share_vector(prices=prices, market=self.market)[0])
                if name == 'coalition' and observation.offer is not None:
                    estimate *= len(observation.offer.members) * observation.offer.payout_weight
                evaluation = observation.last_evaluation
                if (self.use_memory and evaluation is not None and evaluation.fine > 0
                        and evaluation.price is not None and action.price >= evaluation.price - 0.5):
                    estimate -= evaluation.fine
            result[name] = estimate / (self.reward_scale * (1.0 - self.gamma))
        return result

    def _state_key(self, observation: AgentObservation) -> str:
        """Coarse local context, independent of exact agent/plan identifiers."""
        rivals = self._rivals(observation)
        reference = float(np.mean(list(rivals.values()))) if rivals else observation.cost + self.opening_markup
        scale = max(self.market.tau, (self.market.p_max - self.market.p_min) * 0.05)
        regime = bisect_right((0.5, 1.5, 4.0, 8.0), (reference - observation.cost) / scale)
        reserve = 2 if observation.restricted else int(observation.balance < self.reward_scale)
        size = premium = fairness = 0
        if observation.offer is not None and not observation.restricted:
            offer = observation.offer
            size = 1 if len(offer.members) <= 3 else 2
            premium = bisect_right((0.0, 2.0, 6.0), (offer.recommended_price - reference) / scale)
            fairness = bisect_right((0.75, 1.25), offer.payout_weight * len(offer.members))
        fined = int(self.use_memory and observation.last_evaluation is not None and observation.last_evaluation.fine > 0)
        return '|'.join(map(str, (regime, reserve, size, premium, fairness, fined)))

    def _link_request(self, observation: AgentObservation, action: AgentAction) -> LinkAction | None:
        if observation.restricted:
            return None
        hostile = [other for other, trust in sorted(observation.neighbors.items()) if trust < -0.25]
        if hostile:
            target = min(hostile, key=lambda other: observation.neighbors[other])
            return LinkAction(kind=LinkActionKind.remove, target=target)
        if action.coalition_choice is not None and observation.offer is not None:
            partners = [other for other in sorted(observation.neighbors) if other in observation.offer.members]
            if partners:
                target = max(partners, key=lambda other: observation.neighbors[other])
                return LinkAction(kind=LinkActionKind.maintain, target=target)
        budget = min(self.max_link_fee, max(0.0, observation.balance) * 0.01)
        candidates = [
            other for other, fee in sorted(observation.link_costs.items())
            if other != observation.agent_id and other not in observation.neighbors and fee <= budget
            and observation.market.last_prices.get(other, 0.0) > observation.cost
        ]
        if candidates:
            target = min(candidates, key=lambda other: (observation.link_costs[other], other))
            return LinkAction(kind=LinkActionKind.add, target=target)
        return None

    def _settings(self) -> dict[str, Any]:
        names = (
            'learning_rate', 'gamma', 'epsilon', 'epsilon_min', 'epsilon_decay',
            'reward_scale', 'opening_markup', 'use_memory', 'manage_links', 'max_link_fee',
        )
        return {name: getattr(self, name) for name in names}

    def save(self, path: str | Path) -> None:
        """Save a portable JSON checkpoint; no pickle or implicit disk writes."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': CHECKPOINT_VERSION, 'actions': list(ACTIONS),
            'market': asdict(self.market), 'settings': self._settings(),
            'updates': self.updates, 'q_table': self.q_table,
        }
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(destination)

    def load(self, path: str | Path) -> None:
        """Restore learning; reject incompatible market/settings explicitly."""
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('version') != CHECKPOINT_VERSION or payload.get('actions') != list(ACTIONS):
            raise ValueError('Incompatible AskarovAgent checkpoint version/actions')
        if payload.get('market') != asdict(self.market) or payload.get('settings') != self._settings():
            raise ValueError('Checkpoint market/settings differ from this agent')
        table = payload.get('q_table')
        updates = payload.get('updates')
        if not isinstance(table, dict) or type(updates) is not int or updates < 0:
            raise ValueError('Invalid checkpoint table/update count')
        for state, row in table.items():
            if (not isinstance(state, str) or not isinstance(row, dict) or not set(row) <= set(ACTIONS)
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in row.values())):
                raise ValueError('Invalid checkpoint Q values')
        self.q_table = table
        self.updates = updates
        self._pending = None

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, market: dict[str, Any] | None = None) -> 'AskarovAgent':
        """Construct with the saved settings, checking an optional target market."""
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(
            market=payload['market'] if market is None else market,
            **payload['settings'], checkpoint_path=str(path),
        )
