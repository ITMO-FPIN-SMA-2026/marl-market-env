"""Masked tabular Q-learning using the public market-agent contract 2.0.0."""
from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from social_market_env.config import MarketConfig
from social_market_env.contracts import CONTRACTS_VERSION, AgentAction, AgentObservation, MarketAgentAPI


@dataclass(frozen=True)
class TverdokhlebConfig:
    """Serializable hyperparameters; history is the default state representation."""

    state_mode: str = 'history'
    min_price: float = 1.0
    max_price: float = 100.0
    price_levels: int = 33
    market_bins: int = 6
    learning_rate: float = 0.15
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    reward_scale: float = 100.0
    history_size: int = 5
    balance_reference: float = 1000.0
    allow_loss_leading: bool = False

    def __post_init__(self) -> None:
        if self.state_mode not in ('basic', 'context', 'history'):
            raise ValueError("state_mode must be 'basic', 'context' or 'history'")
        if type(self.allow_loss_leading) is not bool:
            raise ValueError('allow_loss_leading must be a bool')
        for name, value in asdict(self).items():
            if name not in ('state_mode', 'allow_loss_leading') and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f'{name} must be a finite number')
        for name in ('price_levels', 'market_bins', 'history_size'):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name == 'price_levels' else 1):
                raise ValueError(f'invalid {name}')
        if not 0 < self.min_price < self.max_price:
            raise ValueError('require 0 < min_price < max_price')
        if not 0 < self.learning_rate <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError('invalid learning_rate or gamma')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('invalid epsilon range')
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError('epsilon_decay must be in (0, 1]')
        if self.reward_scale <= 0 or self.balance_reference <= 0:
            raise ValueError('reward_scale and balance_reference must be positive')


class TverdokhlebAgent(MarketAgentAPI):
    """Learns a grid price, following an offer, or abstaining; links stay unset.

    All state variants have the same action space. Constructor arguments are
    YAML-compatible, and the environment runner injects its market configuration.
    """

    FORMAT_VERSION = 3
    ACTION_LAYOUT = 'absolute-price-grid+accept-offer+abstain'

    def __init__(
        self, config: TverdokhlebConfig | dict | None = None, *, market: dict | None = None,
        model_path: str | Path | None = None, checkpoint_path: str | Path | None = None,
    ) -> None:
        if model_path is not None:
            if config is not None:
                raise ValueError('model_path supplies its own config; do not also pass config')
            restored = self.load(model_path)
            if market is not None and asdict(MarketConfig(**market)) != restored.market:
                raise ValueError('checkpoint market differs from the experiment; retrain for this market')
            self.__dict__.update(restored.__dict__)
            self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
            return
        supplied = asdict(config) if isinstance(config, TverdokhlebConfig) else dict(config or {})
        if market is None:
            market = {'p_min': supplied.get('min_price', 1.0), 'p_max': supplied.get('max_price', 100.0)}
        self.market = asdict(MarketConfig(**market))
        for name, market_name in (('min_price', 'p_min'), ('max_price', 'p_max')):
            if name in supplied and supplied[name] != self.market[market_name]:
                raise ValueError('agent price bounds differ from the experiment market')
            supplied[name] = self.market[market_name]
        self.config = TverdokhlebConfig(**supplied)
        c = self.config
        self.prices = tuple(c.min_price + i * (c.max_price - c.min_price) / (c.price_levels - 1)
                            for i in range(c.price_levels))
        self.accept_index = len(self.prices)
        self.abstain_index = self.accept_index + 1
        self.n_actions = len(self.prices) + 2
        self.training = False
        self.q: dict[tuple[int, ...], list[float]] = {}
        self.updates = 0
        self.metadata: dict = {}
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self._episode_checkpoint: Path | None = None
        self.rng = random.Random()
        self.recent_rewards: deque[float] = deque(maxlen=c.history_size)
        self.previous_market_price: float | None = None
        self.decisions = self.known_state_decisions = 0

    @property
    def epsilon(self) -> float:
        c = self.config
        return max(c.epsilon_end, c.epsilon_start * c.epsilon_decay ** self.updates) if self.training else 0.0

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        """Clear episode memory while preserving the Q-table and learning progress."""
        self.training = training
        # The environment already supplies separate seeds for all roles and agents.
        self.rng.seed(seed)
        self.recent_rewards.clear()
        self.previous_market_price = None
        self.decisions = self.known_state_decisions = 0
        self._episode_checkpoint = (Path(self.checkpoint_path.format(agent_id=agent_id, seed=seed))
                                    if self.checkpoint_path is not None else None)
        if training:
            seeds = self.metadata.setdefault('training_reset_seeds', [])
            if seed not in seeds:
                seeds.append(seed)
        # Never reset Q values or the exploration schedule between episodes.

    def _price_bucket(self, price: float | None, bins: int) -> int:
        if price is None:
            return -1
        c = self.config
        position = (price - c.min_price) / (c.max_price - c.min_price)
        return max(0, min(bins - 1, int(position * bins)))

    def encode_state(self, observation: AgentObservation) -> tuple[int, ...]:
        c = self.config
        base = (self._price_bucket(observation.market.average_price, c.market_bins),
                int(bool(observation.neighbors)), int(observation.restricted),
                int(observation.offer is not None))
        if c.state_mode == 'basic':
            return base
        balance_ratio = observation.balance / c.balance_reference
        reserve = 0 if balance_ratio < 0 else 1 if balance_ratio < 0.5 else 2
        current, previous = observation.market.average_price, self.previous_market_price
        difference = 0.0 if current is None or previous is None else current - previous
        tolerance = (c.max_price - c.min_price) / c.market_bins / 2
        trend = -1 if difference < -tolerance else 1 if difference > tolerance else 0
        recent = mean(self.recent_rewards) / c.reward_scale if self.recent_rewards else 0.0
        outcome = -1 if recent < 0 else 1 if recent > 1 else 0
        if c.state_mode == 'context':
            # Same current information as history, without either history feature.
            trend = outcome = 0
        offer_margin = offer_weight = -1
        if observation.offer is not None:
            margin = observation.offer.recommended_price - observation.cost
            offer_margin = sum(margin >= threshold for threshold in (0, 5, 15, 30))
            relative_weight = observation.offer.payout_weight * len(observation.offer.members)
            offer_weight = 0 if relative_weight < 0.8 else 1 if relative_weight <= 1.2 else 2
        return base + (self._price_bucket(observation.cost, 4), reserve, trend, outcome,
                       offer_margin, offer_weight)

    def available_actions(self, observation: AgentObservation) -> dict[int, AgentAction]:
        protect_margin = observation.restricted or not self.config.allow_loss_leading
        floor = max(self.config.min_price, observation.cost) if protect_margin else self.config.min_price
        skip = AgentAction(price=min(self.config.max_price, floor), participate=False)
        if not observation.active:
            return {self.abstain_index: skip}
        actions = {index: AgentAction(price=price) for index, price in enumerate(self.prices) if price >= floor}
        actions[self.abstain_index] = skip
        offer = observation.offer
        if offer is not None and not observation.restricted and floor <= offer.recommended_price <= self.config.max_price:
            actions[self.accept_index] = AgentAction(price=offer.recommended_price, coalition_choice=offer.plan_id)
        return actions

    def _values(self, state: tuple[int, ...]) -> list[float]:
        return self.q.get(state, [0.0] * self.n_actions)

    def act(self, observation: AgentObservation) -> AgentAction:
        """Choose a legal action; evaluation uses a frozen table and seeded tie-breaking."""
        available = self.available_actions(observation)
        if not observation.active:
            return available[self.abstain_index]
        state = self.encode_state(observation)
        self.decisions += 1
        self.known_state_decisions += int(state in self.q)
        if self.training and self.rng.random() < self.epsilon:
            index = self.rng.choice(list(available))
        elif state not in self.q and not self.training:
            # Even an agent allowed to learn loss-leading prices has a non-loss fallback.
            prices = [i for i in available if i < len(self.prices) and self.prices[i] >= observation.cost]
            index = min(prices, key=lambda i: abs(self.prices[i] - observation.cost - 5)) if prices else self.abstain_index
        else:
            values = self._values(state)
            best = max(values[index] for index in available)
            index = self.rng.choice([index for index in available if values[index] == best])
        return available[index]

    def _action_index(self, observation: AgentObservation, action: AgentAction) -> int:
        for index, candidate in self.available_actions(observation).items():
            if (candidate.participate == action.participate and candidate.coalition_choice == action.coalition_choice
                    and candidate.link_action == action.link_action
                    and math.isclose(candidate.price, action.price, abs_tol=1e-8, rel_tol=0)):
                return index
        # Do not credit a nearby grid action for a different sanitized action.
        raise ValueError('executed action does not match an available agent action; check market bounds')

    def observe_transition(
        self, *, observation: AgentObservation, action: AgentAction, reward: float,
        next_observation: AgentObservation | None, done: bool,
    ) -> None:
        """Update memory and, in training mode, learn from the actual net reward."""
        if not observation.active:
            return
        if not math.isfinite(reward):
            raise ValueError('reward must be finite')
        state = self.encode_state(observation)
        self.previous_market_price = observation.market.average_price
        self.recent_rewards.append(reward)
        if not self.training:
            return
        terminal = done or next_observation is None or not next_observation.active
        target = reward / self.config.reward_scale
        if not terminal:
            values = self._values(self.encode_state(next_observation))
            target += self.config.gamma * max(values[index] for index in self.available_actions(next_observation))
        index = self._action_index(observation, action)
        values = self.q.setdefault(state, [0.0] * self.n_actions)
        values[index] += self.config.learning_rate * (target - values[index])
        self.updates += 1
        if terminal and self._episode_checkpoint is not None:
            self.save(self._episode_checkpoint)

    def save(self, path: str | Path) -> None:
        """Atomically write a portable JSON model without serializing executable code."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format_version': self.FORMAT_VERSION, 'contracts_version': CONTRACTS_VERSION,
            'action_layout': self.ACTION_LAYOUT, 'market': self.market,
            'config': asdict(self.config), 'updates': self.updates, 'metadata': self.metadata,
            'q_table': [{'state': list(state), 'values': values} for state, values in sorted(self.q.items())],
        }
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> TverdokhlebAgent:
        """Restore a model in evaluation mode, validating its contract and table."""
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('format_version') != cls.FORMAT_VERSION:
            raise ValueError('unsupported model format: MVP/v2 checkpoints require retraining for TverdokhlebAgent v3')
        if payload.get('contracts_version') != CONTRACTS_VERSION or payload.get('action_layout') != cls.ACTION_LAYOUT:
            raise ValueError('incompatible model contract or action layout')
        agent = cls(config=payload['config'], market=payload['market'])
        if type(payload['updates']) is not int or payload['updates'] < 0:
            raise ValueError('invalid update counter')
        agent.updates = payload['updates']
        if not isinstance(payload.get('metadata', {}), dict):
            raise ValueError('metadata must be an object')
        agent.metadata = payload.get('metadata', {})
        dimensions = 4 if agent.config.state_mode == 'basic' else 10
        for row in payload['q_table']:
            state, values = tuple(row['state']), row['values']
            if len(state) != dimensions or any(type(x) is not int for x in state):
                raise ValueError('invalid state in model')
            if len(values) != agent.n_actions or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise ValueError('invalid Q values in model')
            if state in agent.q:
                raise ValueError('duplicate state in model')
            agent.q[state] = list(values)
        return agent
