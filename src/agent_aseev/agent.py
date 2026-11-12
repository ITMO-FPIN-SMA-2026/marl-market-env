from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT: str = str(Path(__file__).with_name("georgii_q.json"))

import numpy as np

from social_market_env.contracts import AgentAction, AgentObservation, MarketAgentAPI

State = tuple[int, int, int, int]

MARKUPS: tuple[float, ...] = (1.05, 1.15, 1.25, 1.35, 1.5, 1.7, 2.0, 2.5)
ACCEPT_OFFER: int = len(MARKUPS)
N_ACTIONS: int = len(MARKUPS) + 1


class GeorgiiAgent(MarketAgentAPI):
    def __init__(
        self,
        *,
        market: dict[str, Any] | None = None,
        checkpoint: str | None = DEFAULT_CHECKPOINT,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99,
        reward_scale: float = 100.0,
    ) -> None:
        self.checkpoint = checkpoint
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_scale = reward_scale
        self.q: dict[State, list[float]] = {}
        self.training = False
        self.rng = np.random.default_rng(0)
        self._last: tuple[State, int] | None = None
        if checkpoint and Path(checkpoint).exists():
            self.load(checkpoint)

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        self.agent_id = agent_id
        self.training = training
        self.rng = np.random.default_rng(seed)
        self._last = None

    def act(self, observation: AgentObservation) -> AgentAction:
        if not observation.active:
            return AgentAction(price=observation.cost, participate=False)

        state = self._encode(observation)
        allowed = self._allowed(observation)
        if self.training and self.rng.random() < self.epsilon:
            idx = int(self.rng.choice(allowed))
        else:
            values = self._values(state)
            idx = max(allowed, key=lambda a: values[a])
        self._last = (state, idx)

        if idx == ACCEPT_OFFER:
            offer = observation.offer
            return AgentAction(price=offer.recommended_price, coalition_choice=offer.plan_id)
        return AgentAction(price=observation.cost * MARKUPS[idx])

    def observe_transition(self, *, observation, action, reward, next_observation, done) -> None:
        if not self.training or self._last is None:
            return
        state, idx = self._last
        target = reward / self.reward_scale
        if not done and next_observation is not None and next_observation.active:
            nxt = self._values(self._encode(next_observation))
            target += self.gamma * max(nxt[a] for a in self._allowed(next_observation))
        values = self._values(state)
        values[idx] += self.alpha * (target - values[idx])
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _values(self, state: State) -> list[float]:
        if state not in self.q:
            self.q[state] = [0.0] * N_ACTIONS
        return self.q[state]

    @staticmethod
    def _allowed(obs: AgentObservation) -> list[int]:
        actions = list(range(len(MARKUPS)))
        if obs.offer is not None and not obs.restricted:
            actions.append(ACCEPT_OFFER)
        return actions

    @staticmethod
    def _encode(obs: AgentObservation) -> State:
        avg = obs.market.average_price
        if obs.last_price is None or not avg:
            rel = 1
        else:
            ratio = obs.last_price / avg
            rel = 0 if ratio < 0.97 else (2 if ratio > 1.03 else 1)
        profit_ok = 1 if obs.last_profit > 0 else 0
        restricted = 1 if obs.restricted else 0
        has_offer = 1 if obs.offer is not None else 0
        return (rel, profit_ok, restricted, has_offer)

    def save(self, path: str) -> None:
        data = {"epsilon": self.epsilon, "q": [[list(s), v] for s, v in self.q.items()]}
        Path(path).write_text(json.dumps(data))

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        self.q = {tuple(s): list(v) for s, v in data["q"]}
        self.epsilon = data.get("epsilon", self.epsilon_min)

    def greedy_policy(self) -> dict[State, str]:
        names = [f"x{m}" for m in MARKUPS] + ["оффер"]
        return {s: names[int(np.argmax(v))] for s, v in sorted(self.q.items())}
