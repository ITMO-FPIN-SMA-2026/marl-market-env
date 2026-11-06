from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from social_market_env.contracts import AgentAction, AgentObservation, MarketAgentAPI

State = Tuple[int, int, int, int]


class QLearningMarketAgent(MarketAgentAPI):
    def __init__(
        self,
        markups: Optional[List[float]] = None,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        reward_scale: float = 100.0,
    ) -> None:
        self.markups = markups or [1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_scale = reward_scale

        self.q: Dict[State, List[float]] = defaultdict(lambda: [0.0] * len(self.markups))
        self.rng = random.Random()
        self.agent_id: Optional[int] = None
        self.training = True
        self._last_action_index: Dict[int, int] = {}

    def reset(self, agent_id: int, seed: Optional[int] = None) -> None:
        self.agent_id = agent_id
        self._last_action_index.clear()
        if seed is not None and self.agent_id is not None:
            self.rng.seed(seed * 1000 + agent_id + len(self.q))

    def act(self, observation: AgentObservation) -> AgentAction:
        if not observation.active:
            return AgentAction(price=observation.cost, participate=False)

        state = self._encode(observation)
        if self.training and self.rng.random() < self.epsilon:
            idx = self.rng.randrange(len(self.markups))
        else:
            values = self.q[state]
            idx = max(range(len(values)), key=values.__getitem__)

        self._last_action_index[observation.step] = idx
        price = observation.cost * self.markups[idx]
        return AgentAction(price=price, participate=True)

    def observe_transition(self, observation, action, reward, next_observation, done) -> None:
        if not self.training or not observation.active:
            return
        idx = self._last_action_index.get(observation.step)
        if idx is None:
            return

        s = self._encode(observation)
        r = reward / self.reward_scale
        if done or next_observation is None or not next_observation.active:
            target = r
        else:
            target = r + self.gamma * max(self.q[self._encode(next_observation)])
        self.q[s][idx] += self.alpha * (target - self.q[s][idx])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _encode(self, obs: AgentObservation) -> State:
        if obs.last_price is None or not obs.market_average_price:
            rel = 1
        else:
            ratio = obs.last_price / obs.market_average_price
            rel = 0 if ratio < 0.9 else (2 if ratio > 1.1 else 1)
        profit_ok = 1 if obs.last_profit > 0 else 0
        rich = 1 if obs.balance > 200 else 0
        has_nbrs = 1 if obs.local_neighbors else 0
        return (rel, profit_ok, rich, has_nbrs)

    def greedy_policy(self) -> Dict[State, float]:
        return {s: self.markups[max(range(len(v)), key=v.__getitem__)] for s, v in self.q.items()}
