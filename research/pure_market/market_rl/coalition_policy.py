"""Discounted collective control, separate from settlement of today's cash."""

from itertools import combinations

import numpy as np


class CoalitionPolicy:
    """One joint Q table per member set, choosing a common price.

    Membership decisions belong to market agents. Between two occurrences of
    the same coalition, its members can trade alone or in other coalitions.
    SMDP Q-learning accumulates ALL their intervening profits, with discounting,
    rather than pretending that recurrence happens at the next market step.
    """

    def __init__(self, n_prices, seed=0, gamma=0.95, learning_rate=0.1):
        self.n_prices = n_prices
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.q = {}
        self.pending = {}
        self.updates = 0

    @staticmethod
    def state(env, group):
        mask = sum(1 << i for i in group)
        edges = sum(int(env.graph[i, j]) << k for k, (i, j) in enumerate(combinations(range(4), 2))
                    if i in group or j in group)
        price_bin = int(np.argmin(np.abs(np.array(env.config.prices) - env.public.mean_price)))
        balance_bin = int(np.digitize(min(env.balances[list(group)]), [0, 100, 300]))
        remaining_bin = min(2, (env.config.horizon - env.t - 1) // 10)
        fine_mask = sum(int(env.last_fine[i] > 0) << i for i in group)
        return (mask, edges, price_bin, balance_bin, remaining_bin, fine_mask)

    def reset_episode(self):
        self.pending = {}

    def _update(self, pending, continuation=0.0):
        state, action, total, discount = pending
        row = self.q.setdefault(state, np.zeros(self.n_prices))
        row[action] += self.learning_rate * (total + discount * continuation - row[action])
        self.updates += 1

    def plan(self, env, actions, training=True, epsilon=0.2):
        actual = {frozenset(c) for c in env.form_coalitions(actions) if len(c) > 1}
        result = {}
        for size in range(2, 5):
            for group in combinations(range(4), size):
                key = frozenset(group)
                state = self.state(env, group)
                if training and key in actual and key in self.pending:
                    self._update(self.pending.pop(key), float(self.q.get(state, np.zeros(self.n_prices)).max()))
                values = self.q.get(state, np.zeros(self.n_prices))
                if key in actual and training and self.rng.random() < epsilon:
                    price = int(self.rng.integers(self.n_prices))
                else:
                    greedy = np.flatnonzero(values == values.max())
                    price = int(self.rng.choice(greedy))
                result[key] = price
                if training and key in actual:
                    self.pending[key] = [state, price, 0.0, 1.0]
        return result

    def observe(self, rewards, done):
        for group, pending in self.pending.items():
            pending[2] += pending[3] * float(rewards[list(group)].sum()) / 100.0
            pending[3] *= self.gamma
        if done:
            for pending in self.pending.values():
                self._update(pending)
            self.pending = {}

    def values(self, env, solo_agents):
        """Private V_C and Delta_C estimates; sanctions are already in rewards.

        Q maximizes discounted *joint* returns, whereas the solo reference
        learns in a separate market without coalitions. No future money is paid.
        """
        from .agents import encode_state

        from dataclasses import replace

        solo = [a.state_value(encode_state(replace(o, coalition=(i,)), env.config.prices), env.action_mask(i)[::5])
                for i, (a, o) in enumerate(zip(solo_agents, env.observations()))]
        result = {}
        for size in range(2, 5):
            for group in combinations(range(4), size):
                state = self.state(env, group)
                row = self.q.get(state)
                joint = float(row.max() * 100) if row is not None else None
                baseline = float(sum(solo[i] for i in group))
                result["-".join(map(str, group))] = {
                    "joint_value": joint, "solo_value": baseline,
                    "additional_value": joint - baseline if joint is not None else None,
                }
        return result

    def to_dict(self):
        return {"state_version": 2, "n_prices": self.n_prices, "gamma": self.gamma, "learning_rate": self.learning_rate,
                "q": [[list(s), v.tolist()] for s, v in self.q.items()],
                "updates": self.updates, "rng": self.rng.bit_generator.state}

    @classmethod
    def from_dict(cls, data):
        policy = cls(data["n_prices"], gamma=data["gamma"], learning_rate=data["learning_rate"])
        policy.q = {(tuple(s) + (0,) if data.get("state_version", 1) == 1 and len(s) == 5 else tuple(s)):
                    np.array(v, dtype=float) for s, v in data["q"]}
        policy.updates = data["updates"]
        policy.rng.bit_generator.state = data["rng"]
        return policy
