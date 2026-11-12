"""Market, hidden network, coalition mechanics and an arbiter integration boundary."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from .coalitions import allocate, clear_market, components


@dataclass(frozen=True)
class MarketConfig:
    costs: tuple = (1.0, 2.0, 3.0, 4.0)
    balances: tuple = (80.0, 100.0, 120.0, 140.0)
    prices: tuple = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
    horizon: int = 100
    demand_a: float = 100.0
    demand_b: float = 5.0
    topology: str = "complete"
    dynamic_network: bool = True
    allocation: str = "equal"
    base_link_cost: float = 0.2
    credit_limit: float = 0.0
    bankruptcy_patience: int = 3
    demand_model: str = "assignment"
    logit_value: float = 8.0
    logit_temperature: float = 1.5
    harm_baseline: str = "auto"

    def __post_init__(self):
        if len(self.costs) != 4 or len(self.balances) != 4:
            raise ValueError("This minimal version has exactly four agents")
        if not np.isfinite((*self.costs, *self.balances, *self.prices)).all():
            raise ValueError("Costs, balances and prices must be finite")
        if min(self.costs) < 0 or min(self.balances) < 0:
            raise ValueError("Initial costs and balances must be nonnegative")
        if not self.prices or min(self.prices) <= 0 or tuple(sorted(set(self.prices))) != tuple(self.prices):
            raise ValueError("Prices must be positive, distinct and increasing")
        if max(self.costs) > max(self.prices):
            raise ValueError("Each agent needs a price at least as high as its cost")
        numeric = (self.demand_a, self.demand_b, self.base_link_cost, self.credit_limit)
        if not np.isfinite(numeric).all() or min(numeric) < 0 or self.demand_b <= 0:
            raise ValueError("Invalid demand or financial parameters")
        if self.horizon < 1 or self.bankruptcy_patience < 1:
            raise ValueError("Horizon and bankruptcy patience must be positive")
        if self.topology not in {"complete", "star", "path", "inner", "random"}:
            raise ValueError("Unknown topology")
        if self.allocation not in {"equal", "shapley", "myerson", "owen"}:
            raise ValueError("Unknown allocation")
        if self.demand_model not in {"assignment", "logit"}:
            raise ValueError("Unknown demand model")
        if not np.isfinite((self.logit_value, self.logit_temperature)).all() or self.logit_temperature <= 0:
            raise ValueError("Logit utility must be finite and temperature positive")
        if self.harm_baseline not in {"auto", "proposals", "nash"}:
            raise ValueError("Unknown harm benchmark")

    @property
    def effective_harm_baseline(self):
        return ("nash" if self.demand_model == "logit" else "proposals") if self.harm_baseline == "auto" else self.harm_baseline

    @property
    def welfare_bound(self):
        """Public upper bound: zero costs and, for logit, free products."""
        if self.demand_model == "logit":
            return float(self.demand_a * self.logit_temperature *
                         np.logaddexp(0., np.log(4) + self.logit_value / self.logit_temperature))
        return self.demand_a**2 / (2 * self.demand_b)


@dataclass(frozen=True)
class AgentAction:
    price: int  # index in config.prices
    coalition: int = -1  # -1: solo; 0..3: public negotiation label


@dataclass(frozen=True)
class PublicObservation:
    """The ONLY observation passed to an external arbiter. No private objects."""
    step: int
    prices: tuple  # None for exited agents
    total_quantity: float
    mean_price: float
    social_welfare: float
    price_change: float
    welfare_change: float


@dataclass(frozen=True)
class ArbiterAction:
    """An external policy can return one of the three actions from the brief."""
    kind: Literal["observe", "link_cost", "sanction"] = "observe"
    amount: float = 0.0
    target: int | None = None


@dataclass(frozen=True)
class LocalObservation:
    cost: float
    balance: float
    neighbors: tuple
    coalition: tuple
    last_profit: float
    public_prices: tuple
    link_costs: tuple
    trust: tuple
    active: bool
    last_fine: float = 0.0


class MarketEnv:
    n_agents = 4

    def __init__(self, config=None):
        self.config = config or MarketConfig()
        self.n_actions = len(self.config.prices) * (self.n_agents + 1)
        self._pending = None

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.costs = np.array(self.config.costs, dtype=float)
        self.balances = np.array(self.config.balances, dtype=float)
        self.active = np.ones(4, dtype=bool)
        self.debt_steps = np.zeros(4, dtype=int)
        self.last_profit = np.zeros(4)
        self.last_fine = np.zeros(4)
        self.previous_coalitions = [(i,) for i in range(4)]
        # Initial similarity: location / education / social features. Quadratic
        # polynomial with random seeded coefficients defines link probabilities.
        attributes = self.rng.uniform(size=(4, 3))
        similarity = 1 - np.abs(attributes[:, None] - attributes[None, :])
        coefficients = self.rng.normal(0, 0.6, size=(2, 3))
        self._affinity = -0.3 + (similarity * coefficients[0] + similarity**2 * coefficients[1]).sum(axis=2)
        self._history = np.zeros((4, 4))
        self._utility = np.zeros((4, 4))
        self._expected_gain = np.zeros((4, 4))
        self.link_costs = np.full((4, 4), self.config.base_link_cost)
        np.fill_diagonal(self.link_costs, 0)
        self.graph = np.zeros((4, 4), dtype=bool)
        for i in range(4):
            for j in range(i + 1, 4):
                topology = self.config.topology
                exists = {"complete": True, "star": i == 0, "path": j == i + 1,
                          "inner": (i, j) in {(0, 1), (2, 3)}}.get(topology)
                if topology == "random":
                    exists = self.rng.random() < self._sigmoid(self._affinity[i, j])
                self.graph[i, j] = self.graph[j, i] = exists
        self.public = PublicObservation(0, (None,) * 4, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._pending = None
        self.done = False
        return self.observations()

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def decode_action(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.n_actions:
            raise ValueError("Invalid action index")
        return AgentAction(index // 5, index % 5 - 1)

    def action_mask(self, i):
        mask = np.ones(self.n_actions, dtype=bool)
        if not self.active[i]:
            mask[:] = False
            mask[0] = True  # ignored by clearing; no other actions after exit
        elif self.balances[i] <= -self.config.credit_limit:
            # An insolvent agent must trade alone at a non-loss-making price.
            mask = np.array([a.coalition == -1 and self.config.prices[a.price] >= self.costs[i]
                             for a in map(self.decode_action, range(self.n_actions))])
        return mask

    def observations(self):
        return tuple(LocalObservation(
            cost=float(self.costs[i]), balance=float(self.balances[i]),
            neighbors=tuple(np.flatnonzero(self.graph[i]).tolist()),
            coalition=next(c for c in self.previous_coalitions if i in c),
            last_profit=float(self.last_profit[i]), public_prices=self.public.prices,
            link_costs=tuple(self.link_costs[i]), trust=tuple(self._history[i]),
            active=bool(self.active[i]),
            last_fine=float(self.last_fine[i]),
        ) for i in range(4))

    def _clear(self, prices):
        return clear_market(prices, self.costs, self.active, self.config.demand_a, self.config.demand_b,
                            model=self.config.demand_model, logit_value=self.config.logit_value,
                            logit_temperature=self.config.logit_temperature)

    def form_coalitions(self, actions):
        """Private coalition-level input, unavailable to the arbiter."""
        if len(actions) != 4:
            raise ValueError("Exactly four actions are required")
        actions = tuple(self.decode_action(a) for a in actions)
        for i, action in enumerate(actions):
            index = action.price * 5 + action.coalition + 1
            if not self.action_mask(i)[index]:
                raise ValueError(f"Agent {i}: action is unavailable because of its balance")
        groups = []
        for label in range(-1, 4):
            ids = [i for i, a in enumerate(actions) if self.active[i] and a.coalition == label]
            groups.extend([(i,) for i in ids] if label == -1 else components(ids, self.graph))
        return tuple(groups)

    def market_step(self, actions, coalition_prices=None):
        """Phase 1: network -> coalitions -> learned joint prices -> public data.

        coalition_prices maps frozensets of members to price-grid indices. The
        training runner supplies these from its coalition Q-learning policy.
        """
        if self.done or self._pending is not None:
            raise RuntimeError("Reset a finished episode / settle the pending round first")
        groups = self.form_coalitions(actions)
        actions = tuple(self.decode_action(a) for a in actions)
        coalition_prices = coalition_prices or {}
        for group, price in coalition_prices.items():
            if not group or not set(group) <= set(range(4)) or not isinstance(price, (int, np.integer)) or not 0 <= price < len(self.config.prices):
                raise ValueError("Invalid coalition price")
        if any(len(c) > 1 and frozenset(c) not in coalition_prices for c in groups):
            raise ValueError("Each coalition needs a price chosen by its collective policy")
        raw_prices = np.array([self.config.prices[a.price] for a in actions], dtype=float)

        def joint_price(group):
            chosen = coalition_prices.get(frozenset(group))
            if chosen is None:
                # A counterfactual subgroup can execute its parent contract.
                parent = next(c for c in groups if set(group) <= set(c))
                chosen = coalition_prices[frozenset(parent)]
            return self.config.prices[chosen]

        prices = raw_prices.copy()
        for group in groups:
            if len(group) > 1:
                prices[list(group)] = joint_price(group)
        quantities, profits, mean_price, cs = self._clear(prices)
        distributed = profits.copy()
        gains = np.zeros(4)
        counterfactuals = []
        nash = None
        if self.config.effective_harm_baseline == "nash":
            from .equilibrium import competitive_prices

            nash = competitive_prices(self.config, tuple(np.flatnonzero(self.active).tolist()))
        for group in groups:
            if len(group) < 2:
                continue
            # Outsiders retain their actual prices in every counterfactual.
            baseline_prices = prices.copy()
            baseline_prices[list(group)] = raw_prices[list(group)]
            _, baseline, baseline_price, baseline_surplus = self._clear(baseline_prices)
            if nash is None:
                benchmark_price, benchmark_welfare = baseline_price, float(baseline_surplus + baseline.sum())
            else:
                benchmark_prices = prices.copy()
                benchmark_prices[list(group)] = np.array(nash)[list(group)]
                _, competitive_profit, benchmark_price, competitive_surplus = self._clear(benchmark_prices)
                benchmark_welfare = float(competitive_surplus + competitive_profit.sum())
            counterfactuals.append({"members": tuple(group), "mean_price": benchmark_price,
                                    "social_welfare": benchmark_welfare,
                                    "benchmark": self.config.effective_harm_baseline})

            @lru_cache(None)
            def value(subset):
                if len(subset) < 2:
                    return 0.0
                ids = sorted(subset)
                counterfactual = baseline_prices.copy()
                counterfactual[ids] = joint_price(subset)
                return float((self._clear(counterfactual)[1] - baseline)[ids].sum())

            shares = allocate(group, value, self.config.allocation, self.graph, self.previous_coalitions)
            for i in group:
                distributed[i] = baseline[i] + shares[i]
                gains[i] = shares[i]
        sw = float(cs + profits.sum())
        observation = PublicObservation(
            step=self.t + 1,
            prices=tuple(float(prices[i]) if self.active[i] else None for i in range(4)),
            total_quantity=float(quantities.sum()), mean_price=mean_price, social_welfare=sw,
            price_change=mean_price - self.public.mean_price if self.t else 0.0,
            welfare_change=sw - self.public.social_welfare if self.t else 0.0,
        )
        self._pending = (observation, groups, profits, distributed, gains, cs, tuple(counterfactuals))
        return observation

    def settle(self, action=None):
        """Phase 2: external arbiter action -> rewards/balances -> next hidden graph."""
        if self._pending is None:
            raise RuntimeError("Call market_step before settle")
        action = action or ArbiterAction()
        if action.kind not in {"observe", "link_cost", "sanction"}:
            raise ValueError("Unknown arbiter action")
        if not np.isfinite(action.amount) or action.amount < 0:
            raise ValueError("The intervention amount must be finite and nonnegative")
        if action.kind == "sanction":
            if not isinstance(action.target, (int, np.integer)) or not 0 <= action.target < 4:
                raise ValueError("A sanction needs a public agent ID 0..3")
        elif action.target is not None:
            raise ValueError("Only a sanction accepts a target")
        if action.kind == "observe" and action.amount != 0:
            raise ValueError("Observation has no intervention amount")
        public, groups, profits, distributed, gains, cs, counterfactuals = self._pending
        fines = np.zeros(4)
        if action.kind == "sanction" and self.active[action.target]:
            fines[action.target] = action.amount
        elif action.kind == "link_cost":
            # Uniform public intervention; affects future formation, never
            # directly deletes an existing link or reveals a hidden pair.
            self.link_costs += action.amount * (1 - np.eye(4))
        rewards = distributed - fines
        self.balances += rewards
        link_expenses = self._evolve_network(groups, gains, rewards)
        rewards -= link_expenses
        self.balances -= link_expenses
        self.debt_steps = np.where(self.balances < 0, self.debt_steps + 1, 0)
        self.active &= self.debt_steps < self.config.bankruptcy_patience
        self.graph &= self.active[:, None] & self.active[None, :]
        self.t += 1
        self.done = self.t >= self.config.horizon or not self.active.any()
        self.public = public
        self.previous_coalitions = list(groups) + [(i,) for i in range(4) if not any(i in c for c in groups)]
        self.last_profit = rewards.copy()
        self.last_fine = fines.copy()
        self._pending = None
        # Private evaluation data stays here. Never pass this dict to an arbiter.
        info = {
            "coalitions": tuple(groups), "graph": self.graph.copy(),
            "operating_profits": profits.copy(), "coalition_gains": gains.copy(),
            "fines": fines, "link_expenses": link_expenses,
            "consumer_surplus": cs, "net_welfare": public.social_welfare - float(link_expenses.sum()),
            "counterfactuals": counterfactuals,
        }
        return self.observations(), rewards, self.done, info

    def _evolve_network(self, groups, gains, rewards):
        together = np.zeros((4, 4))
        for group in groups:
            if len(group) > 1:
                together[np.ix_(group, group)] = 1
        self._history = 0.8 * self._history + 0.2 * together
        pair_rewards = np.tanh((rewards[:, None] + rewards[None, :]) / 100)
        pair_gains = np.tanh((gains[:, None] + gains[None, :]) / 100)
        self._utility = 0.8 * self._utility + 0.2 * together * pair_rewards
        self._expected_gain = 0.8 * self._expected_gain + 0.2 * together * pair_gains
        expenses = np.zeros(4)
        if not self.config.dynamic_network:
            return expenses
        for i in range(4):
            for j in range(i + 1, 4):
                if not (self.active[i] and self.active[j]):
                    continue
                existing = self.graph[i, j]
                cost = 0.0 if existing else self.link_costs[i, j]
                logit = (self._affinity[i, j] + 1.3 * existing + self._history[i, j]
                         + self._utility[i, j] + self._expected_gain[i, j] - cost)
                exists = self.rng.random() < self._sigmoid(logit)
                if exists and not existing:
                    fee = cost / 2
                    affordable = min(self.balances[i] - expenses[i], self.balances[j] - expenses[j]) >= fee
                    exists = affordable
                    if exists:
                        expenses[i] += fee
                        expenses[j] += fee
                self.graph[i, j] = self.graph[j, i] = exists
        return expenses
