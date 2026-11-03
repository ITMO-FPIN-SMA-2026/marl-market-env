"""Prevent harmful coordination using public prices and bounded link-cost changes.

The original five-policy experiment is preserved in research/arbiter_comparison.
This policy needs no neural-network, plotting or research dependencies.
"""

from collections import deque
from dataclasses import dataclass
from itertools import combinations
import math
from statistics import fmean, pstdev
from typing import Any

from social_market_env.config import MarketConfig
from social_market_env.contracts import ArbiterAction, ArbiterActionKind, ArbiterAPI, ArbiterObservation, Pair


@dataclass(frozen=True)
class Decision:
    """Public explanation of the last decision; does not represent a hidden graph."""

    reason: str
    welfare_gap: float = 0.0
    competitive_average_price: float | None = None
    candidate_pairs: int = 0
    changed_pairs: int = 0


class PreventiveArbiter(ArbiterAPI):
    """Raise costs of suspected pairs after a persistent consumer-surplus deficit.

    In a static network, a healthy market or after exhausting the cost budget,
    observe. No sanctions, direct edge deletion or access to private state.

    ``market``, ``network_mode`` and ``link_cost_decay`` are public configuration.
    The runner injects market automatically; pass mode/decay explicitly in kwargs.
    The defaults reproduce the prototype's prevention parameters. The old fine
    fallback is intentionally replaced with observation.
    """

    def __init__(
        self,
        *,
        market: dict[str, Any] | None = None,
        network_mode: str = 'evolving',
        history_window: int = 20,
        welfare_gap_threshold: float = 0.08,
        patience: int = 2,
        price_margin: float = 2.0,
        pair_budget: int = 24,
        link_cost_delta: float = 5.0,
        link_cost_cap: float = 10.0,
        link_cost_decay: float = 0.0,
    ) -> None:
        if network_mode not in ('static', 'evolving'):
            raise ValueError('network_mode must be static or evolving')
        for name, value in [('history_window', history_window), ('patience', patience), ('pair_budget', pair_budget)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in [('welfare_gap_threshold', welfare_gap_threshold), ('price_margin', price_margin),
                            ('link_cost_delta', link_cost_delta), ('link_cost_cap', link_cost_cap),
                            ('link_cost_decay', link_cost_decay)]:
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a finite non-negative number')
        if welfare_gap_threshold >= 1:
            raise ValueError('welfare_gap_threshold must be less than 1')
        if link_cost_decay > 1:
            raise ValueError('link_cost_decay must be in [0, 1]')
        if link_cost_delta == 0 or link_cost_cap == 0:
            raise ValueError('link_cost_delta and link_cost_cap must be positive')
        self.market = MarketConfig(**(market or {}))
        if not math.isfinite(self.market.demand_a) or self.market.demand_a <= 0:
            raise ValueError('market.demand_a must be finite and positive')
        if not math.isfinite(self.market.demand_b) or self.market.demand_b <= 0:
            raise ValueError('market.demand_b must be finite and positive')
        if not all(math.isfinite(x) for x in (self.market.p_min, self.market.p_max)):
            raise ValueError('market price bounds must be finite')
        if not 0 < self.market.p_min < self.market.p_max:
            raise ValueError('market price bounds must satisfy 0 < p_min < p_max')
        self.network_mode = network_mode
        self.history_window = history_window
        self.welfare_gap_threshold = welfare_gap_threshold
        self.patience = patience
        self.price_margin = price_margin
        self.pair_budget = pair_budget
        self.link_cost_delta = link_cost_delta
        self.link_cost_cap = link_cost_cap
        self.link_cost_decay = link_cost_decay
        self.reset(seed=0, training=False)

    def reset(self, *, seed: int, training: bool) -> None:
        """Clear all episode memory. This deterministic policy does not fit weights."""
        self._prices: deque[dict[int, float]] = deque(maxlen=self.history_window)
        self._applied_costs: dict[Pair, float] = {}
        self._last_step: int | None = None
        self._pending = ArbiterAction()
        self._persistence = 0
        self.last_decision = Decision('reset')

    def _reconcile(self, observation: ArbiterObservation) -> None:
        """Track actual sanitised actions using public last_action, not requested deltas.

        At act(t), costs reflect the end of t-1. The environment will decay them
        once more before applying the new action, which is accounted for below.
        """
        if self._last_step is not None:
            if observation.step != self._last_step + 1:
                raise ValueError('Observations must be consecutive; reset before a new episode')
            retain = 1.0 - self.link_cost_decay
            self._applied_costs = {pair: cost * retain for pair, cost in self._applied_costs.items()
                                   if cost * retain > 0}
            applied = observation.last_action if observation.last_action is not None else self._pending
            if applied.kind == ArbiterActionKind.change_link_cost:
                for pair, delta in applied.link_cost_changes.items():
                    canonical = tuple(sorted(pair))
                    self._applied_costs[canonical] = max(0.0, self._applied_costs.get(canonical, 0.0) + delta)
        self._last_step = observation.step

    def act(self, observation: ArbiterObservation) -> ArbiterAction:
        self._reconcile(observation)
        active = set(observation.active_agents)
        prices = {i: price for i, price in observation.prices.items() if i in active}
        target = observation.target_social_welfare
        gap = (target - observation.social_welfare) / target if target > 0 else 0.0
        scale = self.market.p_max - self.market.p_min
        competitive = ((self.market.demand_a - math.sqrt(2.0 * self.market.demand_b * target))
                       / self.market.demand_b) if target > 0 else None
        harmful_signal = target > 0 and bool(prices) and gap > self.welfare_gap_threshold
        self._persistence = self._persistence + 1 if harmful_signal else 0

        # Scores use prior quotes only. Keeping price snapshots avoids retaining
        # complete observation/fines histories and never requires private data.
        risks = {}
        for i, price in prices.items():
            old = [frame[i] for frame in self._prices if i in frame]
            level = max(0.0, (price - competitive) / scale) if competitive is not None else 0.0
            trend = max(0.0, (price - fmean(old)) / scale) if old else 0.0
            risks[i] = level + 0.25 * trend
        candidates = [i for i in sorted(prices)
                      if competitive is not None and prices[i] > competitive + self.price_margin]
        pairs = list(combinations(candidates, 2))
        retain = 1.0 - self.link_cost_decay
        effective_costs = {pair: cost * retain for pair, cost in self._applied_costs.items()}

        def pair_key(pair: Pair) -> tuple[float, float, Pair]:
            i, j = pair
            diffs = [frame[i] - frame[j] for frame in self._prices if i in frame and j in frame]
            instability = pstdev(diffs) / scale if len(diffs) >= 2 else 0.0
            score = min(risks[i], risks[j]) - 0.25 * abs(prices[i] - prices[j]) / scale - instability
            return effective_costs.get(pair, 0.0), -score, pair

        pairs.sort(key=pair_key)
        self._prices.append(dict(prices))
        if self.network_mode == 'static':
            reason = 'static_network'
        elif target <= 0:
            reason = 'no_competitive_reference'
        elif not harmful_signal:
            reason = 'no_welfare_deficit'
        elif self._persistence < self.patience:
            reason = 'waiting_for_persistence'
        elif not pairs:
            reason = 'no_suspected_pairs'
        else:
            selected = [p for p in pairs if effective_costs.get(p, 0) < self.link_cost_cap - 1e-9][:self.pair_budget]
            changes = {p: min(self.link_cost_delta, self.link_cost_cap - effective_costs.get(p, 0.0)) for p in selected}
            if changes:
                self._pending = ArbiterAction(kind=ArbiterActionKind.change_link_cost, link_cost_changes=changes)
                self.last_decision = Decision('prevention', gap, competitive, len(pairs), len(changes))
                return self._pending
            reason = 'cost_cap_reached'
        self._pending = ArbiterAction()
        self.last_decision = Decision(reason, gap, competitive, len(pairs))
        return self._pending
