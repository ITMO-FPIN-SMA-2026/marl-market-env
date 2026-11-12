"""Bounded fines based on persistent public market signals."""

from collections import deque
from dataclasses import dataclass
import math
from statistics import fmean
from typing import Any

from social_market_env.config import MarketConfig
from social_market_env.contracts import ArbiterAction, ArbiterActionKind, ArbiterAPI, ArbiterObservation


@dataclass(frozen=True)
class Decision:
    """Public explanation; a suspicious price is not proof of coalition membership."""

    reason: str
    welfare_gap: float = 0.0
    competitive_average_price: float | None = None
    candidate_agents: int = 0
    sanction_target: int | None = None
    sanction_fine: float = 0.0


class PreventiveArbiter(ArbiterAPI):
    """Observe or fine one seller using public prices and consumer surplus.

    The import name is retained for runner configurations. This version never
    requests link-cost changes and works in static and evolving networks alike.
    Public market parameters are injected by the runner. No graph, coalition
    membership, seller costs, balances or implementation names are inspected.
    """

    def __init__(
        self,
        *,
        market: dict[str, Any] | None = None,
        history_window: int = 20,
        welfare_gap_threshold: float = 0.08,
        patience: int = 4,
        price_margin: float = 2.0,
        fine_amount: float = 25.0,
        fine_cooldown: int = 5,
        fine_budget: float = 100.0,
        min_fine: float = 1.0,
    ) -> None:
        for name, value in [('history_window', history_window), ('patience', patience),
                            ('fine_cooldown', fine_cooldown)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in [('welfare_gap_threshold', welfare_gap_threshold), ('price_margin', price_margin),
                            ('fine_amount', fine_amount), ('fine_budget', fine_budget), ('min_fine', min_fine)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a finite non-negative number')
        if welfare_gap_threshold >= 1:
            raise ValueError('welfare_gap_threshold must be less than 1')
        if min_fine <= 0 or min(fine_amount, fine_budget) < min_fine:
            raise ValueError('fine_amount and fine_budget must be at least min_fine > 0')
        self.market = MarketConfig(**(market or {}))
        if not math.isfinite(self.market.demand_a) or self.market.demand_a <= 0:
            raise ValueError('market.demand_a must be finite and positive')
        if not math.isfinite(self.market.demand_b) or self.market.demand_b <= 0:
            raise ValueError('market.demand_b must be finite and positive')
        if not all(math.isfinite(x) for x in (self.market.p_min, self.market.p_max)):
            raise ValueError('market price bounds must be finite')
        if not 0 < self.market.p_min < self.market.p_max:
            raise ValueError('market price bounds must satisfy 0 < p_min < p_max')
        self.history_window = history_window
        self.welfare_gap_threshold = welfare_gap_threshold
        self.patience = patience
        self.price_margin = price_margin
        self.fine_amount = fine_amount
        self.fine_cooldown = fine_cooldown
        self.fine_budget = fine_budget
        self.min_fine = min_fine
        self.reset(seed=0, training=False)

    def reset(self, *, seed: int, training: bool) -> None:
        """Clear episode memory. The rule itself does not train."""
        self._prices: deque[dict[int, float]] = deque(maxlen=self.history_window)
        self._streaks: dict[int, int] = {}
        self._paid_fines: dict[int, float] = {}
        self._last_fined: dict[int, int] = {}
        self._last_fine_step: int | None = None
        self._last_step: int | None = None
        self._pending = ArbiterAction()
        self.last_decision = Decision('reset')

    def _reconcile(self, observation: ArbiterObservation) -> None:
        """Charge the episode budget only for the actual sanitised previous fine."""
        if self._last_step is not None:
            if observation.step != self._last_step + 1:
                raise ValueError('Observations must be consecutive; reset before a new episode')
            applied = observation.last_action if observation.last_action is not None else self._pending
            if applied.kind == ArbiterActionKind.sanction and applied.target is not None and applied.fine > 0:
                target = applied.target
                self._paid_fines[target] = self._paid_fines.get(target, 0.0) + applied.fine
                self._last_fined[target] = self._last_step
                self._last_fine_step = self._last_step
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
        candidates = [i for i in sorted(prices)
                      if competitive is not None and prices[i] > competitive + self.price_margin]
        harmful_signal = target > 0 and gap > self.welfare_gap_threshold
        # Each seller needs their own consecutive evidence, alongside a group
        # and a market deficit. A newcomer cannot inherit others' history.
        self._streaks = {i: self._streaks.get(i, 0) + 1 for i in candidates} if (
            harmful_signal and len(candidates) >= 2) else {}
        risks = {}
        for i in candidates:
            old = [frame[i] for frame in self._prices if i in frame]
            trend = max(0.0, (prices[i] - fmean(old)) / scale) if old else 0.0
            risks[i] = (prices[i] - competitive) / scale + 0.25 * trend
        self._prices.append(dict(prices))
        eligible = [i for i in candidates if self._streaks.get(i, 0) >= self.patience]
        remaining = {i: max(0.0, self.fine_budget - self._paid_fines.get(i, 0.0)) for i in eligible}
        affordable = [i for i in eligible if remaining[i] >= self.min_fine]

        if target <= 0:
            reason = 'no_competitive_reference'
        elif not harmful_signal:
            reason = 'no_welfare_deficit'
        elif len(candidates) < 2:
            reason = 'no_suspected_group'
        elif not eligible:
            reason = 'waiting_for_persistence'
        elif not affordable:
            reason = 'fine_budget_reached'
        elif self._last_fine_step is not None and observation.step - self._last_fine_step < self.fine_cooldown:
            reason = 'fine_cooldown'
        else:
            seller = min(affordable, key=lambda i: (-risks[i], self._last_fined.get(i, -1), i))
            fine = min(self.fine_amount, remaining[seller])
            self._pending = ArbiterAction(kind=ArbiterActionKind.sanction, target=seller, fine=fine)
            self.last_decision = Decision('sanction', gap, competitive, len(candidates), seller, fine)
            return self._pending
        self._pending = ArbiterAction()
        self.last_decision = Decision(reason, gap, competitive, len(candidates))
        return self._pending
