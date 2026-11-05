"""Legal dynamic action candidates, without recomputing market outcomes."""

from dataclasses import dataclass

import numpy as np

from social_market_env.contracts import AgentObservation, LinkAction, LinkActionKind

from .features import finite
from .settings import MarketSettings, Settings

PRICE_FEATURES = ('price', 'margin', 'average_gap', 'average_present', 'last_gap', 'last_present',
                  'offer_gap', 'offer_present', 'at_or_above_cost')
LINK_FEATURES = ('neighbor_weight', 'weight_present', 'cost', 'cost_present', 'last_price',
                 'price_present', 'offer_member', 'same_previous_price', 'same_price_present')
LINK_KINDS = (None, LinkActionKind.add, LinkActionKind.maintain, LinkActionKind.remove)


@dataclass(frozen=True)
class Candidates:
    can_sell: bool
    can_join: bool
    fallback_price: float
    prices: tuple[float, ...]
    price_features: np.ndarray
    links: tuple[tuple[LinkAction, ...], ...]
    link_features: tuple[np.ndarray, ...]


def build_candidates(o: AgentObservation, settings: Settings, market: MarketSettings) -> Candidates:
    floor = max(market.p_min, float(o.cost)) if o.restricted and finite(o.cost) else market.p_min
    can_sell = bool(o.active and finite(o.cost) and floor <= market.p_max)
    safe_floor = min(market.p_max, floor)
    clip = lambda value: float(np.clip(value, safe_floor, market.p_max))
    fallback = clip(float(o.cost)) if finite(o.cost) else market.p_min
    raw = list(np.linspace(safe_floor, market.p_max, settings.price_nodes))
    for anchor in (o.cost, o.last_price, o.market.average_price):
        if finite(anchor):
            raw.extend(clip(float(anchor) + offset * market.price_offset) for offset in (-1, 0, 0.2, 0.6, 1, 2))
    if o.offer and finite(o.offer.recommended_price):
        raw.append(clip(o.offer.recommended_price))
    prices = tuple(sorted({round(clip(value), 8) for value in raw}))
    # Preserve exact bounds even when a bound has more than eight decimal places.
    prices = tuple(sorted({float(np.clip(value, safe_floor, market.p_max)) for value in prices}))
    can_join = bool(can_sell and not o.restricted and o.offer is not None and o.agent_id in o.offer.members
                    and finite(o.offer.recommended_price) and market.p_min <= o.offer.recommended_price <= market.p_max)
    price_rows = []
    for price in prices:
        row = [price / market.p_max, (price - float(o.cost)) / market.p_max if finite(o.cost) else 0]
        for anchor in (o.market.average_price, o.last_price, o.offer.recommended_price if o.offer else None):
            row.extend([(price - float(anchor)) / market.p_max if finite(anchor) else 0, int(finite(anchor))])
        row.append(int(finite(o.cost) and price >= o.cost))
        price_rows.append(row)
    valid_id = lambda target: isinstance(target, (int, np.integer)) and not isinstance(target, (bool, np.bool_)) and target != o.agent_id
    neighbors = sorted(int(target) for target in o.neighbors if valid_id(target))
    add_targets = sorted(int(target) for target in o.link_costs if valid_id(target) and target not in o.neighbors)
    groups = ((), tuple(LinkAction(LinkActionKind.add, target) for target in add_targets),
              tuple(LinkAction(LinkActionKind.maintain, target) for target in neighbors),
              tuple(LinkAction(LinkActionKind.remove, target) for target in neighbors))
    rows = []
    for group in groups:
        features = []
        for action in group:
            target = action.target
            weight, cost, price = o.neighbors.get(target), o.link_costs.get(target), o.market.last_prices.get(target)
            same_price_present = finite(price) and finite(o.last_price)
            features.append([float(weight) if finite(weight) else 0, int(finite(weight)),
                             float(cost) / settings.reward_scale if finite(cost) else 0, int(finite(cost)),
                             float(price) / market.p_max if finite(price) else 0, int(finite(price)),
                             int(o.offer is not None and target in o.offer.members),
                             int(same_price_present and abs(price - o.last_price) < 0.5), int(same_price_present)])
        rows.append(np.clip(np.asarray(features, dtype=np.float32).reshape(-1, len(LINK_FEATURES)), -10, 10))
    if not o.active:
        groups = ((), (), (), ())
        rows = [np.empty((0, len(LINK_FEATURES)), dtype=np.float32) for _ in groups]
    return Candidates(can_sell, can_join, fallback, prices,
                      np.clip(np.asarray(price_rows, dtype=np.float32), -10, 10), groups, tuple(rows))
