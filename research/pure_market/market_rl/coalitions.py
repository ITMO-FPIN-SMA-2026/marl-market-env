"""Exact small-game allocation. Values may be negative; transfers conserve money."""

from itertools import permutations, product
from typing import Callable

import numpy as np


def components(members, graph):
    remaining = set(members)
    result = []
    while remaining:
        reached = {min(remaining)}
        frontier = list(reached)
        while frontier:
            i = frontier.pop()
            new = {j for j in remaining - reached if graph[i, j]}
            reached.update(new)
            frontier.extend(new)
        remaining.difference_update(reached)
        result.append(tuple(sorted(reached)))
    return result


def allocate(members, value: Callable, method="equal", graph=None, history=()):
    """Split the *realized stage surplus*, not a forecast of discounted value.

    Myerson uses the graph-restricted game. Owen uses the previous round's
    partition, intersected with the current coalition (newcomers are singletons).
    """
    members = tuple(sorted(members))
    if method not in {"equal", "shapley", "myerson", "owen"}:
        raise ValueError(f"Unknown allocation: {method}")
    if not members:
        return {}
    if method == "equal":
        return dict.fromkeys(members, value(frozenset(members)) / len(members))
    if method == "myerson":
        if graph is None:
            raise ValueError("Myerson allocation needs a graph")
        original_value = value

        def value(subset):
            return sum(original_value(frozenset(c)) for c in components(subset, graph))

    orders = permutations(members)
    if method == "owen":
        blocks = [tuple(i for i in block if i in members) for block in history]
        blocks = [b for b in blocks if b]
        flattened = [i for b in blocks for i in b]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Historical coalitions must form a partition")
        blocks += [(i,) for i in members if i not in flattened]
        orders = (
            tuple(i for block in internal for i in block)
            for order in permutations(blocks)
            for internal in product(*(tuple(permutations(block)) for block in order))
        )
    shares = dict.fromkeys(members, 0.0)
    count = 0
    for order in orders:
        before = frozenset()
        for i in order:
            after = before | {i}
            shares[i] += value(after) - value(before)
            before = after
        count += 1
    return {i: amount / count for i, amount in shares.items()}


def clear_market(prices, costs, active, demand_a, demand_b, *, model="assignment",
                 logit_value=8.0, logit_temperature=1.5):
    """Assignment shares, or multinomial logit with an outside purchase option.

    Logit consumer surplus is the inclusive value in monetary units. The
    outside option's utility is zero and its expenditure is zero.
    """
    if model not in {"assignment", "logit"}:
        raise ValueError("Unknown demand model")
    ids = np.flatnonzero(active)
    quantities = np.zeros(len(prices))
    if not len(ids):
        return quantities, quantities.copy(), 0.0, 0.0
    p = prices[ids]
    if model == "logit":
        utilities = (logit_value - p) / logit_temperature
        log_denominator = np.logaddexp(0., np.logaddexp.reduce(utilities))
        quantities[ids] = demand_a * np.exp(utilities - log_denominator)
        total = quantities.sum()
        mean_price = float(p @ quantities[ids] / total) if total > 0 else 0.0
        profits = (prices - costs) * quantities
        consumer_surplus = float(demand_a * logit_temperature * log_denominator)
        return quantities, profits, mean_price, consumer_surplus
    weights = np.ones(1) if len(ids) == 1 else (1 - p / p.sum()) / (len(ids) - 1)
    mean_price = float(weights @ p)
    total = max(0.0, demand_a - demand_b * mean_price)
    quantities[ids] = total * weights
    profits = (prices - costs) * quantities
    consumer_surplus = total**2 / (2 * demand_b)
    return quantities, profits, mean_price, consumer_surplus
