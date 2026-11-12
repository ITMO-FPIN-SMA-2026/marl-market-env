"""Exact pure Nash benchmark on the four-agent price grid, evaluator-only."""

from functools import lru_cache
from itertools import product

import numpy as np

from .coalitions import clear_market


@lru_cache(maxsize=128)
def competitive_prices(config, active_ids):
    """Choose the highest-welfare pure equilibrium if the grid has several.

    Refuse an unsupported grid without a pure equilibrium instead of silently
    calling an arbitrary best-response iterate an equilibrium.
    """
    active_ids = tuple(active_ids)
    if not active_ids:
        return tuple([config.prices[0]] * 4)
    size, count = len(config.prices), len(active_ids)
    profits = np.zeros((size,)*count+(count,))
    welfare = np.zeros((size,)*count)
    active = np.zeros(4, dtype=bool)
    active[list(active_ids)] = True
    for choices in product(range(size), repeat=count):
        prices = np.full(4, config.prices[0], dtype=float)
        prices[list(active_ids)] = np.array(config.prices)[list(choices)]
        _, reward, _, cs = clear_market(prices, np.array(config.costs), active, config.demand_a, config.demand_b,
            model=config.demand_model, logit_value=config.logit_value, logit_temperature=config.logit_temperature)
        profits[choices] = reward[list(active_ids)]
        welfare[choices] = cs + reward.sum()
    stable = np.ones((size,)*count, dtype=bool)
    for player in range(count):
        value = profits[..., player]
        stable &= value >= value.max(axis=player, keepdims=True)-1e-9
    equilibria = np.argwhere(stable)
    if not len(equilibria):
        raise ValueError("This price grid has no pure Nash equilibrium; choose a supported benchmark/grid")
    best = max(equilibria, key=lambda choices: welfare[tuple(choices)])
    result = np.full(4, config.prices[0], dtype=float)
    result[list(active_ids)] = np.array(config.prices)[best]
    return tuple(result.tolist())
