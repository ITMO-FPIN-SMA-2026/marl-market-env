"""Market clearing: total demand, split between sellers, profits and consumer surplus."""

from dataclasses import dataclass
from typing import Self

import numpy as np

from social_market_env.config import MarketConfig


@dataclass(frozen=True)
class MarketOutcome:
    """Result of one market clearing."""

    prices: dict[int, float]
    """Prices of participants."""
    quantities: dict[int, float]
    """Sales of participants."""
    gross_profits: dict[int, float]
    """(price - cost) * quantity before pooling and fines."""
    average_price: float
    """Mean price of participants."""
    total_quantity: float
    """Total demand Q."""

    @classmethod
    def empty(cls) -> Self:
        """Outcome of a step without participants."""
        return cls(
            prices={}, quantities={}, gross_profits={},
            average_price=0.0, total_quantity=0.0
        )


def total_demand(*, average_price: float, market: MarketConfig) -> float:
    """Return Q = max(0, A - B * mean price)."""
    return max(0.0, market.demand_a - market.demand_b * average_price)


def share_vector(*, prices: np.ndarray, market: MarketConfig) -> np.ndarray:
    """Return market shares along the last axis of a price vector or matrix; they sum to 1."""
    if prices.shape[-1] == 1:
        return np.ones_like(prices, dtype=float)
    if market.share_model == 'tz_linear':
        weights = 1.0 - prices / prices.sum(axis=-1, keepdims=True)
    else:
        weights = np.exp(-(prices - prices.min(axis=-1, keepdims=True)) / market.tau)
    return weights / weights.sum(axis=-1, keepdims=True)


def market_shares(
    *, prices: dict[int, float], market: MarketConfig
) -> dict[int, float]:
    """Return each participant's share of total demand."""
    ids = list(prices)
    shares = share_vector(
        prices=np.array(
            [prices[agent_id] for agent_id in ids], dtype=float
        ),
        market=market,
    )
    return {agent_id: float(share) for agent_id, share in zip(ids, shares)}


def clear_market(
    *,
    prices: dict[int, float],
    costs: dict[int, float],
    market: MarketConfig,
) -> MarketOutcome:
    """Compute demand, sales and gross profits for the given prices."""
    if not prices:
        return MarketOutcome.empty()
    average_price = float(np.mean(list(prices.values())))
    total_quantity = total_demand(average_price=average_price, market=market)
    shares = market_shares(prices=prices, market=market)
    quantities = {
        agent_id: total_quantity * shares[agent_id] for agent_id in prices
    }
    return MarketOutcome(
        prices=dict(prices),
        quantities=quantities,
        gross_profits={
            agent_id: (prices[agent_id] - costs[agent_id])
            * quantities[agent_id]
            for agent_id in prices
        },
        average_price=average_price,
        total_quantity=total_quantity,
    )


def consumer_surplus(
    *, total_quantity: float, market: MarketConfig
) -> float:
    """Return the buyers' surplus Q^2 / (2B) under linear demand."""
    return total_quantity**2 / (2.0 * market.demand_b)
