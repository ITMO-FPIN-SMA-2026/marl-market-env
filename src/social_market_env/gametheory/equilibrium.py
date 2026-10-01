"""Numerical equilibria of the one-shot pricing game and the coalition value v(C)."""

from dataclasses import dataclass

import numpy as np

from social_market_env.config import MarketConfig
from social_market_env.core.market import clear_market, share_vector

COARSE_STEP: float = 0.5
FINE_STEP: float = 0.01
MAX_SWEEPS: int = 5


@dataclass(frozen=True)
class Equilibrium:
    """Prices and profits where no player gains by changing its price."""

    prices: dict[int, float]
    """Equilibrium price of every agent."""
    profits: dict[int, float]
    """Gross profit of every agent at these prices."""
    total_quantity: float
    """Total demand at these prices."""
    converged: bool
    """False if the iteration limit was reached first."""


@dataclass(frozen=True)
class GroupProblem:
    """Joint-profit pricing problem of one group against fixed prices of everyone else."""

    costs: np.ndarray
    """Unit costs of all players in price-vector order."""
    group: np.ndarray
    """Boolean mask of the players whose profits are pooled."""
    market: MarketConfig
    """Market the group sells in."""


def best_response_price(*, cost: float, other_prices: np.ndarray, market: MarketConfig) -> float:
    """Return the price that maximizes one agent's profit against fixed prices of the others."""
    prices = np.concatenate(([cost], np.asarray(other_prices, dtype=float)))
    costs = np.concatenate(([cost], np.zeros(prices.size - 1)))
    group = np.arange(prices.size) == 0
    return _best_member_price(member=0, prices=prices, problem=GroupProblem(costs=costs, group=group, market=market))


def group_equilibrium(
    *,
    groups: list[tuple[int, ...]],
    costs: dict[int, float],
    market: MarketConfig,
    tolerance: float = 0.02,
    max_iterations: int = 300,
    damping: float = 0.5,
) -> Equilibrium:
    """Find prices where every group best-responds; a group sets individual member prices for its joint profit."""
    prices = {agent_id: min(market.p_max, costs[agent_id] + 1.0) for group in groups for agent_id in group}
    for _ in range(max_iterations):
        responses = _best_responses(groups=groups, prices=prices, costs=costs, market=market)
        shift = max(abs(responses[agent_id] - prices[agent_id]) for agent_id in prices)
        if shift < tolerance:
            return _equilibrium(prices=responses, costs=costs, market=market, converged=True)
        prices = {agent_id: damping * prices[agent_id] + (1 - damping) * responses[agent_id] for agent_id in prices}
    return _equilibrium(prices=prices, costs=costs, market=market, converged=False)


def nash_equilibrium(*, costs: dict[int, float], market: MarketConfig) -> Equilibrium:
    """Return the competitive equilibrium where every agent prices alone."""
    return group_equilibrium(groups=[(agent_id,) for agent_id in sorted(costs)], costs=costs, market=market)


def coalition_equilibrium(*, members: frozenset[int], costs: dict[int, float], market: MarketConfig) -> Equilibrium:
    """Return the equilibrium where members act as one cartel and everyone else prices alone."""
    outsiders = [(agent_id,) for agent_id in sorted(costs) if agent_id not in members]
    return group_equilibrium(groups=[tuple(sorted(members)), *outsiders], costs=costs, market=market)


class CharacteristicFunction:
    """Lazy memoized v(S): joint equilibrium profit of S acting as a cartel among fixed active agents."""

    def __init__(self, *, costs: dict[int, float], market: MarketConfig) -> None:
        self._costs = dict(costs)
        self._market = market
        self._cache: dict[frozenset[int], float] = {frozenset(): 0.0}

    def __call__(self, subset: frozenset[int]) -> float:
        """Return v(subset)."""
        if subset not in self._cache:
            equilibrium = coalition_equilibrium(members=subset, costs=self._costs, market=self._market)
            self._cache[subset] = sum(equilibrium.profits[agent_id] for agent_id in subset)
        return self._cache[subset]


def _best_responses(
    *, groups: list[tuple[int, ...]], prices: dict[int, float], costs: dict[int, float], market: MarketConfig
) -> dict[int, float]:
    ids = list(prices)
    position = {agent_id: index for index, agent_id in enumerate(ids)}
    price_vector = np.array([prices[agent_id] for agent_id in ids])
    cost_vector = np.array([costs[agent_id] for agent_id in ids])
    responses: dict[int, float] = {}
    for group in groups:
        members = [position[agent_id] for agent_id in group]
        problem = GroupProblem(costs=cost_vector, group=np.isin(np.arange(len(ids)), members), market=market)
        response = _group_response(members=members, prices=price_vector, problem=problem)
        responses.update({agent_id: float(response[position[agent_id]]) for agent_id in group})
    return responses


def _group_response(*, members: list[int], prices: np.ndarray, problem: GroupProblem) -> np.ndarray:
    """Coordinate ascent on the group's joint profit, one member price at a time."""
    current = prices.copy()
    sweeps = 1 if len(members) == 1 else MAX_SWEEPS
    for _ in range(sweeps):
        largest_move = 0.0
        for member in members:
            price = _best_member_price(member=member, prices=current, problem=problem)
            largest_move = max(largest_move, abs(price - current[member]))
            current[member] = price
        if largest_move <= FINE_STEP:
            break
    return current


def _best_member_price(*, member: int, prices: np.ndarray, problem: GroupProblem) -> float:
    market = problem.market
    coarse = np.arange(market.p_min, market.p_max + 1e-9, COARSE_STEP)
    best = _argmax_member_price(candidates=coarse, member=member, prices=prices, problem=problem)
    low = max(market.p_min, best - COARSE_STEP)
    high = min(market.p_max, best + COARSE_STEP)
    fine = np.arange(low, high + 1e-9, FINE_STEP)
    return _argmax_member_price(candidates=fine, member=member, prices=prices, problem=problem)


def _argmax_member_price(*, candidates: np.ndarray, member: int, prices: np.ndarray, problem: GroupProblem) -> float:
    price_matrix = np.tile(prices, (candidates.size, 1))
    price_matrix[:, member] = candidates
    profits = _joint_profit(price_matrix=price_matrix, problem=problem)
    return float(candidates[int(np.argmax(profits))])


def _joint_profit(*, price_matrix: np.ndarray, problem: GroupProblem) -> np.ndarray:
    """Pooled profit of the group for every row of candidate prices, mirroring clear_market."""
    market = problem.market
    quantity = np.maximum(0.0, market.demand_a - market.demand_b * price_matrix.mean(axis=1))
    shares = share_vector(prices=price_matrix, market=market)
    margins = price_matrix - problem.costs
    return quantity * (shares * margins)[:, problem.group].sum(axis=1)


def _equilibrium(
    *, prices: dict[int, float], costs: dict[int, float], market: MarketConfig, converged: bool
) -> Equilibrium:
    outcome = clear_market(prices=prices, costs=costs, market=market)
    return Equilibrium(
        prices=dict(prices),
        profits=dict(outcome.gross_profits),
        total_quantity=outcome.total_quantity,
        converged=converged,
    )
