import numpy as np
import pytest

from social_market_env.config import MarketConfig
from social_market_env.gametheory.equilibrium import (
    CharacteristicFunction,
    best_response_price,
    coalition_equilibrium,
    nash_equilibrium,
)

COSTS = {agent_id: 18.0 for agent_id in range(12)}
MARKET = MarketConfig(demand_a=120.0, demand_b=1.0, tau=5.0)
MIXED_COSTS = {
    0: 19.46, 1: 16.89, 2: 15.29, 3: 15.12, 4: 20.69, 5: 21.39,
    6: 19.25, 7: 20.11, 8: 18.81, 9: 21.55, 10: 20.71, 11: 15.02,
}
CHEAPEST = min(MIXED_COSTS, key=MIXED_COSTS.__getitem__)
DEAREST = max(MIXED_COSTS, key=MIXED_COSTS.__getitem__)


def test_nash_price_matches_reference_numbers() -> None:
    equilibrium = nash_equilibrium(costs=COSTS, market=MARKET)

    assert equilibrium.converged
    assert equilibrium.prices[0] == pytest.approx(23.5, abs=0.3)


def test_grand_coalition_sets_monopoly_price() -> None:
    equilibrium = coalition_equilibrium(members=frozenset(COSTS), costs=COSTS, market=MARKET)

    assert equilibrium.prices[0] == pytest.approx(69.0, abs=0.05)


def test_cartel_is_worth_more_than_competition() -> None:
    value = CharacteristicFunction(costs=COSTS, market=MARKET)
    everyone = frozenset(COSTS)

    assert value(everyone) > sum(value(frozenset({agent_id})) for agent_id in everyone)


@pytest.mark.parametrize('pair', [(CHEAPEST, DEAREST), (0, 1), (2, 3), (4, 5)])
def test_pair_value_is_superadditive_with_heterogeneous_costs(pair: tuple[int, int]) -> None:
    value = CharacteristicFunction(costs=MIXED_COSTS, market=MARKET)
    first, second = pair

    together = value(frozenset(pair))
    alone = value(frozenset({first})) + value(frozenset({second}))

    assert together >= alone - 1e-6


def test_cartel_members_with_different_costs_get_their_own_prices() -> None:
    members = frozenset({CHEAPEST, DEAREST})

    equilibrium = coalition_equilibrium(members=members, costs=MIXED_COSTS, market=MARKET)

    assert equilibrium.converged
    assert equilibrium.prices[DEAREST] > equilibrium.prices[CHEAPEST]


def test_best_response_price_matches_nash_price_against_nash_rivals() -> None:
    equilibrium = nash_equilibrium(costs=MIXED_COSTS, market=MARKET)
    others = np.array([price for agent_id, price in sorted(equilibrium.prices.items()) if agent_id != 0])

    price = best_response_price(cost=MIXED_COSTS[0], other_prices=others, market=MARKET)

    assert price == pytest.approx(equilibrium.prices[0], abs=0.05)
