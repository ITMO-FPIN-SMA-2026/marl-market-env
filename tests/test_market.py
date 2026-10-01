import pytest

from social_market_env.config import MarketConfig
from social_market_env.core.market import clear_market, consumer_surplus


def test_quantities_add_up_to_demand_and_cheaper_sells_more() -> None:
    outcome = clear_market(
        prices={0: 20.0, 1: 30.0, 2: 40.0},
        costs={0: 10.0, 1: 10.0, 2: 10.0},
        market=MarketConfig(tau=5.0),
    )

    assert outcome.average_price == pytest.approx(30.0)
    assert outcome.total_quantity == pytest.approx(90.0)
    assert sum(outcome.quantities.values()) == pytest.approx(90.0)
    assert outcome.quantities[0] > outcome.quantities[1] > outcome.quantities[2]
    assert outcome.gross_profits[0] == pytest.approx(10.0 * outcome.quantities[0])


def test_tz_linear_shares_follow_assignment_formula() -> None:
    outcome = clear_market(
        prices={0: 10.0, 1: 30.0},
        costs={0: 0.0, 1: 0.0},
        market=MarketConfig(share_model='tz_linear'),
    )

    assert outcome.quantities[0] / outcome.total_quantity == pytest.approx(0.75)


def test_consumer_surplus_is_triangle_under_demand() -> None:
    assert consumer_surplus(
        total_quantity=50.0, market=MarketConfig(demand_b=2.0)
    ) == pytest.approx(625.0)


def test_empty_market_has_no_sales() -> None:
    assert clear_market(
        prices={}, costs={}, market=MarketConfig()
    ).total_quantity == 0.0
