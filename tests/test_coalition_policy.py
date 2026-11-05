"""Тесты этапа 1 политики коалиций: математика долей, цены, структура и прогон в среде."""

import numpy as np
import pytest

from coalition_policy import AdaptiveCoalitionPolicy, PolicySettings
from coalition_policy.candidates import generate_candidates
from coalition_policy.payouts import level_game, payout_weights, restrict_to_graph, shapley_table
from coalition_policy.pricing import blended_prices, mean_price_cap
from coalition_policy.structure import maximal_packings, select_partition
from social_market_env.config import MarketConfig, config_from_dict
from social_market_env.contracts import CoalitionObservation
from social_market_env.core.market import clear_market, consumer_surplus
from social_market_env.gametheory.payouts import myerson_values, shapley_values
from social_market_env.runner.experiment import build_environment

POLICY = 'coalition_policy:AdaptiveCoalitionPolicy'


def group(*members: int) -> frozenset[int]:
    return frozenset(members)


def observation(*, active: tuple[int, ...], edges: tuple[tuple[int, int], ...], restricted=()) -> CoalitionObservation:
    return CoalitionObservation(
        step=0, active_agents=active, restricted=restricted, costs={agent: 18.5 for agent in active},
        balances={agent: 1000.0 for agent in active}, edges={edge: 0.0 for edge in edges}, link_costs={},
        last_prices={}, last_profits={}, compliance_history={},
    )


def random_game(count: int) -> np.ndarray:
    values = np.random.default_rng(1).uniform(0, 10, size=1 << count)
    values[0] = 0.0
    return values


def as_function(values: np.ndarray, players: list[int]):
    return lambda subset: float(values[sum(1 << players.index(player) for player in subset)])


def test_shapley_matches_lecture_example() -> None:
    # Биты масок: агент 1 — бит 0, агент 2 — бит 1, агент 3 — бит 2. Пример из Шепли.docx.
    values = np.array([0.0, 1.0, 2.0, 5.0, 1.0, 7.0, 6.0, 10.0])

    assert shapley_table(values) == pytest.approx([19 / 6, 19 / 6, 22 / 6])


def test_fast_shapley_matches_environment_library() -> None:
    players = [3, 5, 7, 9, 11]
    values = random_game(5)

    expected = shapley_values(members=players, value=as_function(values, players))

    assert shapley_table(values) == pytest.approx([expected[player] for player in players])


def test_fast_myerson_matches_environment_library() -> None:
    players = [3, 5, 7, 9, 11]
    edges = [(3, 5), (5, 7), (9, 11)]
    values = random_game(5)

    expected = myerson_values(members=players, value=as_function(values, players), edges=edges)
    fast = shapley_table(restrict_to_graph(values, players=players, edges=edges))

    assert fast == pytest.approx([expected[player] for player in players])


def test_level_game_matches_market_and_keeps_nash_floor() -> None:
    market = MarketConfig()
    costs = {0: 16.0, 1: 19.0, 2: 21.0}
    prices = {0: 30.0, 1: 31.0, 2: 24.0}
    nash = {0: 23.0, 1: 24.0, 2: 24.0}
    no_floor = {0: 0.0, 1: 0.0, 2: 0.0}

    table = level_game(players=[0, 1], prices=prices, nash_prices=nash, nash_profits=no_floor, costs=costs,
                       market=market)
    alone = clear_market(prices={0: 30.0, 1: 24.0, 2: 24.0}, costs=costs, market=market).gross_profits[0]
    both = clear_market(prices=prices, costs=costs, market=market).gross_profits

    assert table[0b01] == pytest.approx(alone)
    assert table[0b11] == pytest.approx(both[0] + both[1])
    high_floor = level_game(players=[0, 1], prices=prices, nash_prices=nash, nash_profits={0: 1e6, 1: 0.0, 2: 0.0},
                            costs=costs, market=market)
    assert high_floor[0b01] == 1e6


def test_weights_are_valid_and_center_of_star_gets_more() -> None:
    # Ценность есть только у групп от двух агентов: без центра звезды листья разобщены.
    sizes = np.array([bin(mask).count('1') for mask in range(16)])
    values = np.where(sizes >= 2, sizes.astype(float), 0.0)

    weights = payout_weights(scheme='myerson', players=[0, 1, 2, 3], values=values, edges=[(0, 1), (0, 2), (0, 3)])

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[0] > weights[1]
    assert weights[1] == pytest.approx(weights[2])


def test_price_cap_gives_requested_welfare_gap() -> None:
    market = MarketConfig()
    nash = {agent: 24.0 for agent in range(4)}
    target = {agent: 60.0 for agent in range(4)}
    welfare = consumer_surplus(total_quantity=96.0, market=market)
    cap = mean_price_cap(target_welfare=welfare, gap=0.08, market=market)

    prices = blended_prices(nash=nash, target=target, members={0, 1, 2, 3}, mean_cap=cap, market=market)
    quantity = market.demand_a - np.mean(list(prices.values()))

    assert consumer_surplus(total_quantity=quantity, market=market) == pytest.approx(0.92 * welfare)


def test_cap_does_not_hold_back_price_cuts() -> None:
    # При tz_linear сговор снижает цену, такому плану потолок мешать не должен.
    market = MarketConfig(share_model='tz_linear')

    prices = blended_prices(nash={0: 100.0, 1: 100.0}, target={0: 70.0, 1: 70.0}, members={0, 1}, mean_cap=100.0,
                            market=market)

    assert prices == {0: 70.0, 1: 70.0}


def test_packings_are_disjoint_and_maximal() -> None:
    candidates = [group(0, 1), group(1, 2), group(2, 3), group(0, 1, 2, 3)]

    packings = maximal_packings(candidates, limit=10)

    assert set(packings) == {(group(0, 1), group(2, 3)), (group(0, 1, 2, 3),), (group(1, 2),)}


def test_hysteresis_keeps_previous_partition() -> None:
    settings = PolicySettings(hysteresis=0.1, max_evaluations=5)
    old, new = (group(0, 1),), (group(1, 2),)
    scores = {old: 100.0, new: 105.0}
    candidates = [group(0, 1), group(1, 2)]

    assert select_partition(candidates=candidates, previous=(), score=scores.get, settings=settings) == new
    assert select_partition(candidates=candidates, previous=old, score=scores.get, settings=settings) == old
    scores[new] = 120.0
    assert select_partition(candidates=candidates, previous=old, score=scores.get, settings=settings) == new


@pytest.mark.parametrize('structure', ['antagonism', 'grand'])
def test_modes_without_candidates_on_split_graph(structure: str) -> None:
    seen = observation(active=(0, 1, 2, 3), edges=((0, 1), (2, 3)))

    assert generate_candidates(observation=seen, previous=(), settings=PolicySettings(structure=structure)) == []


def test_candidates_are_connected_and_skip_restricted() -> None:
    seen = observation(active=(0, 1, 2, 3), edges=((0, 1), (1, 2), (2, 3)), restricted=(1,))

    assert generate_candidates(observation=seen, previous=(group(0, 1),), settings=PolicySettings()) == [group(2, 3)]


def test_partial_mode_limits_group_size() -> None:
    edges = tuple((first, second) for first in range(6) for second in range(first + 1, 6))
    seen = observation(active=tuple(range(6)), edges=edges)

    found = generate_candidates(observation=seen, previous=(), settings=PolicySettings(structure='partial'))

    assert found and all(2 <= len(item) <= 3 for item in found)


def run_config(**overrides) -> list[dict]:
    kwargs = overrides.pop('kwargs', {})
    data = {
        'horizon': 12,
        'strict': True,
        'agents': {'count': 8, 'default': {'impl': 'social_market_env.baselines.agents:CartelAgent'}},
        'coalitions': {'policy': {'impl': POLICY, 'kwargs': kwargs}},
        **overrides,
    }
    env = build_environment(config=config_from_dict(data), seed=0)
    env.reset(seed=0, training=False)
    records = []
    while not env.done:
        records.append(env.step().record)
    return records


@pytest.mark.parametrize('topology', ['complete', 'star', 'path', 'inner_coalitions', 'random'])
@pytest.mark.parametrize('mode', ['static', 'evolving'])
@pytest.mark.parametrize('payout', ['equal', 'shapley', 'myerson'])
def test_no_rejected_plans_in_any_environment(topology: str, mode: str, payout: str) -> None:
    records = run_config(network={'topology': topology, 'mode': mode}, kwargs={'payout': payout})

    assert all(not record['rejected_plans'] and not record['events'] for record in records)


def test_antagonism_makes_no_plans() -> None:
    assert not any(record['plans'] for record in run_config(kwargs={'structure': 'antagonism'}))


def test_grand_mode_covers_everyone_on_complete_static_graph() -> None:
    records = run_config(network={'topology': 'complete', 'mode': 'static'}, kwargs={'structure': 'grand'})

    assert all(len(record['plans']) == 1 and len(record['plans'][0]['members']) == 8 for record in records)


def test_compliant_market_stays_within_price_level() -> None:
    records = run_config(network={'topology': 'complete', 'mode': 'static'}, kwargs={'price_level': 'moderate'})

    gaps = [1 - record['social_welfare'] / record['target_social_welfare'] for record in records]
    assert 0.05 < max(gaps) <= 0.07 + 1e-6


def test_policy_is_deterministic() -> None:
    first = run_config(kwargs={'payout': 'shapley'})
    second = run_config(kwargs={'payout': 'shapley'})

    assert [record['plans'] for record in first] == [record['plans'] for record in second]


def test_unknown_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        AdaptiveCoalitionPolicy(payout='owen')
    with pytest.raises(TypeError):
        AdaptiveCoalitionPolicy(unknown=1)
