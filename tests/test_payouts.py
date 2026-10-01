import pytest

from social_market_env.gametheory.graphs import connected_components
from social_market_env.gametheory.payouts import (
    coalition_surplus,
    equal_split,
    myerson_values,
    owen_values,
    shapley_values,
    to_weights,
)


def glove_game(subset: frozenset[int]) -> float:
    return 1.0 if 1 in subset and subset & {2, 3} else 0.0


def synergy_game(subset: frozenset[int]) -> float:
    worth = {1: 4.0, 2: 4.0, 3: 0.0}
    bonus = 6.0 if {1, 2} <= subset else 0.0
    return sum(worth[player] for player in subset) + bonus


def test_shapley_glove_game() -> None:
    assert shapley_values(members=[1, 2, 3], value=glove_game) == pytest.approx({1: 2 / 3, 2: 1 / 6, 3: 1 / 6})


def test_shapley_axioms() -> None:
    values = shapley_values(members=[1, 2, 3], value=synergy_game)

    assert sum(values.values()) == pytest.approx(synergy_game(frozenset({1, 2, 3})))
    assert values[1] == pytest.approx(values[2])
    assert values[3] == pytest.approx(0.0)


def test_myerson_on_path_ignores_disconnected_pairs() -> None:
    values = myerson_values(members=[1, 2, 3], value=glove_game, edges={(1, 2), (2, 3)})

    assert values == pytest.approx({1: 0.5, 2: 0.5, 3: 0.0})


def test_owen_splits_between_unions_first() -> None:
    values = owen_values(members=[1, 2, 3], value=glove_game, unions=[frozenset({2, 3})])

    assert values == pytest.approx({1: 0.5, 2: 0.25, 3: 0.25})


def test_to_weights_clips_negatives_and_falls_back_to_equal_split() -> None:
    assert to_weights({1: 3.0, 2: 1.0, 3: -2.0}) == pytest.approx({1: 0.75, 2: 0.25, 3: 0.0})
    assert to_weights({1: -1.0, 2: 0.0}) == equal_split([1, 2])


def test_coalition_surplus_is_value_above_standalone() -> None:
    assert coalition_surplus(members=frozenset({1, 2}), value=synergy_game, expected_fines=1.0) == pytest.approx(5.0)


def test_connected_components_ignore_foreign_edges() -> None:
    components = connected_components(nodes=[1, 2, 3, 4], edges=[(1, 2), (3, 9)])

    assert components == [frozenset({1, 2}), frozenset({3}), frozenset({4})]
