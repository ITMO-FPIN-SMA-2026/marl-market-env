import numpy as np
import pytest

from social_market_env.config import ArbiterConfig, MarketConfig
from social_market_env.contracts import (
    AgentAction,
    ArbiterAction,
    ArbiterActionKind,
    CoalitionDecision,
    CoalitionPlan,
    LinkAction,
    LinkActionKind,
)
from social_market_env.core.guard import CallGuard
from social_market_env.core.sanitize import (
    NO_PARTICIPATION,
    InvalidActionError,
    sanitize_agent_action,
    sanitize_arbiter_action,
    sanitize_coalition_decision,
)

LIMITS = ArbiterConfig(max_fine=10.0)


def test_restricted_agent_cannot_dump_or_join() -> None:
    action = sanitize_agent_action(
        action=AgentAction(price=5.0, coalition_choice=3), cost=18.0, restricted=True, market=MarketConfig()
    )

    assert (action.price, action.coalition_choice) == (18.0, None)


def test_price_is_clipped_to_bounds() -> None:
    action = sanitize_agent_action(action=AgentAction(price=500.0), cost=18.0, restricted=False, market=MarketConfig())

    assert action.price == 100.0


def test_guard_falls_back_and_records_event_when_not_strict() -> None:
    guard = CallGuard(strict=False, time_limit=1.0)

    result = guard.call(
        role='agent:1',
        method='act',
        fn=lambda: sanitize_agent_action(action='oops', cost=1.0, restricted=False, market=MarketConfig()),
        fallback=NO_PARTICIPATION,
    )

    assert result is NO_PARTICIPATION
    assert guard.drain_events()[0].role == 'agent:1'


def test_guard_reraises_in_strict_mode() -> None:
    guard = CallGuard(strict=True, time_limit=1.0)
    bad_sanction = ArbiterAction(kind=ArbiterActionKind.sanction, target=99)

    with pytest.raises(InvalidActionError):
        guard.call(
            role='arbiter',
            method='act',
            fn=lambda: sanitize_arbiter_action(action=bad_sanction, active={0, 1}, config=LIMITS),
            fallback=ArbiterAction(),
        )


def test_string_price_is_rejected() -> None:
    with pytest.raises(InvalidActionError):
        sanitize_agent_action(action=AgentAction(price='5'), cost=18.0, restricted=False, market=MarketConfig())


def test_bool_coalition_choice_is_rejected() -> None:
    with pytest.raises(InvalidActionError):
        sanitize_agent_action(
            action=AgentAction(price=5.0, coalition_choice=True), cost=18.0, restricted=False, market=MarketConfig()
        )


def test_sanction_with_invalid_kind_is_rejected() -> None:
    with pytest.raises(InvalidActionError):
        sanitize_arbiter_action(action=ArbiterAction(kind='bogus'), active={0, 1}, config=LIMITS)


def test_numpy_scalars_in_agent_action_become_plain_python_values() -> None:
    raw = AgentAction(
        price=np.float32(30.5),
        participate=np.bool_(True),
        coalition_choice=np.int64(2),
        link_action=LinkAction(kind=LinkActionKind.add, target=np.int64(3)),
    )

    action = sanitize_agent_action(action=raw, cost=18.0, restricted=False, market=MarketConfig())

    assert action == AgentAction(
        price=30.5, participate=True, coalition_choice=2, link_action=LinkAction(kind=LinkActionKind.add, target=3)
    )
    assert (type(action.price), type(action.participate), type(action.coalition_choice)) == (float, bool, int)
    assert type(action.link_action.target) is int


def test_numpy_bool_coalition_choice_is_rejected() -> None:
    with pytest.raises(InvalidActionError):
        sanitize_agent_action(
            action=AgentAction(price=5.0, coalition_choice=np.bool_(True)),
            cost=18.0,
            restricted=False,
            market=MarketConfig(),
        )


def test_numpy_sanction_target_is_accepted_as_int() -> None:
    raw = ArbiterAction(kind=ArbiterActionKind.sanction, target=np.int64(1), fine=np.float64(5.0))

    action = sanitize_arbiter_action(action=raw, active={0, 1}, config=LIMITS)

    assert action == ArbiterAction(kind=ArbiterActionKind.sanction, target=1, fine=5.0)
    assert (type(action.target), type(action.fine)) == (int, float)


def test_numpy_link_cost_pairs_become_plain_ints() -> None:
    raw = ArbiterAction(
        kind=ArbiterActionKind.change_link_cost, link_cost_changes={(np.int64(1), np.int64(0)): np.float32(2.0)}
    )

    action = sanitize_arbiter_action(action=raw, active={0, 1}, config=LIMITS)

    [(pair, delta)] = action.link_cost_changes.items()
    assert (pair, delta) == ((0, 1), 2.0)
    assert (type(pair[0]), type(pair[1]), type(delta)) == (int, int, float)


def test_numpy_plan_is_rebuilt_with_plain_types() -> None:
    plan = CoalitionPlan(
        plan_id=np.int64(4),
        members=(np.int64(0), np.int64(1)),
        recommended_prices={np.int64(0): np.float32(60.0), np.int64(1): np.float64(61.0)},
        payout_weights={np.int64(0): np.float64(0.5), np.int64(1): np.float64(0.5)},
    )

    decision = sanitize_coalition_decision(CoalitionDecision(plans=[plan]))

    clean = decision.plans[0]
    assert decision == CoalitionDecision(
        plans=(
            CoalitionPlan(
                plan_id=4, members=(0, 1), recommended_prices={0: 60.0, 1: 61.0}, payout_weights={0: 0.5, 1: 0.5}
            ),
        )
    )
    assert {type(value) for value in (clean.plan_id, *clean.members, *clean.recommended_prices)} == {int}
    assert {type(value) for value in (*clean.recommended_prices.values(), *clean.payout_weights.values())} == {float}


def test_token_fine_below_min_fine_becomes_observe() -> None:
    raw = ArbiterAction(kind=ArbiterActionKind.sanction, target=1, fine=0.5)

    action = sanitize_arbiter_action(action=raw, active={0, 1}, config=ArbiterConfig(min_fine=1.0))

    assert action == ArbiterAction()


def test_huge_link_cost_delta_is_clipped_to_max() -> None:
    raw = ArbiterAction(kind=ArbiterActionKind.change_link_cost, link_cost_changes={(0, 1): 1e308, (0, 2): -1e308})

    action = sanitize_arbiter_action(action=raw, active={0, 1, 2}, config=ArbiterConfig(max_link_cost_delta=5.0))

    assert action.link_cost_changes == {(0, 1): 5.0, (0, 2): -5.0}
