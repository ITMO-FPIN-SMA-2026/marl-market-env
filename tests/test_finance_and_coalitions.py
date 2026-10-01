import pytest

from social_market_env.config import MarketConfig
from social_market_env.contracts import AgentAction, CoalitionDecision, CoalitionPlan
from social_market_env.core.coalitions import assess_compliance, cooperation_signals, distribute_pool, validate_plans
from social_market_env.core.finance import Account, settle


def make_plan(*, plan_id: int, members: tuple[int, ...]) -> CoalitionPlan:
    return CoalitionPlan(
        plan_id=plan_id,
        members=members,
        recommended_prices={member: 60.0 for member in members},
        payout_weights={member: 1 / len(members) for member in members},
    )


def test_agent_is_eliminated_after_more_than_k_negative_steps() -> None:
    account = Account(balance=0.0)
    for _ in range(3):
        settle(account=account, amount=-1.0, max_negative_steps=3)

    assert account.active and account.restricted
    settle(account=account, amount=-1.0, max_negative_steps=3)
    assert not account.active


def test_recovery_resets_negative_streak() -> None:
    account = Account(balance=0.0)
    settle(account=account, amount=-1.0, max_negative_steps=3)
    settle(account=account, amount=5.0, max_negative_steps=3)

    assert account.negative_steps == 0 and not account.restricted


def test_overlapping_and_disconnected_plans_are_rejected() -> None:
    decision = CoalitionDecision(
        plans=(
            make_plan(plan_id=0, members=(0, 1)),
            make_plan(plan_id=1, members=(1, 2)),
            make_plan(plan_id=2, members=(2, 3)),
        )
    )

    validation = validate_plans(
        decision=decision, eligible={0, 1, 2, 3}, edges={(0, 1), (1, 2)}, market=MarketConfig()
    )

    assert [plan.plan_id for plan in validation.accepted] == [0]
    assert {rejection.plan_id: rejection.reason for rejection in validation.rejected} == {
        1: 'overlaps another plan',
        2: 'members are not connected in the network',
    }


def test_pool_conserves_money_and_leaves_defectors_their_profit() -> None:
    plan = CoalitionPlan(
        plan_id=0,
        members=(0, 1, 2),
        recommended_prices={0: 60.0, 1: 60.0, 2: 60.0},
        payout_weights={0: 0.5, 1: 0.3, 2: 0.2},
    )
    actions = {
        0: AgentAction(price=60.0, coalition_choice=0),
        1: AgentAction(price=60.2, coalition_choice=0),
        2: AgentAction(price=30.0, coalition_choice=0),
        3: AgentAction(price=25.0),
    }
    profits = {0: 100.0, 1: 50.0, 2: 400.0, 3: 80.0}

    compliance = assess_compliance(
        plans=(plan,), actions=actions, participating={0, 1, 2, 3}, tolerance=0.5
    )
    received = distribute_pool(plans=(plan,), compliance=compliance, profits=profits)

    assert (compliance[0].compliers, compliance[0].defectors) == ((0, 1), (2,))
    assert sum(received.values()) == pytest.approx(sum(profits.values()))
    assert received[0] == pytest.approx(150.0 * 0.5 / 0.8)
    assert received[2] == 400.0
    assert cooperation_signals(compliance) == {(0, 1): 1.0, (0, 2): -1.0, (1, 2): -1.0}
