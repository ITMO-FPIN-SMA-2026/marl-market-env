import pytest

from social_market_env.config import MarketConfig, ScoringConfig
from social_market_env.contracts import ArbiterAction, ArbiterActionKind
from social_market_env.core.coalitions import PlanCompliance
from social_market_env.core.scoring import ArbiterVerdict, arbiter_reward, find_harm, judge_arbiter

MEMBERS = frozenset({0, 1})
PAIRS = frozenset({(0, 1)})


def test_cartel_pricing_is_harmful_and_competition_is_not() -> None:
    costs = {agent_id: 18.0 for agent_id in range(4)}
    competitive = {agent_id: 30.0 for agent_id in range(4)}
    record = PlanCompliance(plan_id=7, compliers=(0, 1), defectors=(), refusers=())
    cartel = {0: 69.0, 1: 69.0, 2: 30.0, 3: 30.0}

    harm = find_harm(
        compliance=[record], prices=cartel, costs=costs, nash_prices=competitive, market=MarketConfig(), margin=0.05
    )
    fair = find_harm(
        compliance=[record],
        prices=competitive,
        costs=costs,
        nash_prices=competitive,
        market=MarketConfig(),
        margin=0.05,
    )

    assert (harm.plan_ids, harm.pairs) == (frozenset({7}), PAIRS)
    assert not fair.plan_ids


@pytest.mark.parametrize(
    ('action', 'prevention_counts', 'expected'),
    [
        (ArbiterAction(kind=ArbiterActionKind.sanction, target=0, fine=10.0), True, (True, False, False)),
        (ArbiterAction(kind=ArbiterActionKind.sanction, target=3, fine=10.0), True, (True, True, True)),
        (
            ArbiterAction(kind=ArbiterActionKind.change_link_cost, link_cost_changes={(0, 1): 1.0}),
            True,
            (True, False, False),
        ),
        (
            ArbiterAction(kind=ArbiterActionKind.change_link_cost, link_cost_changes={(0, 1): 1.0}),
            False,
            (True, False, True),
        ),
        (
            ArbiterAction(kind=ArbiterActionKind.change_link_cost, link_cost_changes={(0, 1): 1.0, (1, 2): 1.0}),
            True,
            (True, True, False),
        ),
        (
            ArbiterAction(kind=ArbiterActionKind.change_link_cost, link_cost_changes={(1, 2): -1.0}),
            True,
            (True, False, True),
        ),
        (ArbiterAction(), True, (False, False, True)),
    ],
    ids=[
        'sanction-harmful',
        'sanction-innocent',
        'raise-harmful-evolving',
        'raise-harmful-static',
        'blanket-raise',
        'lower-innocent',
        'observe',
    ],
)
def test_verdicts(action: ArbiterAction, prevention_counts: bool, expected: tuple[bool, bool, bool]) -> None:
    verdict = judge_arbiter(
        action=action, harmful_members=MEMBERS, harmful_pairs=PAIRS, prevention_counts=prevention_counts
    )

    assert (verdict.intervention, verdict.false_positive, verdict.false_negative) == expected


def test_blanket_raise_without_harm_is_a_false_positive() -> None:
    action = ArbiterAction(kind=ArbiterActionKind.change_link_cost, link_cost_changes={(0, 1): 1e-9, (1, 2): 1e-9})

    verdict = judge_arbiter(
        action=action, harmful_members=frozenset(), harmful_pairs=frozenset(), prevention_counts=True
    )

    assert verdict == ArbiterVerdict(intervention=True, false_positive=True, false_negative=False)


def test_arbiter_reward_penalizes_every_component() -> None:
    verdict = ArbiterVerdict(intervention=True, false_positive=True, false_negative=False)

    reward = arbiter_reward(welfare=90.0, target_welfare=100.0, verdict=verdict, config=ScoringConfig())

    assert reward == pytest.approx(-0.1 - 0.1 - 0.5)
