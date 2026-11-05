"""Тесты этапа 2 политики коалиций: исходы, доверие, цена, Оуэн и ядро."""

import numpy as np
import pytest

from coalition_policy import AdaptiveCoalitionPolicy, PolicySettings
from coalition_policy.beliefs import FineLedger, LinkSurvival, TrustTracker
from coalition_policy.payouts import core_excess, owen_table, payout_weights, project_to_core, shapley_table
from coalition_policy.pricing import PriceLadder, future_weight, largest_incentive_share, temptation
from coalition_policy.rewards import PlanResult, StepOutcome, restore_outcome
from coalition_policy.value import ValueOracle
from social_market_env.config import MarketConfig, NetworkConfig, config_from_dict
from social_market_env.contracts import CoalitionDecision, CoalitionObservation, CoalitionPlan, ComplianceRecord
from social_market_env.core.market import clear_market
from social_market_env.gametheory.payouts import owen_values
from social_market_env.runner.experiment import build_environment

POLICY = 'coalition_policy:AdaptiveCoalitionPolicy'
LECTURE_GAME = np.array([0.0, 1.0, 2.0, 5.0, 1.0, 7.0, 6.0, 10.0])
COSTS = {0: 18.0, 1: 18.0, 2: 19.0}


def seen(*, step=0, prices=None, profits=None, history=None, link_costs=None, edges=None) -> CoalitionObservation:
    return CoalitionObservation(
        step=step, active_agents=(0, 1, 2), restricted=(), costs=COSTS, balances={0: 1.0, 1: 1.0, 2: 1.0},
        edges=edges if edges is not None else {(0, 1): 0.0, (1, 2): 0.0}, link_costs=link_costs or {},
        last_prices=prices or {}, last_profits=profits or {}, compliance_history=history or {},
    )


def outcome(
    *, compliers=(), undercutters=(), absent=(), fines=None, raised=(), gap=0.0, step=0, reward=1.0
) -> StepOutcome:
    members = tuple(sorted((*compliers, *undercutters, *absent)))
    plans = (PlanResult(0, members, tuple(compliers), tuple(undercutters), tuple(absent)),) if members else ()
    return StepOutcome(step=step, plans=plans, fines=fines or {}, raised_pairs=frozenset(raised), welfare_gap=gap,
                       reward=reward)


def plan() -> CoalitionPlan:
    return CoalitionPlan(plan_id=7, members=(0, 1), recommended_prices={0: 30.0, 1: 30.0},
                         payout_weights={0: 0.75, 1: 0.25})


def test_owen_matches_environment_library() -> None:
    values = np.random.default_rng(2).uniform(0, 10, size=32)
    values[0] = 0.0
    players = [0, 1, 2, 3, 4]

    def as_function(subset: frozenset[int]) -> float:
        return float(values[sum(1 << player for player in subset)])

    expected = owen_values(members=players, value=as_function, unions=[frozenset({0, 1}), frozenset({2, 3, 4})])

    assert owen_table(values, blocks=[0b00011, 0b11100]) == pytest.approx([expected[p] for p in players])


def test_owen_with_singleton_unions_is_shapley() -> None:
    assert owen_table(LECTURE_GAME, blocks=[1, 2, 4]) == pytest.approx(shapley_table(LECTURE_GAME))


def test_lecture_shapley_is_outside_the_core_and_projection_fixes_it() -> None:
    shapley = shapley_table(LECTURE_GAME)

    fixed = project_to_core(shapley, LECTURE_GAME)

    # Подгруппа {1, 3} стоит 7, а получает 41/6: Шепли здесь не устойчив (план, раздел 3.5).
    assert core_excess(shapley, LECTURE_GAME) == pytest.approx(1 / 6)
    assert core_excess(fixed, LECTURE_GAME) <= 1e-6
    assert fixed.sum() == pytest.approx(10.0)


def test_projection_with_empty_core_keeps_total_and_improves() -> None:
    # У игры большинства из трёх игроков ядро пусто.
    values = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    start = np.array([1.0, 0.0, 0.0])

    fixed = project_to_core(start, values)

    assert fixed.sum() == pytest.approx(1.0)
    assert core_excess(fixed, values) < core_excess(start, values)


def test_insurance_bonus_raises_weight() -> None:
    plain = payout_weights(scheme='shapley', players=[0, 1, 2], values=LECTURE_GAME, edges=())
    insured = payout_weights(scheme='shapley', players=[0, 1, 2], values=LECTURE_GAME, edges=(), bonus={0: 5.0})

    assert insured[0] > plain[0]
    assert sum(insured.values()) == pytest.approx(1.0)


def test_outcome_separates_compliers_undercutters_and_absent() -> None:
    oracle = ValueOracle(market=MarketConfig())
    plans = (CoalitionPlan(plan_id=1, members=(0, 1, 2), recommended_prices={0: 30.0, 1: 30.0, 2: 30.0},
                           payout_weights={0: 0.4, 1: 0.3, 2: 0.3}),)
    prices = {0: 30.0, 1: 25.0}
    history = {0: (ComplianceRecord(4, 1, True),), 1: (ComplianceRecord(4, 1, False),),
               2: (ComplianceRecord(4, 1, False),)}
    gross = clear_market(prices=prices, costs={0: 18.0, 1: 18.0}, market=oracle.market).gross_profits

    result = restore_outcome(
        observation=seen(step=4), decision=CoalitionDecision(plans=plans),
        next_observation=seen(step=5, prices=prices, profits={0: gross[0], 1: gross[1], 2: 0.0}, history=history),
        oracle=oracle, tolerance=0.5,
    )

    assert result.plans[0].compliers == (0,)
    assert result.plans[0].undercutters == (1,)
    assert result.plans[0].absent == (2,)
    assert result.compliance == pytest.approx(1 / 3)
    assert not result.fines


def test_outcome_restores_fines_and_arbiter_traces() -> None:
    oracle = ValueOracle(market=MarketConfig())
    prices = {0: 30.0, 1: 30.0, 2: 24.0}
    gross = clear_market(prices=prices, costs=COSTS, market=oracle.market).gross_profits
    pool = gross[0] + gross[1]
    history = {0: (ComplianceRecord(0, 7, True),), 1: (ComplianceRecord(0, 7, True),)}
    profits = {0: 0.75 * pool - 40.0, 1: 0.25 * pool, 2: gross[2]}

    result = restore_outcome(
        observation=seen(link_costs={(0, 1): 0.0}), decision=CoalitionDecision(plans=(plan(),)),
        next_observation=seen(step=1, prices=prices, profits=profits, history=history, link_costs={(0, 1): 5.0}),
        oracle=oracle, tolerance=0.5,
    )

    assert result.fines == pytest.approx({0: 40.0})
    assert result.raised_pairs == {(0, 1)}
    assert result.arbiter_reacted


def test_last_step_has_no_outcome() -> None:
    oracle = ValueOracle(market=MarketConfig())

    assert restore_outcome(observation=seen(), decision=CoalitionDecision(), next_observation=None, oracle=oracle,
                           tolerance=0.5) is None


def test_trust_falls_after_undercut_and_ban_expires() -> None:
    tracker = TrustTracker(settings=PolicySettings(ban_steps=2, ban_trust=0.5))
    before = tracker.trust(1)

    tracker.update(outcome(compliers=(0,), undercutters=(1,), step=0))
    # Прощающий триггер: разовое нарушение снижает доверие, но не исключает.
    assert tracker.trust(1) < before < tracker.trust(0)
    assert tracker.banned(1) == set()

    for step in (1, 2, 3):
        tracker.update(outcome(undercutters=(1,), step=step))
    assert tracker.banned(4) == {1}
    assert tracker.banned(6) == {1}
    assert tracker.banned(7) == set()


def test_absence_costs_less_trust_than_undercut() -> None:
    tracker = TrustTracker(settings=PolicySettings())

    tracker.update(outcome(undercutters=(0,), absent=(1,)))

    assert tracker.trust(0) < tracker.trust(1)


def test_fine_ledger_pays_debt_only_to_compliers_and_once() -> None:
    ledger = FineLedger(settings=PolicySettings(insurance_rate=0.5))

    ledger.update(outcome(compliers=(0,), undercutters=(1,), fines={0: 40.0, 1: 40.0}))
    paid = ledger.peek_compensation([0, 1])
    ledger.settle(paid)

    assert paid == {0: 20.0}
    assert ledger.peek_compensation([0]) == {0: 10.0}
    assert ledger.expected([0]) > 0


def test_link_survival_falls_with_link_cost_and_ignores_static_network() -> None:
    settings = PolicySettings()
    links = LinkSurvival(config=NetworkConfig(), settings=settings)
    cheap = seen(edges={(0, 1): 1.0, (1, 2): 1.0})
    costly = seen(edges={(0, 1): 1.0, (1, 2): 1.0}, link_costs={(0, 1): 10.0})

    assert links.group_survival(group={0, 1, 2}, observation=cheap) > 0.9
    assert links.group_survival(group={0, 1, 2}, observation=costly) < 0.05
    static = LinkSurvival(config=NetworkConfig(), settings=PolicySettings(network_mode='static'))
    assert static.group_survival(group={0, 1, 2}, observation=costly) == 1.0


def test_survival_uses_the_most_reliable_tree() -> None:
    links = LinkSurvival(config=NetworkConfig(), settings=PolicySettings())
    triangle = seen(edges={(0, 1): 1.0, (1, 2): 1.0, (0, 2): 1.0}, link_costs={(0, 1): 10.0})

    # Дорогое ребро можно обойти через третьего участника.
    assert links.group_survival(group={0, 1, 2}, observation=triangle) > 0.9


def test_ladder_rises_falls_and_remembers_arbiter_ceiling() -> None:
    ladder = PriceLadder(settings=PolicySettings(start_gap=0.02, gap_growth=2.0, partial_growth=1.0, gap_cut=0.5,
                                                 ceiling_margin=0.5))

    ladder.update(outcome(compliers=(0, 1)))
    assert ladder.gap == pytest.approx(0.04)
    # Часть предала, но соблюдавшие заработали выше Нэша: цена держится.
    ladder.update(outcome(compliers=(0,), undercutters=(1,)))
    assert ladder.gap == pytest.approx(0.04)
    # Соблюдавшие заработали меньше Нэша: цена снижается, но не до нуля.
    ladder.update(outcome(compliers=(0,), undercutters=(1,), reward=-1.0))
    assert ladder.gap == pytest.approx(0.02)
    ladder.update(outcome(compliers=(0,), undercutters=(1,), reward=-1.0))
    ladder.update(outcome(compliers=(0,), undercutters=(1,), reward=-1.0))
    assert ladder.gap == pytest.approx(0.01)
    ladder.gap = 0.2
    ladder.update(outcome(compliers=(0, 1), raised=[(0, 1)], gap=0.2))
    assert ladder.ceiling == pytest.approx(0.1)
    assert ladder.gap == pytest.approx(0.05)
    for _ in range(10):
        ladder.update(outcome(compliers=(0, 1)))
    assert ladder.gap == pytest.approx(0.1)


def test_ladder_ignores_steps_without_plans() -> None:
    ladder = PriceLadder(settings=PolicySettings())
    before = ladder.gap

    ladder.update(outcome(raised=[(0, 1)], gap=0.5))

    assert ladder.gap == before


def test_temptation_grows_with_price() -> None:
    market = MarketConfig()
    costs = {agent: 18.5 for agent in range(6)}

    low = temptation(agent=0, prices={agent: 28.0 for agent in costs}, costs=costs, market=market)
    high = temptation(agent=0, prices={agent: 60.0 for agent in costs}, costs=costs, market=market)

    assert high > low


def test_future_weight_shrinks_towards_episode_end() -> None:
    assert future_weight(patience=0.9, steps_left=0) == 0.0
    assert future_weight(patience=0.9, steps_left=1) == pytest.approx(0.9)
    assert future_weight(patience=0.9, steps_left=1000) == pytest.approx(9.0)
    assert future_weight(patience=1.0, steps_left=5) == 5.0


def test_largest_incentive_share_finds_boundary() -> None:
    assert largest_incentive_share(upper=1.0, holds=lambda share: True) == 1.0
    assert largest_incentive_share(upper=1.0, holds=lambda share: share <= 0.3) == pytest.approx(0.3, abs=0.02)
    assert largest_incentive_share(upper=1.0, holds=lambda share: False) == 0.0


def play(*, kwargs=None, agent='social_market_env.baselines.agents:CartelAgent', agent_kwargs=None, overrides=None,
         topology='complete', horizon=30, arbiter=None, seed=0) -> list[dict]:
    data = {
        'horizon': horizon,
        'strict': True,
        'agents': {'count': 8, 'default': {'impl': agent, 'kwargs': agent_kwargs or {}}, 'overrides': overrides or {}},
        'coalitions': {'policy': {'impl': POLICY, 'kwargs': {'horizon': horizon, **(kwargs or {})}}},
        'network': {'topology': topology, 'mode': 'evolving'},
    }
    if arbiter:
        data['arbiter'] = {'policy': arbiter}
    env = build_environment(config=config_from_dict(data), seed=seed)
    env.reset(seed=seed, training=False)
    records = []
    while not env.done:
        records.append(env.step().record)
    return records


def test_ladder_raises_price_with_compliant_agents() -> None:
    records = play()

    assert records[-5]['average_price'] > records[0]['average_price'] + 5


def test_price_falls_towards_episode_end() -> None:
    # При конечном горизонте терпеливый агент теряет всё меньше: цена у конца эпизода падает.
    records = play(kwargs={'patience': 0.5})

    assert records[-1]['average_price'] < max(record['average_price'] for record in records)


def test_undercutter_is_excluded_from_next_plans() -> None:
    records = play(kwargs={'ban_trust': 0.5},
                   overrides={3: {'impl': 'coalition_lab.agents:FlakyAgent', 'kwargs': {'betray_rate': 1.0}}})

    # Постоянный предатель теряет доверие за несколько шагов и выпадает из планов.
    invited = [any(3 in plan['members'] for plan in record['plans']) for record in records]

    assert invited[0]
    assert not all(invited[:10])


def test_memory_off_keeps_inviting_undercutter() -> None:
    records = play(kwargs={'memory': False},
                   overrides={3: {'impl': 'coalition_lab.agents:FlakyAgent', 'kwargs': {'betray_rate': 1.0}}})

    assert all(3 in sum((plan['members'] for plan in record['plans']), []) for record in records[:10])


def test_price_follows_agents_patience() -> None:
    patient = play(agent='coalition_lab.agents:PatientAgent', agent_kwargs={'patience': 0.95})
    impatient = play(agent='coalition_lab.agents:PatientAgent', agent_kwargs={'patience': 0.0})

    def peak(records: list[dict]) -> float:
        return max(record['average_price'] - record['nash_average_price'] for record in records)

    # Близорукие соблюдают только у цены Нэша, где предавать незачем: цена там и останется.
    assert peak(patient) > 3 * peak(impatient)


def test_owen_unions_survive_group_merge() -> None:
    policy = AdaptiveCoalitionPolicy(payout='owen')
    first, second = frozenset({0, 1}), frozenset({2, 3})

    policy._previous = (first, second)
    policy._remember_unions((first | second,))

    assert set(policy._unions[first | second]) == {first, second}


def test_fresh_equilibrium_budget_is_respected() -> None:
    records = play(kwargs={'max_fresh_equilibria': 0}, topology='random')

    assert all(not record['events'] for record in records)


def test_diagnostics_do_not_change_decisions() -> None:
    plain = play(topology='path')
    reported = play(kwargs={'diagnostics': True}, topology='path')

    assert [record['plans'] for record in plain] == [record['plans'] for record in reported]
