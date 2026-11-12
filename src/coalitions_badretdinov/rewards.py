"""Восстановление исхода шага из следующего наблюдения.

Этап 2 плана. Среда отдаёт политике одно число: сумму дохода соблюдавших по всем планам.
Этого мало, поэтому всё нужное восстанавливается из наблюдения следующего шага (раздел 4.6).

Что нельзя восстановить точно:

- последний шаг эпизода: следующего наблюдения нет, такой шаг пропускается;
- штрафы отделяются от платы за связь только приблизительно: обе суммы вычитаются из дохода,
  а плата агента равна c_ij и видна только косвенно. Поэтому суммы меньше MIN_FINE отбрасываются.
"""

from dataclasses import dataclass

from coalitions_badretdinov.value import ValueOracle
from social_market_env.contracts import CoalitionDecision, CoalitionObservation, CoalitionPlan, Pair
from social_market_env.core.market import clear_market
from social_market_env.core.scoring import social_welfare

MIN_FINE: float = 1.0
REWARD_SCALE: float = 100.0


@dataclass(frozen=True)
class PlanResult:
    """Как участники обошлись с одним планом."""

    plan_id: int
    members: tuple[int, ...]
    compliers: tuple[int, ...]
    """Вступили и держали цену."""
    undercutters: tuple[int, ...]
    """Не соблюли и продали дешевле плана.

    Сюда попадают и предатели, и отказавшиеся. Среда наказывает в памяти H только первых,
    но вред плану одинаковый: настоящий AskarovAgent всегда именно отказывается.
    """
    absent: tuple[int, ...]
    """Не соблюли, но и не сбивали цену: не вышли на рынок или продали не дешевле."""


@dataclass(frozen=True)
class StepOutcome:
    """Всё, что политика узнала о прошлом шаге."""

    step: int
    plans: tuple[PlanResult, ...]
    fines: dict[int, float]
    """Оценка штрафов по агентам."""
    raised_pairs: frozenset[Pair]
    """Пары, стоимость связи которых выросла: след действий арбитра."""
    welfare_gap: float
    """Падение SW относительно конкурентного уровня."""
    reward: float
    """Прирост соблюдавших над Нэшем, в масштабе REWARD_SCALE.

    Сырая награда среды не подходит для обучения: это весь доход соблюдавших,
    поэтому план с прибылью на уровне Нэша выглядел бы успехом.
    """

    @property
    def compliance(self) -> float | None:
        """Доля соблюдавших среди всех участников планов, None если планов не было."""
        total = sum(len(plan.members) for plan in self.plans)
        return sum(len(plan.compliers) for plan in self.plans) / total if total else None

    @property
    def arbiter_reacted(self) -> bool:
        """Арбитр поднял стоимость связей или оштрафовал кого-то."""
        return bool(self.raised_pairs or self.fines)


def restore_outcome(
    *,
    observation: CoalitionObservation,
    decision: CoalitionDecision,
    next_observation: CoalitionObservation | None,
    oracle: ValueOracle,
    tolerance: float,
) -> StepOutcome | None:
    """Вернуть исход шага; None, если следующего наблюдения нет."""
    if next_observation is None:
        return None
    step = observation.step
    prices = next_observation.last_prices
    costs = observation.costs
    market = clear_market(prices=prices, costs={agent: costs[agent] for agent in prices}, market=oracle.market)
    results = tuple(
        _plan_result(plan=plan, step=step, next_observation=next_observation, tolerance=tolerance)
        for plan in decision.plans
    )
    expected = dict(market.gross_profits)
    for plan, result in zip(decision.plans, results):
        expected.update(_pool_shares(plan=plan, compliers=result.compliers, gross=market.gross_profits))
    fines = _fines(expected=expected, received=next_observation.last_profits)
    nash = oracle.nash(costs)
    target = social_welfare(total_quantity=nash.total_quantity, market=oracle.market)
    welfare = social_welfare(total_quantity=market.total_quantity, market=oracle.market)
    gain = sum(
        expected.get(member, 0.0) - fines.get(member, 0.0) - nash.profits[member]
        for result in results
        for member in result.compliers
    )
    return StepOutcome(
        step=step,
        plans=results,
        fines=fines,
        raised_pairs=_raised_pairs(before=observation.link_costs, after=next_observation.link_costs),
        welfare_gap=1.0 - welfare / target if target > 0 else 0.0,
        reward=gain / REWARD_SCALE,
    )


def _plan_result(
    *, plan: CoalitionPlan, step: int, next_observation: CoalitionObservation, tolerance: float
) -> PlanResult:
    compliers: list[int] = []
    undercutters: list[int] = []
    absent: list[int] = []
    for member in sorted(plan.members):
        records = next_observation.compliance_history.get(member, ())
        # Агент входит максимум в один план за шаг, поэтому его запись последняя.
        complied = bool(records) and records[-1].step == step and records[-1].complied
        price = next_observation.last_prices.get(member)
        if complied:
            compliers.append(member)
        elif price is not None and price < plan.recommended_prices[member] - tolerance:
            undercutters.append(member)
        else:
            absent.append(member)
    return PlanResult(plan_id=plan.plan_id, members=tuple(sorted(plan.members)), compliers=tuple(compliers),
                      undercutters=tuple(undercutters), absent=tuple(absent))


def _pool_shares(*, plan: CoalitionPlan, compliers: tuple[int, ...], gross: dict[int, float]) -> dict[int, float]:
    """Повторить деление котла средой: доли нормируются по тем, кто соблюдал."""
    if not compliers:
        return {}
    pool = sum(gross[member] for member in compliers)
    total = sum(plan.payout_weights[member] for member in compliers)
    if total <= 0:
        return {member: pool / len(compliers) for member in compliers}
    return {member: pool * plan.payout_weights[member] / total for member in compliers}


def _fines(*, expected: dict[int, float], received: dict[int, float]) -> dict[int, float]:
    gaps = {agent: expected[agent] - received[agent] for agent in expected if agent in received}
    return {agent: gap for agent, gap in gaps.items() if gap >= MIN_FINE}


def _raised_pairs(*, before: dict[Pair, float], after: dict[Pair, float]) -> frozenset[Pair]:
    return frozenset(pair for pair, cost in after.items() if cost > before.get(pair, 0.0) + 1e-9)
