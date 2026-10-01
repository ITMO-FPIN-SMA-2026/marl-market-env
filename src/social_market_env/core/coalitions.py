"""Coalition plans: validation, offers, compliance and the shared pool."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from social_market_env.config import MarketConfig
from social_market_env.contracts import AgentAction, CoalitionDecision, CoalitionOffer, CoalitionPlan, Pair
from social_market_env.gametheory.graphs import as_pair, is_connected

WEIGHT_TOLERANCE: float = 1e-6


@dataclass(frozen=True)
class PlanRejection:
    """Why a proposed plan was not accepted."""

    plan_id: int
    """Rejected plan."""
    reason: str
    """Human-readable reason for the log."""


@dataclass(frozen=True)
class PlanValidation:
    """Accepted and rejected plans of one decision."""

    accepted: tuple[CoalitionPlan, ...]
    """Plans that take effect on this step."""
    rejected: tuple[PlanRejection, ...]
    """Plans that were dropped."""


@dataclass(frozen=True)
class PlanCompliance:
    """How the members treated one plan."""

    plan_id: int
    """Plan identifier."""
    compliers: tuple[int, ...]
    """Joined and kept the recommended price."""
    defectors: tuple[int, ...]
    """Joined but set a different price."""
    refusers: tuple[int, ...]
    """Did not join or did not sell."""


def validate_plans(
    *, decision: CoalitionDecision, eligible: set[int], edges: set[Pair], market: MarketConfig
) -> PlanValidation:
    """Accept plans in order, rejecting invalid or overlapping ones."""
    accepted: list[CoalitionPlan] = []
    rejected: list[PlanRejection] = []
    used: set[int] = set()
    seen_ids: set[int] = set()
    for plan in decision.plans:
        reason = _rejection_reason(plan=plan, eligible=eligible, edges=edges, used=used,
                                   seen_ids=seen_ids, market=market)
        seen_ids.add(plan.plan_id)
        if reason is None:
            accepted.append(plan)
            used |= set(plan.members)
        else:
            rejected.append(PlanRejection(plan_id=plan.plan_id, reason=reason))
    return PlanValidation(accepted=tuple(accepted), rejected=tuple(rejected))


def build_offers(plans: Sequence[CoalitionPlan]) -> dict[int, CoalitionOffer]:
    """Return the offer every plan member sees."""
    return {
        member: CoalitionOffer(
            plan_id=plan.plan_id,
            members=tuple(sorted(plan.members)),
            recommended_price=plan.recommended_prices[member],
            payout_weight=plan.payout_weights[member],
        )
        for plan in plans
        for member in plan.members
    }


def assess_compliance(
    *, plans: Sequence[CoalitionPlan], actions: dict[int, AgentAction], participating: set[int], tolerance: float
) -> list[PlanCompliance]:
    """Split each plan's members into compliers, defectors and refusers."""
    return [_assess_plan(plan=plan, actions=actions, participating=participating, tolerance=tolerance)
            for plan in plans]


def distribute_pool(
    *, plans: Sequence[CoalitionPlan], compliance: list[PlanCompliance], profits: dict[int, float]
) -> dict[int, float]:
    """Return what every participant receives: compliers share their pool, others keep their profit."""
    received = dict(profits)
    by_plan = {record.plan_id: record for record in compliance}
    for plan in plans:
        received.update(_pool_payouts(plan=plan, compliers=by_plan[plan.plan_id].compliers, profits=profits))
    return received


def cooperation_signals(compliance: list[PlanCompliance]) -> dict[Pair, float]:
    """Return +1 for pairs that complied together and -1 for pairs where one betrayed the other."""
    signals: dict[Pair, float] = {}
    for record in compliance:
        signals.update({as_pair(pair): 1.0 for pair in combinations(record.compliers, 2)})
        joined = record.compliers + record.defectors
        signals.update({as_pair((defector, other)): -1.0 for defector in record.defectors for other in joined
                        if other != defector})
    return signals


def _rejection_reason(
    *, plan: CoalitionPlan, eligible: set[int], edges: set[Pair], used: set[int], seen_ids: set[int],
    market: MarketConfig
) -> str | None:
    members = set(plan.members)
    weights = plan.payout_weights
    checks = [
        (plan.plan_id in seen_ids, 'duplicate plan id'),
        (len(members) < 2 or len(members) != len(plan.members), 'needs at least two distinct members'),
        (not members <= eligible, 'member is inactive or restricted'),
        (bool(members & used), 'overlaps another plan'),
        (set(plan.recommended_prices) != members or set(weights) != members,
         'prices or weights do not match members'),
        (any(weight < 0 for weight in weights.values()) or
         abs(sum(weights.values()) - 1.0) > WEIGHT_TOLERANCE,
         'weights must be non-negative and sum to 1'),
        (any(not market.p_min <= price <= market.p_max for price in plan.recommended_prices.values()),
         'recommended price out of bounds'),
    ]
    for failed, reason in checks:
        if failed:
            return reason
    return None if is_connected(nodes=members, edges=edges) else 'members are not connected in the network'


def _assess_plan(
    *, plan: CoalitionPlan, actions: dict[int, AgentAction], participating: set[int], tolerance: float
) -> PlanCompliance:
    compliers: list[int] = []
    defectors: list[int] = []
    refusers: list[int] = []
    for member in sorted(plan.members):
        action = actions.get(member)
        if action is None or member not in participating or action.coalition_choice != plan.plan_id:
            refusers.append(member)
        elif abs(action.price - plan.recommended_prices[member]) <= tolerance:
            compliers.append(member)
        else:
            defectors.append(member)
    return PlanCompliance(plan_id=plan.plan_id, compliers=tuple(compliers), defectors=tuple(defectors),
                          refusers=tuple(refusers))


def _pool_payouts(
    *, plan: CoalitionPlan, compliers: tuple[int, ...], profits: dict[int, float]
) -> dict[int, float]:
    if not compliers:
        return {}
    pool = sum(profits[member] for member in compliers)
    weights = {member: plan.payout_weights[member] for member in compliers}
    total = sum(weights.values())
    if total <= 0:
        return {member: pool / len(compliers) for member in compliers}
    return {member: pool * weights[member] / total for member in compliers}
