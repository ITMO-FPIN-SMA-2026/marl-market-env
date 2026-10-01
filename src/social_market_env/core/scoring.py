"""Social welfare, ground-truth harm of coalitions and the arbiter's score."""

from dataclasses import dataclass
from itertools import combinations
from typing import Self

from social_market_env.config import MarketConfig, ScoringConfig
from social_market_env.contracts import ArbiterAction, ArbiterActionKind, Pair
from social_market_env.core.coalitions import PlanCompliance
from social_market_env.core.market import clear_market, consumer_surplus
from social_market_env.gametheory.graphs import as_pair


@dataclass(frozen=True)
class HarmFootprint:
    """Harmful plans of one step and who took part in them."""

    plan_ids: frozenset[int]
    """Plans that lowered welfare below their competitive counterfactual."""
    members: frozenset[int]
    """Complying members of harmful plans."""
    pairs: frozenset[Pair]
    """Pairs of complying members inside the same harmful plan."""

    @classmethod
    def none(cls) -> Self:
        """Footprint of a step without harmful plans."""
        return cls(plan_ids=frozenset(), members=frozenset(), pairs=frozenset())


@dataclass(frozen=True)
class ArbiterVerdict:
    """How the environment judges one arbiter action."""

    intervention: bool
    """Anything but observe."""
    false_positive: bool
    """Fined an agent or raised a link cost outside harmful plans."""
    false_negative: bool
    """A harmful plan existed and was neither sanctioned nor made costlier in an evolving network."""

    @classmethod
    def clean(cls) -> Self:
        """Verdict before the arbiter acts."""
        return cls(intervention=False, false_positive=False, false_negative=False)


def social_welfare(*, total_quantity: float, market: MarketConfig) -> float:
    """Return SW as consumer surplus."""
    return consumer_surplus(total_quantity=total_quantity, market=market)


def find_harm(
    *,
    compliance: list[PlanCompliance],
    prices: dict[int, float],
    costs: dict[int, float],
    nash_prices: dict[int, float],
    market: MarketConfig,
    margin: float,
) -> HarmFootprint:
    """Flag plans whose compliers push welfare below the level they would give by pricing competitively."""
    actual = _welfare_at(prices=prices, costs=costs, market=market)
    harmful = [
        record
        for record in compliance
        if len(record.compliers) >= 2
        and actual < (1.0 - margin) * _counterfactual_welfare(
            record=record, prices=prices, costs=costs, nash_prices=nash_prices, market=market
        )
    ]
    return HarmFootprint(
        plan_ids=frozenset(record.plan_id for record in harmful),
        members=frozenset(member for record in harmful for member in record.compliers),
        pairs=frozenset(as_pair(pair) for record in harmful for pair in combinations(record.compliers, 2)),
    )


def judge_arbiter(
    *,
    action: ArbiterAction,
    harmful_members: frozenset[int],
    harmful_pairs: frozenset[Pair],
    prevention_counts: bool,
) -> ArbiterVerdict:
    """Classify the arbiter action; link cost raises prevent harm only when the network can react to them."""
    is_sanction = action.kind == ArbiterActionKind.sanction
    sanction_hit = is_sanction and action.target in harmful_members
    raised = _raised_pairs(action)
    prevention_hit = prevention_counts and any(pair in harmful_pairs for pair in raised)
    # Raising costs outside harmful plans taxes innocent links, just like fining an innocent agent.
    blanket_raise = any(pair not in harmful_pairs for pair in raised)
    return ArbiterVerdict(
        intervention=action.kind != ArbiterActionKind.observe,
        false_positive=(is_sanction and not sanction_hit) or blanket_raise,
        false_negative=bool(harmful_members) and not (sanction_hit or prevention_hit),
    )


def arbiter_reward(*, welfare: float, target_welfare: float, verdict: ArbiterVerdict, config: ScoringConfig) -> float:
    """Return -alpha*|SW-SW*|/SW* - beta*I - lambda_fp*FP - lambda_fn*FN."""
    gap = abs(welfare - target_welfare) / max(target_welfare, 1e-9)
    return (
        -config.alpha * gap
        - config.beta * verdict.intervention
        - config.lambda_fp * verdict.false_positive
        - config.lambda_fn * verdict.false_negative
    )


def _raised_pairs(action: ArbiterAction) -> list[Pair]:
    if action.kind != ArbiterActionKind.change_link_cost:
        return []
    return [pair for pair, delta in action.link_cost_changes.items() if delta > 0]


def _welfare_at(*, prices: dict[int, float], costs: dict[int, float], market: MarketConfig) -> float:
    outcome = clear_market(prices=prices, costs=costs, market=market)
    return social_welfare(total_quantity=outcome.total_quantity, market=market)


def _counterfactual_welfare(
    *,
    record: PlanCompliance,
    prices: dict[int, float],
    costs: dict[int, float],
    nash_prices: dict[int, float],
    market: MarketConfig,
) -> float:
    competitive = {**prices, **{member: nash_prices[member] for member in record.compliers}}
    return _welfare_at(prices=competitive, costs=costs, market=market)
