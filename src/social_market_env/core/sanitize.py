"""Validation of everything team code returns to the environment."""

import math
import numbers
from enum import Enum
from typing import TypeVar

import numpy as np

from social_market_env.config import ArbiterConfig, MarketConfig
from social_market_env.contracts import (
    AgentAction,
    ArbiterAction,
    ArbiterActionKind,
    CoalitionDecision,
    CoalitionPlan,
    LinkAction,
    LinkActionKind,
    Pair,
)
from social_market_env.gametheory.graphs import as_pair

EnumT = TypeVar('EnumT', bound=Enum)

NO_PARTICIPATION: AgentAction = AgentAction(price=0.0, participate=False)


class InvalidActionError(ValueError):
    """Raised when team code returns something the environment cannot use."""


def sanitize_agent_action(*, action: object, cost: float, restricted: bool, market: MarketConfig) -> AgentAction:
    """Check the type, clip the price, apply restrictions of a negative reserve and coerce numpy scalars."""
    if not isinstance(action, AgentAction):
        raise InvalidActionError(f'expected AgentAction, got {type(action).__name__}')
    link_action = _sanitize_link_action(action.link_action)
    choice = _sanitize_choice(action.coalition_choice)
    if not isinstance(action.participate, bool | np.bool_):
        raise InvalidActionError(f'participate must be a bool, got {action.participate!r}')
    price = _finite(value=action.price, name='price')
    floor = max(market.p_min, cost) if restricted else market.p_min
    return AgentAction(
        price=min(market.p_max, max(floor, price)),
        participate=bool(action.participate),
        coalition_choice=None if restricted else choice,
        link_action=link_action,
    )


def sanitize_coalition_decision(decision: object) -> CoalitionDecision:
    """Check that the decision and all its plans have the right types and rebuild them with plain numbers."""
    if not isinstance(decision, CoalitionDecision):
        raise InvalidActionError(f'expected CoalitionDecision, got {type(decision).__name__}')
    if not all(isinstance(plan, CoalitionPlan) for plan in decision.plans):
        raise InvalidActionError('every plan must be a CoalitionPlan')
    return CoalitionDecision(plans=tuple(_sanitize_plan(plan) for plan in decision.plans))


def sanitize_arbiter_action(*, action: object, active: set[int], config: ArbiterConfig) -> ArbiterAction:
    """Check the type, the sanction target and link pairs; clip the fine and the link cost deltas."""
    if not isinstance(action, ArbiterAction):
        raise InvalidActionError(f'expected ArbiterAction, got {type(action).__name__}')
    kind = _parse_kind(kind_enum=ArbiterActionKind, value=action.kind)
    if kind == ArbiterActionKind.sanction:
        return _sanitize_sanction(action=action, active=active, config=config)
    if kind == ArbiterActionKind.change_link_cost:
        changes = _sanitize_link_costs(
            changes=action.link_cost_changes, active=active, max_delta=config.max_link_cost_delta
        )
        return ArbiterAction(kind=kind, link_cost_changes=changes)
    return ArbiterAction()


def _sanitize_sanction(*, action: ArbiterAction, active: set[int], config: ArbiterConfig) -> ArbiterAction:
    if not _is_id(action.target) or action.target not in active:
        raise InvalidActionError(f'sanction target {action.target!r} is not an active agent')
    fine = min(config.max_fine, max(0.0, _finite(value=action.fine, name='fine')))
    # A token fine would clear a false negative without costing the cartel anything.
    if fine < config.min_fine:
        return ArbiterAction()
    return ArbiterAction(kind=ArbiterActionKind.sanction, target=int(action.target), fine=fine)


def _sanitize_plan(plan: CoalitionPlan) -> CoalitionPlan:
    ids = [plan.plan_id, *plan.members, *plan.recommended_prices, *plan.payout_weights]
    numbers_in_plan = [*plan.recommended_prices.values(), *plan.payout_weights.values()]
    if not all(_is_id(value) for value in ids):
        raise InvalidActionError(f'plan {plan.plan_id!r} must use integer ids')
    if not all(_is_number(value) and math.isfinite(float(value)) for value in numbers_in_plan):
        raise InvalidActionError(f'plan {plan.plan_id!r} has non-numeric prices or weights')
    return CoalitionPlan(
        plan_id=int(plan.plan_id),
        members=tuple(int(member) for member in plan.members),
        recommended_prices={int(key): float(value) for key, value in plan.recommended_prices.items()},
        payout_weights={int(key): float(value) for key, value in plan.payout_weights.items()},
    )


def _sanitize_link_action(link_action: object) -> LinkAction | None:
    if link_action is None:
        return None
    if not isinstance(link_action, LinkAction) or not _is_id(link_action.target):
        raise InvalidActionError('link_action must be a LinkAction with an integer target')
    kind = _parse_kind(kind_enum=LinkActionKind, value=link_action.kind)
    return LinkAction(kind=kind, target=int(link_action.target))


def _sanitize_choice(choice: object) -> int | None:
    if choice is None:
        return None
    if not _is_id(choice):
        raise InvalidActionError(f'coalition_choice must be an int or None, got {choice!r}')
    return int(choice)


def _parse_kind(*, kind_enum: type[EnumT], value: object) -> EnumT:
    try:
        return kind_enum(value)
    except ValueError as error:
        raise InvalidActionError(f'invalid {kind_enum.__name__} value {value!r}') from error


def _is_id(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool | np.bool_)


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def _sanitize_link_costs(*, changes: dict[Pair, float], active: set[int], max_delta: float) -> dict[Pair, float]:
    result: dict[Pair, float] = {}
    for (first, second), delta in changes.items():
        if not (_is_id(first) and _is_id(second)):
            raise InvalidActionError(f'link cost pair ({first!r}, {second!r}) must use integer ids')
        if first != second and first in active and second in active:
            clipped = min(max_delta, max(-max_delta, _finite(value=delta, name='link cost delta')))
            result[as_pair((int(first), int(second)))] = clipped
    return result


def _finite(*, value: object, name: str) -> float:
    if not _is_number(value):
        raise InvalidActionError(f'{name} must be a number, got {value!r}')
    number = float(value)
    if not math.isfinite(number):
        raise InvalidActionError(f'{name} must be finite')
    return number
