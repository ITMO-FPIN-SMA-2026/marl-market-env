"""JSON-friendly record of one step for logs and the viewer."""

from dataclasses import asdict
from typing import Any

from social_market_env.contracts import CoalitionPlan, Pair
from social_market_env.core.coalitions import PlanCompliance
from social_market_env.core.guard import GuardEvent
from social_market_env.core.state import EnvState, StepContext
from social_market_env.gametheory.graphs import graph_matrices

DIGITS: int = 6


def pair_key(pair: Pair) -> str:
    """Return the 'i-j' string used for pairs in logs."""
    return f'{pair[0]}-{pair[1]}'


def build_step_record(*, state: EnvState, context: StepContext, events: list[GuardEvent]) -> dict[str, Any]:
    """Collect everything that happened on the step into plain data."""
    return {
        'step': state.step,
        **_market_part(context),
        **_agents_part(state=state, context=context),
        'edges': {pair_key(pair): round(weight, DIGITS) for pair, weight in context.edges_snapshot.items()},
        'graph_nodes': sorted(state.accounts),
        'graph_matrices': graph_matrices(
            nodes=state.accounts,
            edges=context.edges_snapshot,
            coalitions=(record.compliers + record.defectors for record in context.compliance),
        ),
        'coalition_plan_ids': [plan.plan_id for plan in context.plans],
        'coalition_join_vectors': {
            agent_id: list(memory.coalition_join_vector)
            for agent_id, memory in sorted(state.agent_state.items())
        },
        'link_costs': {
            pair_key(pair): round(cost, DIGITS) for pair, cost in sorted(state.network.link_costs.items()) if cost > 0
        },
        'plans': [
            _plan_record(plan=plan, record=record, harmful=plan.plan_id in context.harm.plan_ids)
            for plan, record in zip(context.plans, context.compliance)
        ],
        'rejected_plans': [asdict(rejection) for rejection in context.rejections],
        'arbiter': _arbiter_part(context),
        'coalition_reward': round(context.coalition_reward, DIGITS),
        'events': [asdict(event) for event in events],
    }


def _rounded(values: dict[int, float]) -> dict[int, float]:
    return {key: round(value, DIGITS) for key, value in sorted(values.items())}


def _market_part(context: StepContext) -> dict[str, Any]:
    outcome = context.outcome
    return {
        'prices': _rounded(outcome.prices),
        'quantities': _rounded(outcome.quantities),
        'gross_profits': _rounded(outcome.gross_profits),
        'average_price': round(outcome.average_price, DIGITS),
        'total_quantity': round(outcome.total_quantity, DIGITS),
        'social_welfare': round(context.welfare, DIGITS),
        'target_social_welfare': round(context.target_welfare, DIGITS),
        'nash_average_price': round(context.nash_average_price, DIGITS),
        'cartel_average_price': round(context.cartel_average_price, DIGITS),
    }


def _agents_part(*, state: EnvState, context: StepContext) -> dict[str, Any]:
    accounts = sorted(state.accounts.items())
    return {
        'received': _rounded(context.received),
        'fines': _rounded(context.fines),
        'link_fees': _rounded(context.link_fees),
        'agent_rewards': _rounded(context.agent_rewards),
        'agent_expectations': {
            agent_id: asdict(memory.expectations)
            for agent_id, memory in sorted(state.agent_state.items())
        },
        'agent_evaluations': {
            agent_id: asdict(memory.evaluations[-1]) if memory.evaluations else None
            for agent_id, memory in sorted(state.agent_state.items())
        },
        'balances': {agent_id: round(account.balance, DIGITS) for agent_id, account in accounts},
        'active': {agent_id: account.active for agent_id, account in accounts},
        'restricted': {agent_id: account.restricted for agent_id, account in accounts},
    }


def _plan_record(*, plan: CoalitionPlan, record: PlanCompliance, harmful: bool) -> dict[str, Any]:
    return {
        'plan_id': plan.plan_id,
        'members': list(plan.members),
        'recommended_prices': _rounded(plan.recommended_prices),
        'payout_weights': _rounded(plan.payout_weights),
        'compliers': list(record.compliers),
        'defectors': list(record.defectors),
        'refusers': list(record.refusers),
        'harmful': harmful,
    }


def _arbiter_part(context: StepContext) -> dict[str, Any]:
    action = context.arbiter_action
    verdict = context.verdict
    return {
        'kind': str(action.kind),
        'target': action.target,
        'fine': round(action.fine, DIGITS),
        'link_cost_changes': {
            pair_key(pair): round(delta, DIGITS) for pair, delta in sorted(action.link_cost_changes.items())
        },
        'intervention': verdict.intervention,
        'false_positive': verdict.false_positive,
        'false_negative': verdict.false_negative,
        'reward': round(context.arbiter_reward, DIGITS),
    }
