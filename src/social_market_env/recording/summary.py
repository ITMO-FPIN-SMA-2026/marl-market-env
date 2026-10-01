"""Aggregates of one episode for run comparison."""

from statistics import mean
from typing import Any

DIGITS: int = 6


def summarize_episode(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return totals and averages over the step records of an episode."""
    if not records:
        return {'steps': 0}
    arbiter = [record['arbiter'] for record in records]
    gaps = [
        abs(record['social_welfare'] - record['target_social_welfare']) / max(record['target_social_welfare'], 1e-9)
        for record in records
    ]
    return {
        'steps': len(records),
        'total_agent_reward': round(sum(sum(record['agent_rewards'].values()) for record in records), DIGITS),
        'mean_social_welfare': round(mean(record['social_welfare'] for record in records), DIGITS),
        'mean_welfare_gap': round(mean(gaps), DIGITS),
        'total_arbiter_reward': round(sum(item['reward'] for item in arbiter), DIGITS),
        'interventions': sum(item['intervention'] for item in arbiter),
        'false_positives': sum(item['false_positive'] for item in arbiter),
        'false_negatives': sum(item['false_negative'] for item in arbiter),
        'mean_plans': round(mean(len(record['plans']) for record in records), DIGITS),
        'defections': sum(len(plan['defectors']) for record in records for plan in record['plans']),
        'eliminated': sum(not active for active in records[-1]['active'].values()),
    }
