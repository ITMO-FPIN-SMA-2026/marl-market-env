"""Environment-side reward: private labels never enter an arbiter observation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArbiterScoring:
    target_welfare: float = 1000.0
    alpha: float = 1.0
    beta: float = 0.1
    lambda_fp: float = 0.5
    lambda_fn: float = 0.5
    harm_margin: float = 0.05

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all() or self.target_welfare <= 0:
            raise ValueError("Scoring needs finite values and a positive welfare target")
        if min(self.alpha, self.beta, self.lambda_fp, self.lambda_fn) < 0 or not 0 <= self.harm_margin < 1:
            raise ValueError("Invalid reward weights or harm margin")


def harmful_groups(public, info, margin=0.05):
    return [entry["members"] for entry in info["counterfactuals"]
            if public.mean_price > entry["mean_price"] + 1e-9
            and public.social_welfare < (1 - margin) * entry["social_welfare"]]


def score_arbiter(public, action, info, config):
    """Harm: higher purchase price AND lower welfare than the configured benchmark.

    Outsiders keep their actual prices in this counterfactual. This is a
    Assignment mode uses submitted solo proposals; logit uses the exact grid
    Nash prices. False positives include independent expensive sellers.
    """
    harmful = harmful_groups(public, info, config.harm_margin)
    hit = action.kind == "sanction" and info["fines"].sum() > 0
    false_positive = hit and not any(action.target in group for group in harmful)
    false_negative = any(not (hit and action.target in group) for group in harmful)
    intervention = action.kind != "observe" and action.amount > 0
    reward = (-config.alpha * abs(public.social_welfare - config.target_welfare) / config.target_welfare
              - config.beta * intervention - config.lambda_fp * false_positive - config.lambda_fn * false_negative)
    return float(reward), {"false_positive": bool(false_positive), "false_negative": bool(false_negative),
                           "harmful_coalitions": len(harmful), "sanction": bool(hit)}
