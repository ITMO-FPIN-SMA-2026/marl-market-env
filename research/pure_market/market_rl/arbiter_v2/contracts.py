"""The original arbiter's public contract, without its old environment."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class PublicMarketConfig:
    demand_a: float
    demand_b: float
    p_min: float
    p_max: float
    share_model: str = "tz_linear_weighted"
    welfare_model: str = "total_surplus"
    welfare_scale: float = 1000.0
    logit_value: float = 8.0
    logit_temperature: float = 1.5
    harm_baseline: str = "proposals"


def public_market(config):
    return PublicMarketConfig(config.demand_a, config.demand_b, min(config.prices), max(config.prices),
        "logit_outside" if config.demand_model == "logit" else "tz_linear_weighted",
        welfare_scale=config.welfare_bound, logit_value=config.logit_value,
        logit_temperature=config.logit_temperature, harm_baseline=config.effective_harm_baseline)


class ArbiterActionKind(str, Enum):
    observe = "observe"
    sanction = "sanction"


@dataclass(frozen=True)
class ArbiterAction:
    kind: ArbiterActionKind = ArbiterActionKind.observe
    target: int | None = None
    fine: float = 0.0


@dataclass(frozen=True)
class ArbiterObservation:
    step: int
    active_agents: tuple
    prices: dict
    quantities: dict
    total_quantity: float
    average_price: float
    social_welfare: float
    delta_average_price: float
    delta_social_welfare: float
