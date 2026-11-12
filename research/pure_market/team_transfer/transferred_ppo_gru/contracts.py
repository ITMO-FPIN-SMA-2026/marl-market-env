"""The original arbiter's public contract, without its old environment."""

from dataclasses import asdict, dataclass
from enum import Enum


@dataclass(frozen=True)
class PublicMarketConfig:
    demand_a: float
    demand_b: float
    p_min: float
    p_max: float
    tau: float = 5.0
    share_model: str = "logit"
    welfare_model: str = "consumer_surplus"
    market_family: str = "team_linear_total"

    @property
    def welfare_scale(self):
        return self.demand_a ** 2 / (2 * self.demand_b)


def public_market(config):
    data = dict(config) if isinstance(config, dict) else asdict(config)
    keys = ("demand_a", "demand_b", "p_min", "p_max", "tau", "share_model")
    if set(data) != set(keys):
        raise ValueError("Expected the native team's MarketConfig")
    return PublicMarketConfig(**{key: data[key] for key in keys})


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
