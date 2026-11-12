"""Schema-4 market metadata compatibility using the environment's MarketConfig."""

from dataclasses import asdict

from social_market_env.config import MarketConfig

_MARKET_METADATA = {
    "welfare_model": "consumer_surplus",
    "market_family": "team_linear_total",
}


def market_config(data):
    if isinstance(data, MarketConfig):
        return data
    data = dict(data)
    for key, expected in _MARKET_METADATA.items():
        if data.pop(key, expected) != expected:
            raise ValueError("Checkpoint uses an incompatible market model")
    return MarketConfig(**data)


def market_payload(market):
    return asdict(market) | _MARKET_METADATA
