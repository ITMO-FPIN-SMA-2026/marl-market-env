"""Explicit, checkpointed policy settings; no market simulation lives here."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    architecture_version: int = 3
    hidden_size: int = 64
    candidate_hidden_size: int = 32
    learning_rate: float = 0.0003
    critic_learning_rate: float = 0.001
    gamma: float = 0.95
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    reward_scale: float = 100.0
    balance_scale: float = 1000.0
    horizon: int = 50
    price_nodes: int = 9
    max_gradient_norm: float = 1.0
    advantage_clip: float = 20.0
    price_scale_min: float = 0.02
    price_scale_max: float = 0.30

    def __post_init__(self) -> None:
        for name in ('hidden_size', 'candidate_hidden_size', 'horizon', 'price_nodes'):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name == 'price_nodes' else 1):
                raise ValueError(f'{name} must be a positive integer (price_nodes >= 2)')
        if self.architecture_version not in (1, 2, 3):
            raise ValueError('architecture_version must be 1, 2 or 3')
        for name in ('learning_rate', 'critic_learning_rate', 'reward_scale', 'balance_scale', 'max_gradient_norm', 'advantage_clip'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.price_scale_min < self.price_scale_max <= 1:
            raise ValueError('price scale bounds must satisfy 0 < min < max <= 1')
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError('gamma must be in [0, 1]')
        for name in ('entropy_coefficient', 'value_coefficient'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and non-negative')


@dataclass(frozen=True)
class MarketSettings:
    p_min: float = 1.0
    p_max: float = 100.0
    quantity_scale: float = 120.0
    price_offset: float = 5.0

    @classmethod
    def from_mapping(cls, market: dict[str, Any] | None) -> 'MarketSettings':
        values = market or {}
        result = cls(p_min=float(values.get('p_min', 1)), p_max=float(values.get('p_max', 100)),
                     quantity_scale=max(1.0, abs(float(values.get('demand_a', 120)))),
                     price_offset=float(values.get('tau', 5)))
        if not all(math.isfinite(value) for value in (result.p_min, result.p_max, result.quantity_scale, result.price_offset)):
            raise ValueError('market normalization settings must be finite')
        if not 0 < result.p_min < result.p_max or result.price_offset <= 0:
            raise ValueError('invalid market bounds or price offset')
        return result


def writable_path(path: str | Path) -> Path:
    """Keep generated files out of every other part of the shared repository."""
    result = Path(path).expanduser().resolve()
    repository = next((parent for parent in PACKAGE_ROOT.parents if (parent / '.git').exists()), None)
    if repository is not None and result.is_relative_to(repository) and not result.is_relative_to(PACKAGE_ROOT):
        raise ValueError(f'Output inside the repository must be under {PACKAGE_ROOT}')
    return result
