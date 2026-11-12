"""Learning settings and public action limits."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Settings:
    hidden_size: int = 32
    history_window: int = 20
    batch_episodes: int = 8
    epochs: int = 4
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_gradient_norm: float = 0.5
    target_kl: float = 0.03
    fine_amount: float = 25.0
    fine_budget: float = 100.0
    fine_cooldown: int = 5
    min_fine: float = 1.0
    max_fine: float = 100.0
    use_previous_reward: bool = False
    separate_critic: bool = False

    def __post_init__(self):
        integers = ('hidden_size', 'history_window', 'batch_episodes', 'epochs', 'fine_cooldown')
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in vars(self).items():
            if name in ('use_previous_reward', 'separate_critic'):
                if type(value) is not bool:
                    raise ValueError(f'{name} must be a boolean')
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite and numeric')
        for name in ('learning_rate', 'max_gradient_norm', 'target_kl', 'min_fine'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0 < self.clip_ratio < 1:
            raise ValueError('clip_ratio must be in (0, 1)')
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('gamma and gae_lambda must be in [0, 1]')
        if min(self.entropy_coef, self.value_coef) < 0:
            raise ValueError('Loss coefficients must be non-negative')
        if not 0 < self.min_fine <= self.fine_amount <= self.max_fine:
            raise ValueError('Require 0 < min_fine <= fine_amount <= max_fine')
        if self.fine_budget < self.fine_amount:
            raise ValueError('fine_budget must allow at least one full fixed fine')
