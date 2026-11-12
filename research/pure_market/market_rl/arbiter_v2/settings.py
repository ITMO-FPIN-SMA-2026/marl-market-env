"""Explicit public-only policy contract and optimization settings."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Settings:
    use_recurrence: bool = True
    screening_mode: str = 'calibrated'
    hidden_size: int = 32
    history_window: int = 20
    min_history: int = 5
    fine_amount: float = 25.0
    seller_budget: float = 100.0
    episode_budget: float = 200.0
    cooldown: int = 5
    batch_episodes: int = 8
    epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip_ratio: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    max_gradient_norm: float = .5
    target_kl: float = .03
    initial_fine_probability: float = .1

    def __post_init__(self):
        ints = ('hidden_size', 'history_window', 'min_history', 'cooldown', 'batch_episodes', 'epochs')
        for name in ints:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in vars(self).items():
            if name == 'screening_mode':
                if value not in ('calibrated', 'economic', 'learned'):
                    raise ValueError('screening_mode must be calibrated, economic or learned')
                continue
            if name == 'use_recurrence':
                if type(value) is not bool:
                    raise ValueError('use_recurrence must be a boolean')
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite and numeric')
        if not 3 <= self.min_history <= self.history_window:
            raise ValueError('Require 3 <= min_history <= history_window')
        if not 1 <= self.fine_amount <= 100:
            raise ValueError('fine_amount must be in [1, 100]; align environment limits')
        if min(self.seller_budget, self.episode_budget) < self.fine_amount:
            raise ValueError('Each budget must permit one full fine')
        if min(self.learning_rate, self.max_gradient_norm, self.target_kl, self.value_coef) <= 0:
            raise ValueError('Optimization scales must be positive')
        if self.entropy_coef < 0 or not 0 < self.clip_ratio < 1:
            raise ValueError('Invalid entropy coefficient or clip ratio')
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('Invalid discount or GAE lambda')
        if not 0 < self.initial_fine_probability < 1:
            raise ValueError('initial_fine_probability must be in (0, 1)')
