"""One-step semi-gradient TD actor-critic; reward is supplied by the environment."""

import numpy as np
import torch

from .actions import Candidates
from .policy import ActorCritic, Choice
from .settings import Settings


def update(model: ActorCritic, optimizer: torch.optim.Optimizer, *, critic_optimizer: torch.optim.Optimizer, settings: Settings,
           features: np.ndarray, candidates: Candidates, choice: Choice, reward: float,
           next_features: np.ndarray | None, done: bool) -> dict[str, float]:
    log_probability, entropy, value = model.evaluate_choice(features, candidates, choice)
    with torch.no_grad():
        bootstrap = torch.zeros(()) if done else model.state(next_features)[1]
        target = torch.tensor(reward / settings.reward_scale, dtype=torch.float32) + settings.gamma * bootstrap
    delta = target - value
    advantage = delta.detach().clamp(-settings.advantage_clip, settings.advantage_clip)
    actor_loss = -log_probability * advantage
    critic_loss = delta.square()  # Report MSE for comparability with version 1.
    critic_objective = torch.nn.functional.smooth_l1_loss(value,target)
    actor_objective = actor_loss-settings.entropy_coefficient*entropy
    loss = actor_objective + settings.value_coefficient * critic_objective
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite actor-critic loss')
    optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    actor_objective.backward()
    (settings.value_coefficient*critic_objective).backward()
    actor_norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), settings.max_gradient_norm, error_if_nonfinite=True)
    critic_norm = torch.nn.utils.clip_grad_norm_(model.critic_parameters(), settings.max_gradient_norm, error_if_nonfinite=True)
    optimizer.step()
    critic_optimizer.step()
    return {'loss': float(loss.detach()), 'actor_loss': float(actor_loss.detach()),
            'critic_loss': float(critic_loss.detach()), 'entropy': float(entropy.detach()),
            'td_error': float(delta.detach()), 'target': float(target), 'value': float(value.detach()),
            'critic_objective': float(critic_objective.detach()),
            'gradient_norm': float(torch.sqrt(actor_norm.square()+critic_norm.square())),
            'actor_gradient_norm': float(actor_norm), 'critic_gradient_norm': float(critic_norm),
            'actor_clip_multiplier': min(1.0,settings.max_gradient_norm/(float(actor_norm)+1e-6)),
            'critic_clip_multiplier': min(1.0,settings.max_gradient_norm/(float(critic_norm)+1e-6))}
