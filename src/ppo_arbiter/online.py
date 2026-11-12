"""Persistent PPO that updates after every completed training game."""

from dataclasses import replace
from pathlib import Path

from .policy import PPOArbiter


class OnlinePPOArbiter(PPOArbiter):
    """Resume a local state, learn once per game, and save after the update.

    A missing state starts from warm_start_path, or from random weights when no
    warm start is supplied. Eval remains frozen and never writes a checkpoint.
    The historical PPO source and selected checkpoint are kept unchanged.
    """

    def __init__(self, *, state_path='runs/ppo_arbiter/online.pt',
                 warm_start_path=None, market=None, settings=None, model_seed=0,
                 evaluation='greedy'):
        state_path = Path(state_path)
        checkpoint = state_path if state_path.exists() else warm_start_path
        super().__init__(market=market, settings=settings, model_seed=model_seed,
                         checkpoint_path=checkpoint, checkpoint_out=state_path,
                         evaluation=evaluation)
        # Only the update interval changes; weights, optimizer, architecture,
        # rewards and limits are restored by the original checkpoint loader.
        self.settings = replace(self.settings, batch_episodes=1)
        self.memory.settings = self.settings
