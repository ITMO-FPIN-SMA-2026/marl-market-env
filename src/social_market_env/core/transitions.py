"""Delivery of observe_transition once the next observation of the same role exists."""

from dataclasses import dataclass
from typing import Any

from social_market_env.core.guard import CallGuard


@dataclass
class _Pending:
    """A transition waiting for its next observation."""

    learner: Any
    """Component that receives the transition."""
    observation: Any
    """Observation the action was based on."""
    action: Any
    """Action taken."""
    reward: float
    """Reward for the action."""


class TransitionBuffer:
    """Holds each role's last transition until its next observation is built."""

    def __init__(self, *, guard: CallGuard) -> None:
        self._guard = guard
        self._pending: dict[str, _Pending] = {}

    def clear(self) -> None:
        """Forget pending transitions, e.g. on reset."""
        self._pending.clear()

    def hold(self, *, key: str, learner: Any, observation: Any, action: Any, reward: float) -> None:
        """Remember a transition until its next observation is known."""
        self._pending[key] = _Pending(learner=learner, observation=observation, action=action, reward=reward)

    def release(self, *, key: str, next_observation: Any, done: bool) -> None:
        """Deliver the pending transition of a role, if any."""
        pending = self._pending.pop(key, None)
        if pending is None:
            return
        self._guard.call(
            role=key,
            method='observe_transition',
            fn=lambda: pending.learner.observe_transition(
                observation=pending.observation,
                action=pending.action,
                reward=pending.reward,
                next_observation=next_observation,
                done=done,
            ),
            fallback=None,
        )

    def release_all(self) -> None:
        """Deliver every pending transition as terminal."""
        for key in list(self._pending):
            self.release(key=key, next_observation=None, done=True)
