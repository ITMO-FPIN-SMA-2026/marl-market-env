"""Safe calls into team code: failures become logged fallbacks unless the run is strict."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass(frozen=True)
class GuardEvent:
    """A failed call into team code."""

    role: str
    """Who failed, e.g. 'agent:3', 'coalition', 'arbiter'."""
    method: str
    """Which method failed."""
    error: str
    """repr() of the exception."""


class CallGuard:
    """Calls team code, turning crashes into fallbacks and counting slow calls."""

    def __init__(self, *, strict: bool, time_limit: float) -> None:
        self.strict = strict
        self.time_limit = time_limit
        self.events: list[GuardEvent] = []
        self.slow_calls: dict[str, int] = {}

    def call(self, *, role: str, method: str, fn: Callable[[], T], fallback: T) -> T:
        """Run fn and return its result, or the fallback if it raises and the guard is not strict."""
        started = time.perf_counter()
        try:
            result = fn()
        # Team code is a black box: any exception it raises must not stop the episode in non-strict runs.
        except Exception as error:
            if self.strict:
                raise
            logger.warning('Team code failed in %s.%s: %r', role, method, error)
            self.events.append(GuardEvent(role=role, method=method, error=repr(error)))
            return fallback
        if time.perf_counter() - started > self.time_limit:
            key = f'{role}.{method}'
            self.slow_calls[key] = self.slow_calls.get(key, 0) + 1
        return result

    def drain_events(self) -> list[GuardEvent]:
        """Return and forget the events collected since the last drain."""
        drained, self.events = self.events, []
        return drained
