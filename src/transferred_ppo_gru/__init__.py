"""Compatibility entry point for restored PPO + GRU runs.

The canonical package is arbiter_miroshnik, authored by Miroshnik.
"""

__all__ = ["TeamPPOGRU"]


def __getattr__(name):
    if name == "TeamPPOGRU":
        from arbiter_miroshnik import PPOGRUArbiter
        return PPOGRUArbiter
    raise AttributeError(name)
