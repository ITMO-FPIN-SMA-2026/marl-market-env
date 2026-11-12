"""PPO + GRU social arbiter. Author: Miroshnik.

Native component: arbiter_miroshnik:PPOGRUArbiter. See README.md.
"""

__all__ = ["PPOGRUArbiter", "Settings"]


def __getattr__(name):
    if name == "PPOGRUArbiter":
        from .policy import PPOGRUArbiter
        return PPOGRUArbiter
    if name == "Settings":
        from .settings import Settings
        return Settings
    raise AttributeError(name)
