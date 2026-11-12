"""Transferred original PPO/GRU, as an independent native-environment plugin."""

__all__ = ["TeamPPOGRU"]


def __getattr__(name):
    if name == "TeamPPOGRU":
        from .api import TeamPPOGRU
        return TeamPPOGRU
    raise AttributeError(name)
