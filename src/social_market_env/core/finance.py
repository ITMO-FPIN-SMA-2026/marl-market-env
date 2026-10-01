"""Financial reserve: restriction while negative, elimination after a long negative streak."""

from dataclasses import dataclass


@dataclass
class Account:
    """Reserve and status of one agent."""

    balance: float
    """Accumulated reserve B."""
    active: bool = True
    """False once the agent is eliminated."""
    negative_steps: int = 0
    """Consecutive steps ending with a negative reserve."""

    @property
    def restricted(self) -> bool:
        """True while an active agent's reserve is negative."""
        return self.active and self.balance < 0


def settle(*, account: Account, amount: float, max_negative_steps: int) -> None:
    """Add the step result to the reserve and eliminate after too many negative steps."""
    if not account.active:
        return
    account.balance += amount
    account.negative_steps = account.negative_steps + 1 if account.balance < 0 else 0
    if account.negative_steps > max_negative_steps:
        account.active = False
