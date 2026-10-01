"""Reference arbiters."""

from social_market_env.contracts import ArbiterAction, ArbiterActionKind, ArbiterAPI, ArbiterObservation


class AlwaysObserve(ArbiterAPI):
    """Never intervenes."""

    def reset(self, *, seed: int, training: bool) -> None:
        """Nothing to reset."""

    def act(self, observation: ArbiterObservation) -> ArbiterAction:
        """Observe."""
        return ArbiterAction()


class PunishAll(ArbiterAPI):
    """Fines a different active agent every step; the trivial policy the reward must not favor."""

    def __init__(self, *, fine: float = 50.0) -> None:
        self.fine = fine

    def reset(self, *, seed: int, training: bool) -> None:
        """Nothing to reset."""

    def act(self, observation: ArbiterObservation) -> ArbiterAction:
        """Fine agents in turn."""
        if not observation.active_agents:
            return ArbiterAction()
        target = observation.active_agents[observation.step % len(observation.active_agents)]
        return ArbiterAction(kind=ArbiterActionKind.sanction, target=target, fine=self.fine)


class ThresholdArbiter(ArbiterAPI):
    """Fines the most expensive agent after a jump of the average price."""

    def __init__(self, *, jump_threshold: float = 4.0, fine: float = 50.0) -> None:
        self.jump_threshold = jump_threshold
        self.fine = fine

    def reset(self, *, seed: int, training: bool) -> None:
        """Nothing to reset."""

    def act(self, observation: ArbiterObservation) -> ArbiterAction:
        """Sanction on a price jump, observe otherwise."""
        if observation.delta_average_price <= self.jump_threshold or not observation.prices:
            return ArbiterAction()
        target = max(observation.prices, key=lambda agent_id: observation.prices[agent_id])
        return ArbiterAction(kind=ArbiterActionKind.sanction, target=target, fine=self.fine)
