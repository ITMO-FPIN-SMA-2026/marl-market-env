"""Reference coalition policies."""

from typing import Any

from social_market_env.config import MarketConfig
from social_market_env.contracts import CoalitionDecision, CoalitionObservation, CoalitionPlan, CoalitionPolicyAPI
from social_market_env.gametheory.equilibrium import CharacteristicFunction, coalition_equilibrium
from social_market_env.gametheory.graphs import connected_components
from social_market_env.gametheory.payouts import equal_split, myerson_values, shapley_values, to_weights

PAYOUT_SCHEMES: frozenset[str] = frozenset({'equal', 'shapley', 'myerson'})


class NoCoalitions(CoalitionPolicyAPI):
    """Never proposes a plan."""

    def reset(self, *, seed: int, training: bool) -> None:
        """Nothing to reset."""

    def act(self, observation: CoalitionObservation) -> CoalitionDecision:
        """Propose nothing."""
        return CoalitionDecision()


class ConnectedComponentsCoalitions(CoalitionPolicyAPI):
    """Turns every connected group of eligible agents into a cartel with joint-profit member prices."""

    def __init__(self, *, market: dict[str, Any] | None = None, payout: str = 'equal') -> None:
        if payout not in PAYOUT_SCHEMES:
            raise ValueError(f'payout must be one of {sorted(PAYOUT_SCHEMES)}')
        self.market = MarketConfig(**(market or {}))
        self.payout = payout
        self._prices: dict[tuple[frozenset[int], frozenset[tuple[int, float]]], dict[int, float]] = {}
        self._values: dict[frozenset[tuple[int, float]], CharacteristicFunction] = {}

    def reset(self, *, seed: int, training: bool) -> None:
        """Nothing to reset; caches are keyed on cost values, so they stay valid across episodes."""

    def act(self, observation: CoalitionObservation) -> CoalitionDecision:
        """Propose one plan per connected group of at least two eligible agents."""
        eligible = [agent_id for agent_id in observation.active_agents if agent_id not in observation.restricted]
        groups = [group for group in connected_components(nodes=eligible, edges=observation.edges) if len(group) >= 2]
        plans = tuple(
            self._plan(plan_id=index, members=group, observation=observation) for index, group in enumerate(groups)
        )
        return CoalitionDecision(plans=plans)

    def _plan(self, *, plan_id: int, members: frozenset[int], observation: CoalitionObservation) -> CoalitionPlan:
        return CoalitionPlan(
            plan_id=plan_id,
            members=tuple(sorted(members)),
            recommended_prices=self._cartel_prices(members=members, costs=observation.costs),
            payout_weights=self._weights(members=members, observation=observation),
        )

    def _cartel_prices(self, *, members: frozenset[int], costs: dict[int, float]) -> dict[int, float]:
        key = (members, frozenset(costs.items()))
        if key not in self._prices:
            equilibrium = coalition_equilibrium(members=members, costs=costs, market=self.market)
            self._prices[key] = {member: equilibrium.prices[member] for member in members}
        return dict(self._prices[key])

    def _weights(self, *, members: frozenset[int], observation: CoalitionObservation) -> dict[int, float]:
        if self.payout == 'equal':
            return equal_split(members)
        costs_key = frozenset(observation.costs.items())
        if costs_key not in self._values:
            self._values[costs_key] = CharacteristicFunction(costs=observation.costs, market=self.market)
        value = self._values[costs_key]
        if self.payout == 'myerson':
            return to_weights(myerson_values(members=sorted(members), value=value, edges=set(observation.edges)))
        return to_weights(shapley_values(members=sorted(members), value=value))
