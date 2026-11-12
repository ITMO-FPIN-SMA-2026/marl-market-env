"""Small public-API fixtures for controlled experiments, never the learned policy."""

from social_market_env.config import MarketConfig
from social_market_env.contracts import (AgentAction, CoalitionDecision, CoalitionPlan,
                                         CoalitionPolicyAPI, MarketAgentAPI)


class MarkupStrategy(MarketAgentAPI):
    """Separate non-learning cost-plus-temperature comparison policy."""

    def __init__(self, *, market=None):
        self.market = MarketConfig(**(market or {}))

    def reset(self, *, agent_id, seed, training):
        pass

    def act(self, observation):
        price = min(self.market.p_max, max(self.market.p_min, observation.cost + self.market.tau))
        return AgentAction(price=price, participate=observation.active and observation.cost <= self.market.p_max)


class PriceWave(MarketAgentAPI):
    """Exogenous alternating rival prices; schedule is never passed to our agent."""

    def __init__(self, *, market=None, rising=True, period=20, phase=0):
        self.market = MarketConfig(**(market or {}))
        self.rising, self.period, self.phase = rising, period, phase

    def reset(self, *, agent_id, seed, training):
        self.offset = self.phase % (2 * self.period)

    def act(self, observation):
        phase = (observation.step + self.offset) % (2 * self.period)
        fraction = phase / self.period if phase <= self.period else 2 - phase / self.period
        if not self.rising:
            fraction = 1 - fraction
        price = 22 + 38 * fraction
        return AgentAction(price=min(self.market.p_max, max(observation.cost, price)), participate=observation.active)


class PairOffers(CoalitionPolicyAPI):
    """Connected pairs with favorable/unfavorable payout weights for learning tests.

    Every action still goes through the normal environment's compliance, pooling,
    and reward calculation. There is no forced acceptance or artificial reward.
    """

    def __init__(self, *, market=None, favored=True):
        self.market = MarketConfig(**(market or {}))
        self.favored = favored

    def reset(self, *, seed, training):
        self.offset = seed % 2

    def act(self, observation):
        remaining = set(observation.active_agents) - set(observation.restricted)
        plans = []
        for first, second in sorted(observation.edges):
            if first not in remaining or second not in remaining:
                continue
            remaining.difference_update((first, second))
            leader = first if (observation.step // 10 + self.offset) % 2 == 0 else second
            if not self.favored:
                leader = second if leader == first else first
            prices = {i: min(self.market.p_max, max(self.market.p_min, observation.costs[i] + self.market.tau))
                      for i in (first, second)}
            plans.append(CoalitionPlan(plan_id=len(plans), members=(first, second), recommended_prices=prices,
                                       payout_weights={i: 0.85 if i == leader else 0.15 for i in (first, second)}))
        return CoalitionDecision(plans=tuple(plans))
