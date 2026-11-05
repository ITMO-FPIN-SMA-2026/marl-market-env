"""Тестовые агенты: противники политики коалиций.

Этап 2 плана. CartelAgent из среды всегда соблюдает план, поэтому предательство на нём не проверить.
Это запрограммированное поведение, а не обучение: настоящая проверка — на агентах команды.
"""

import numpy as np

from social_market_env.contracts import AgentAction, AgentObservation, CoalitionOffer
from social_market_env.baselines.agents import BestResponseAgent
from social_market_env.core.market import clear_market
from social_market_env.gametheory.equilibrium import best_response_price, nash_equilibrium


class PatientAgent(BestResponseAgent):
    """Соблюдает план, только если разовый выигрыш предателя не больше будущей потери.

    Рациональный агент из раздела 4.4 плана. patience=0 даёт полностью близорукого агента.
    Предавая, он вступает в план и демпингует; при refuse=True не вступает, как AskarovAgent.
    """

    def __init__(self, *, patience: float = 0.5, refuse: bool = False, peer_cost: float = 18.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.patience = patience
        self.refuse = refuse
        # Чужие себестоимости агент не видит, поэтому считает их средними для рынка.
        # Своя себестоимость вместо этой даёт ошибку больше всего выигрыша от плана.
        self.peer_cost = peer_cost
        self._nash_profit: dict[tuple[float, int], float] = {}

    def act(self, observation: AgentObservation) -> AgentAction:
        """Сравнить выгоду соблюдения и предательства."""
        offer = observation.offer
        if offer is None:
            return super().act(observation)
        response, temptation, payout = self._estimates(observation, offer)
        future = self.patience / (1.0 - self.patience) if self.patience < 1 else float('inf')
        if temptation - payout <= future * (payout - self._nash(observation)):
            return AgentAction(price=offer.recommended_price, coalition_choice=offer.plan_id)
        return AgentAction(price=response, coalition_choice=None if self.refuse else offer.plan_id)

    def _estimates(self, observation: AgentObservation, offer: CoalitionOffer) -> tuple[float, float, float]:
        """Вернуть лучшую цену, прибыль при ней и ожидаемую выплату по плану.

        Цены чужих участников агент не знает, поэтому считает их равными своей рекомендации.
        """
        me = observation.agent_id
        prices = {other: observation.market.last_prices.get(other, offer.recommended_price)
                  for other in observation.link_costs}
        prices.update({member: offer.recommended_price for member in offer.members})
        rivals = np.array([price for other, price in sorted(prices.items()) if other != me])
        response = best_response_price(cost=observation.cost, other_prices=rivals, market=self.market)
        costs = {other: self.peer_cost for other in prices} | {me: observation.cost}
        planned = clear_market(prices=prices, costs=costs, market=self.market).gross_profits
        deviated = clear_market(prices={**prices, me: response}, costs=costs, market=self.market).gross_profits
        pool = sum(planned[member] for member in offer.members)
        return response, deviated[me], offer.payout_weight * pool

    def _nash(self, observation: AgentObservation) -> float:
        count = len(observation.link_costs) + 1
        key = (observation.cost, count)
        if key not in self._nash_profit:
            costs = {index: self.peer_cost for index in range(count)} | {0: observation.cost}
            self._nash_profit[key] = nash_equilibrium(costs=costs, market=self.market).profits[0]
        return self._nash_profit[key]


class FlakyAgent(BestResponseAgent):
    """Случайно предаёт: шум, с которым должна справляться память о доверии."""

    def __init__(self, *, betray_rate: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.betray_rate = betray_rate
        self.rng = np.random.default_rng(0)

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        """Заново посеять случайность."""
        self.rng = np.random.default_rng(seed)

    def act(self, observation: AgentObservation) -> AgentAction:
        """Следовать плану, кроме случайных предательств."""
        offer = observation.offer
        if offer is None:
            return super().act(observation)
        if self.rng.uniform() < self.betray_rate:
            return AgentAction(price=self.best_price(observation), coalition_choice=offer.plan_id)
        return AgentAction(price=offer.recommended_price, coalition_choice=offer.plan_id)
