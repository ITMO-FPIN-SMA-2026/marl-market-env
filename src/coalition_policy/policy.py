"""AdaptiveCoalitionPolicy: фасад контракта CoalitionPolicyAPI.

Конвейер шага (план, раздел 4.1). Шаги, помеченные этапом, ещё не реализованы.

    1. Убеждения: доверие, связи, давление арбитра   — этап 2
    2. Кандидаты                                     — candidates.py
    3. Стратегические рычаги                         — этап 3, сейчас настройки
    4. Выбор разбиения                               — structure.py
    5. Цены                                          — pricing.py
    6. Доли котла                                    — payouts.py
    7. Восстановленная награда и обучение            — этапы 2 и 3
"""

from typing import Any

from coalition_policy.candidates import eligible_agents, generate_candidates
from coalition_policy.payouts import level_game, payout_weights
from coalition_policy.pricing import blended_prices, mean_price_cap
from coalition_policy.settings import PolicySettings
from coalition_policy.structure import select_partition
from coalition_policy.value import Group, Partition, ValueOracle
from social_market_env.config import MarketConfig
from social_market_env.contracts import CoalitionDecision, CoalitionObservation, CoalitionPlan, CoalitionPolicyAPI
from social_market_env.core.coalitions import validate_plans
from social_market_env.core.scoring import social_welfare
from social_market_env.gametheory.payouts import equal_split


class AdaptiveCoalitionPolicy(CoalitionPolicyAPI):
    """Предлагает связные коалиции с удерживаемой ценой и долями по вкладу."""

    def __init__(self, *, market: dict[str, Any] | None = None, **settings: Any) -> None:
        # Секцию market среда подставит сама, остальные ключи приходят из kwargs в YAML.
        self.market = MarketConfig(**(market or {}))
        self.settings = PolicySettings(**settings)
        # Оракул живёт дольше эпизода: себестоимости внутри seed те же, и кэш остаётся полезным.
        self.oracle = ValueOracle(market=self.market)
        self.reset(seed=0, training=False)

    def reset(self, *, seed: int, training: bool) -> None:
        """Забыть память эпизода; обученные параметры появятся на этапе 3 и здесь сбрасываться не будут."""
        self.training = training
        self._previous: Partition = ()
        self._plan_ids: dict[Group, int] = {}

    def act(self, observation: CoalitionObservation) -> CoalitionDecision:
        """Предложить планы на текущий шаг."""
        candidates = generate_candidates(observation=observation, previous=self._previous, settings=self.settings)
        partition = select_partition(
            candidates=candidates,
            previous=self._previous,
            score=lambda option: self._score(option, observation),
            settings=self.settings,
        )
        self._previous = partition
        if not partition:
            return CoalitionDecision()
        prices = self._planned_prices(partition, observation)
        plans = tuple(self._plan(group, prices, observation) for group in partition)
        return self._validated(plans, observation)

    def _planned_prices(self, partition: Partition, observation: CoalitionObservation) -> dict[int, float]:
        """Вернуть цены всех активных продавцов: участники по плану, остальные по Нэшу."""
        costs = observation.costs
        nash = self.oracle.nash(costs)
        target = self.oracle.partition_equilibrium(costs=costs, partition=partition).prices
        welfare = social_welfare(total_quantity=nash.total_quantity, market=self.market)
        return blended_prices(
            nash=nash.prices,
            target=target,
            members=set().union(*partition),
            mean_cap=mean_price_cap(target_welfare=welfare, gap=self.settings.welfare_gap, market=self.market),
            market=self.market,
        )

    def _score(self, partition: Partition, observation: CoalitionObservation) -> float:
        """Оценить разбиение по приросту над Нэшем при ценах, которые оно реально предложит.

        Именно при них, а не при равновесных: потолок цены уравнивает разбиения, и большая
        группа перестаёт выигрывать только за счёт цены картеля.
        """
        prices = self._planned_prices(partition, observation)
        return self.oracle.gain(costs=observation.costs, prices=prices, members=set().union(*partition))

    def _plan(self, group: Group, prices: dict[int, float], observation: CoalitionObservation) -> CoalitionPlan:
        members = sorted(group)
        return CoalitionPlan(
            plan_id=self._plan_id(group),
            members=tuple(members),
            recommended_prices={member: prices[member] for member in members},
            payout_weights=self._weights(members, prices, observation),
        )

    def _plan_id(self, group: Group) -> int:
        # Один состав сохраняет id весь эпизод: так историю соблюдения можно связать с группой.
        return self._plan_ids.setdefault(group, len(self._plan_ids))

    def _weights(
        self, members: list[int], prices: dict[int, float], observation: CoalitionObservation
    ) -> dict[int, float]:
        scheme = self.settings.payout
        # Этап 2: заменить равное деление для больших групп выборкой перестановок.
        if scheme == 'equal' or len(members) > self.settings.max_exact_players:
            return equal_split(members)
        nash = self.oracle.nash(observation.costs)
        values = level_game(
            players=members,
            prices=prices,
            nash_prices=nash.prices,
            nash_profits=nash.profits,
            costs=observation.costs,
            market=self.market,
        )
        return payout_weights(scheme=scheme, players=members, values=values, edges=observation.edges)

    def _validated(self, plans: tuple[CoalitionPlan, ...], observation: CoalitionObservation) -> CoalitionDecision:
        """Прогнать планы через проверку среды: отклонённый план — потерянный шаг."""
        decision = CoalitionDecision(plans=plans)
        validation = validate_plans(
            decision=decision,
            eligible=eligible_agents(observation),
            edges=set(observation.edges),
            market=self.market,
        )
        # Отклонений быть не должно, если они есть — это ошибка в этом пакете.
        if validation.rejected:
            raise ValueError(f'Policy built invalid plans: {validation.rejected}')
        return decision
