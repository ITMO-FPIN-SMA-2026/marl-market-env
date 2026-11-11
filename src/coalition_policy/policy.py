"""AdaptiveCoalitionPolicy: фасад контракта CoalitionPolicyAPI.

Конвейер шага (план, раздел 4.1):

    1. Убеждения: доверие, связи, штрафы      — beliefs.py, rewards.py (этап 2)
    2. Кандидаты                              — candidates.py
    3. Стратегический рычаг                   — learner.py (этап 3)
    4. Выбор разбиения по оценке V(C)         — structure.py, _group_value
    5. Цены: лестница и соблазн               — pricing.py (этап 2)
    6. Доли котла                             — payouts.py
    7. Исход шага и обновление убеждений      — observe_transition
"""

from typing import Any

import numpy as np

from coalition_policy.beliefs import Beliefs
from coalition_policy.candidates import eligible_agents, generate_candidates
from coalition_policy.learner import ARMS, Block, LinearThompson
from coalition_policy.payouts import core_excess, level_game, payout_weights, restrict_to_graph
from coalition_policy.pricing import (
    PriceLadder,
    cap_share,
    future_weight,
    incentive_holds,
    largest_incentive_share,
    mean_price_cap,
    path_prices,
)
from coalition_policy.rewards import StepOutcome, restore_outcome
from coalition_policy.settings import PRICE_LEVELS, PolicySettings
from coalition_policy.structure import select_partition
from coalition_policy.value import Group, Partition, ValueOracle
from social_market_env.config import CoalitionsConfig, MarketConfig, NetworkConfig
from social_market_env.contracts import CoalitionDecision, CoalitionObservation, CoalitionPlan, CoalitionPolicyAPI
from social_market_env.core.coalitions import validate_plans
from social_market_env.core.scoring import social_welfare
from social_market_env.gametheory.graphs import connected_components
from social_market_env.gametheory.payouts import equal_split


class AdaptiveCoalitionPolicy(CoalitionPolicyAPI):
    """Предлагает связные коалиции с удерживаемой ценой и долями по вкладу."""

    def __init__(
        self,
        *,
        market: dict[str, Any] | None = None,
        network: dict[str, Any] | None = None,
        compliance_tolerance: float = CoalitionsConfig.compliance_tolerance,
        checkpoint_path: str | None = None,
        **settings: Any,
    ) -> None:
        # Секцию market среда подставит сама, остальные ключи приходят из kwargs в YAML.
        self.market = MarketConfig(**(market or {}))
        self.settings = PolicySettings(**settings)
        self.compliance_tolerance = compliance_tolerance
        # Оракул живёт дольше эпизода: себестоимости внутри seed те же, и кэш остаётся полезным.
        self.oracle = ValueOracle(market=self.market)
        self.beliefs = Beliefs(settings=self.settings, network=NetworkConfig(**(network or {})))
        # Бандит живёт дольше эпизода: это единственная часть политики, которая учится между эпизодами.
        self.learner = LinearThompson(exploration=self.settings.exploration)
        if checkpoint_path is not None:
            self.learner.load(checkpoint_path)
        self.reset(seed=0, training=False)

    def reset(self, *, seed: int, training: bool) -> None:
        """Забыть память эпизода; обученный бандит при этом сохраняется."""
        self.training = training
        self.learner.reseed(seed)
        self._block: Block | None = None
        self._arm_cap = 1.0
        self._pressure = 0.0
        self._compliance = 1.0
        self.beliefs.reset()
        self.ladder = PriceLadder(settings=self.settings)
        self._previous: Partition = ()
        self._plan_ids: dict[Group, int] = {}
        self._unions: dict[Group, tuple[Group, ...]] = {}
        self._fresh_left = 0
        self.last_outcome: StepOutcome | None = None
        self.last_report: dict[str, float] = {}
        """Показатели последнего шага для экспериментов; на решения не влияют."""

    def act(self, observation: CoalitionObservation) -> CoalitionDecision:
        """Предложить планы на текущий шаг."""
        self._fresh_left = self.settings.max_fresh_equilibria
        self._pull_lever(observation)
        blocked = self.beliefs.blocked(observation.step)
        candidates = generate_candidates(
            observation=observation, previous=self._previous, settings=self.settings, blocked=blocked
        )
        partition = select_partition(
            candidates=candidates,
            previous=self._previous,
            score=lambda option: self._score(option, observation),
            settings=self.settings,
        )
        self._remember_unions(partition)
        self._previous = partition
        self.last_report = {'gap': self.ladder.gap, 'ceiling': self.ladder.ceiling, 'blocked': float(len(blocked))}
        if not partition:
            return CoalitionDecision()
        prices = self._final_prices(partition, observation)
        plans = tuple(self._plan(group, prices, observation) for group in partition)
        self._pay_insurance(plans)
        if self.settings.diagnostics:
            self.last_report['core_gap'] = max(self._core_gap(plan, prices, observation) for plan in plans)
        return self._validated(plans, observation)

    def observe_transition(
        self,
        *,
        observation: CoalitionObservation,
        action: CoalitionDecision,
        reward: float,
        next_observation: CoalitionObservation | None,
        done: bool,
    ) -> None:
        """Восстановить исход шага и обновить убеждения.

        Убеждения меняются и при training=False: это память внутри эпизода, а не обучение весов.
        """
        outcome = restore_outcome(
            observation=observation,
            decision=action,
            next_observation=next_observation,
            oracle=self.oracle,
            tolerance=self.compliance_tolerance,
        )
        if outcome is not None:
            self.last_outcome = outcome
            self.beliefs.learn(outcome)
            self._remember_signals(outcome, observation)
            if self.settings.pricing == 'adaptive':
                self.ladder.update(outcome)
        if done:
            self._close_block()

    def _remember_signals(self, outcome: StepOutcome, observation: CoalitionObservation) -> None:
        """Обновить сглаженные сигналы контекста и копилку текущего блока бандита."""
        reacted = float(outcome.arbiter_reacted)
        self._pressure = 0.8 * self._pressure + 0.2 * reacted
        if outcome.compliance is not None:
            self._compliance = 0.8 * self._compliance + 0.2 * outcome.compliance
        if self._block is not None:
            pairs = max(1, len(observation.link_costs))
            agents = max(1, len(observation.active_agents))
            self._block.total += outcome.reward / agents
            self._block.total -= self.settings.link_penalty * len(outcome.raised_pairs) / pairs
            self._block.steps += 1

    def _pull_lever(self, observation: CoalitionObservation) -> None:
        """На границе блока закрыть прошлый выбор и сделать новый."""
        if self.settings.learning != 'bandit':
            return
        if observation.step % self.settings.block_steps == 0:
            self._close_block()
            context = self._context(observation)
            arm = self.learner.choose(context, explore=self.training)
            self._block = Block(arm=arm, context=context)
            self._arm_cap = PRICE_LEVELS[ARMS[arm]]
        # Лестница не должна расти выше варианта: иначе смена варианта дала бы резкий скачок цены.
        self.ladder.gap = min(self.ladder.gap, self._arm_cap)

    def _close_block(self) -> None:
        block, self._block = self._block, None
        if block is not None and block.steps > 0 and self.training:
            self.learner.update(arm=block.arm, context=block.context, reward=block.total / block.steps)

    def _context(self, observation: CoalitionObservation) -> np.ndarray:
        """Собрать признаки ситуации для бандита, все примерно в [0, 1]."""
        active = observation.active_agents
        pairs = max(1, len(active) * (len(active) - 1) // 2)
        parts = connected_components(nodes=active, edges=observation.edges)
        costs = observation.link_costs.values()
        history = observation.edges.values()
        outcome = self.last_outcome
        return np.array([
            1.0,
            len(observation.edges) / pairs,
            max((len(part) for part in parts), default=0) / max(1, len(active)),
            sum(history) / len(observation.edges) if observation.edges else 0.0,
            self._compliance,
            min(1.0, sum(costs) / max(1, len(observation.link_costs)) / 10.0),
            self._pressure,
            max(0.0, outcome.welfare_gap) if outcome is not None else 0.0,
            1.0 - observation.step / self.settings.horizon,
        ])

    def _allowed_gap(self) -> float:
        if self.settings.pricing != 'adaptive':
            return self.settings.welfare_gap
        return min(self.ladder.gap, self._arm_cap)

    def _path(
        self, partition: Partition, observation: CoalitionObservation
    ) -> tuple[dict[int, float], dict[int, float], set[int], float]:
        """Вернуть данные пути цен: Нэш, цель, участников и наибольшую долю под потолком SW."""
        costs = observation.costs
        nash = self.oracle.nash(costs)
        target = self.oracle.partition_equilibrium(costs=costs, partition=partition).prices
        members = set().union(*partition)
        welfare = social_welfare(total_quantity=nash.total_quantity, market=self.market)
        cap = mean_price_cap(target_welfare=welfare, gap=self._allowed_gap(), market=self.market)
        return nash.prices, target, members, cap_share(nash=nash.prices, target=target, members=members, mean_cap=cap)

    def _capped_prices(self, partition: Partition, observation: CoalitionObservation) -> dict[int, float]:
        nash, target, members, share = self._path(partition, observation)
        return path_prices(nash=nash, target=target, members=members, share=share, market=self.market)

    def _final_prices(self, partition: Partition, observation: CoalitionObservation) -> dict[int, float]:
        """Вернуть цены плана: потолок SW, затем ограничение на соблазн предать."""
        nash, target, members, share = self._path(partition, observation)

        def at(value: float) -> dict[int, float]:
            return path_prices(nash=nash, target=target, members=members, share=value, market=self.market)

        if self.settings.pricing == 'fixed':
            return at(share)
        costs = observation.costs
        nash_profits = self.oracle.nash(costs).profits
        # Веса считаем один раз на верхней цене: иначе каждый шаг бисекции стоил бы счёта долей.
        groups = {index: self._weights(group, at(share), observation) for index, group in enumerate(partition)}
        steps_left = self.settings.horizon - observation.step - 1
        future = {
            member: future_weight(
                patience=self.settings.patience * self.beliefs.trust.trust(member), steps_left=steps_left
            )
            for member in members
        }

        def holds(value: float) -> bool:
            return incentive_holds(prices=at(value), groups=groups, future=future, nash_profits=nash_profits,
                                   costs=costs, market=self.market)

        result = largest_incentive_share(upper=share, holds=holds)
        self.last_report['ic_share'] = result / share if share > 0 else 1.0
        return at(result)

    def _score(self, partition: Partition, observation: CoalitionObservation) -> float:
        """Оценить разбиение суммой V(C) групп; непосчитанные сверх бюджета отбрасываются."""
        if not self.oracle.knows(costs=observation.costs, partition=partition):
            if self._fresh_left <= 0:
                return float('-inf')
            self._fresh_left -= 1
        prices = self._capped_prices(partition, observation)
        return sum(self._group_value(group, prices, observation) for group in partition)

    def _group_value(self, group: Group, prices: dict[int, float], observation: CoalitionObservation) -> float:
        """Вернуть V(C) из раздела 4.3 плана: дисконтированный дополнительный доход группы.

        Цены здесь уже учитывают соседние планы, поэтому внешние эффекты разбиения внутри оценки.
        """
        gain = self.oracle.gain(costs=observation.costs, prices=prices, members=group)
        rho = self.beliefs.expected_compliance(group)
        survival = self.beliefs.links.group_survival(group=group, observation=observation)
        fines = self.beliefs.fines.expected(group)
        return (rho * gain - fines) / (1.0 - self.settings.discount * survival)

    def _plan(self, group: Group, prices: dict[int, float], observation: CoalitionObservation) -> CoalitionPlan:
        members = sorted(group)
        return CoalitionPlan(
            plan_id=self._plan_id(group),
            members=tuple(members),
            recommended_prices={member: prices[member] for member in members},
            payout_weights=self._weights(group, prices, observation),
        )

    def _plan_id(self, group: Group) -> int:
        # Один состав сохраняет id весь эпизод: так историю соблюдения можно связать с группой.
        return self._plan_ids.setdefault(group, len(self._plan_ids))

    def _remember_unions(self, partition: Partition) -> None:
        """Запомнить, из каких прошлых групп собрана новая: это априорные союзы Оуэна.

        Союзы закрепляются при первом появлении группы и живут, пока группа существует.
        """
        for group in partition:
            if group in self._unions:
                continue
            parts = [group & old for old in self._previous if len(group & old) >= 2]
            covered = frozenset().union(*parts) if parts else frozenset()
            self._unions[group] = (*parts, *(frozenset({member}) for member in sorted(group - covered)))

    def _blocks(self, group: Group, members: list[int]) -> list[int]:
        index = {member: position for position, member in enumerate(members)}
        return [sum(1 << index[member] for member in union) for union in self._unions.get(group, ())]

    def _weights(self, group: Group, prices: dict[int, float], observation: CoalitionObservation) -> dict[int, float]:
        members = sorted(group)
        scheme = self.settings.payout
        # N в ЛР равно 12. Для больших групп таблица 2^n не влезет, тогда делим поровну.
        if scheme == 'equal' or len(members) > self.settings.max_exact_players:
            return equal_split(members)
        values = self._level_table(members, prices, observation)
        bonus = self.beliefs.fines.peek_compensation(members) if scheme == 'hybrid' else None
        return payout_weights(scheme=scheme, players=members, values=values, edges=observation.edges,
                              blocks=self._blocks(group, members), bonus=bonus)

    def _level_table(self, members: list[int], prices: dict[int, float], observation: CoalitionObservation):
        nash = self.oracle.nash(observation.costs)
        return level_game(players=members, prices=prices, nash_prices=nash.prices, nash_profits=nash.profits,
                          costs=observation.costs, market=self.market)

    def _pay_insurance(self, plans: tuple[CoalitionPlan, ...]) -> None:
        """Списать долг страховки ровно один раз за шаг: веса могут считаться несколько раз."""
        if self.settings.payout == 'hybrid':
            for plan in plans:
                self.beliefs.fines.settle(self.beliefs.fines.peek_compensation(plan.members))

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

    def _core_gap(self, plan: CoalitionPlan, prices: dict[int, float], observation: CoalitionObservation) -> float:
        """Вернуть долю котла, которую недополучает самая недовольная подгруппа плана."""
        members = sorted(plan.members)
        table = self._level_table(members, prices, observation)
        game = restrict_to_graph(table, players=members, edges=observation.edges)
        payouts = game[-1] * np.array([plan.payout_weights[member] for member in members])
        return max(0.0, core_excess(payouts, game)) / max(float(game[-1]), 1e-9)
