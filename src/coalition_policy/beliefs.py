"""Убеждения политики: чему она верит об агентах, связях и арбитре.

Этап 2 плана (раздел 5.4). Память живёт один эпизод и сбрасывается в reset.
"""

import math
from collections.abc import Iterable

import networkx as nx

from coalition_policy.rewards import StepOutcome
from coalition_policy.settings import PolicySettings
from social_market_env.config import NetworkConfig
from social_market_env.contracts import CoalitionObservation, Pair

FINE_RATE_DECAY: float = 0.8


class TrustTracker:
    """Доверие к агентам: бета-распределение с затуханием и временные исключения."""

    def __init__(self, *, settings: PolicySettings) -> None:
        self.settings = settings
        self._good: dict[int, float] = {}
        self._bad: dict[int, float] = {}
        self._banned_until: dict[int, int] = {}

    def trust(self, agent: int) -> float:
        """Вернуть оценку вероятности, что агент соблюдёт план."""
        good = self.settings.trust_prior_good + self._good.get(agent, 0.0)
        bad = self.settings.trust_prior_bad + self._bad.get(agent, 0.0)
        return good / (good + bad)

    def banned(self, step: int) -> set[int]:
        """Вернуть агентов, которым на этом шаге нельзя предлагать план."""
        return {agent for agent, until in self._banned_until.items() if step < until}

    def update(self, outcome: StepOutcome) -> None:
        """Учесть, как участники обошлись с планами."""
        decay = self.settings.trust_decay
        for plan in outcome.plans:
            for agent in plan.members:
                self._good[agent] = decay * self._good.get(agent, 0.0)
                self._bad[agent] = decay * self._bad.get(agent, 0.0)
            for agent in plan.compliers:
                self._good[agent] += 1.0
            for agent in plan.undercutters:
                self._bad[agent] += 1.0
                # Прощающий триггер: разовое нарушение не исключает, только устойчиво низкое доверие.
                # Исключение за первое нарушение снижало охват до 50 % и само вредило группе.
                if self.trust(agent) < self.settings.ban_trust:
                    # Срок считаем от следующего шага: на нём политика уже знает о нарушении.
                    self._banned_until[agent] = outcome.step + 2 + self.settings.ban_steps
            # Неучастие не бьёт по котлу напрямую, поэтому оно весит вдвое меньше демпинга.
            for agent in plan.absent:
                self._bad[agent] += 0.5


class FineLedger:
    """Частота штрафов и долг компенсации оштрафованным участникам."""

    def __init__(self, *, settings: PolicySettings) -> None:
        self.settings = settings
        self._rate: dict[int, float] = {}
        self._owed: dict[int, float] = {}

    def expected(self, agents: Iterable[int]) -> float:
        """Вернуть ожидаемый штраф группы за шаг: член E[F_C] из ТЗ."""
        return sum(self._rate.get(agent, 0.0) for agent in agents)

    def update(self, outcome: StepOutcome) -> None:
        """Учесть штрафы прошлого шага: обновить частоту и добавить долг перед соблюдавшими."""
        for agent in set(self._rate) | set(outcome.fines):
            fine = outcome.fines.get(agent, 0.0)
            self._rate[agent] = FINE_RATE_DECAY * self._rate.get(agent, 0.0) + (1 - FINE_RATE_DECAY) * fine
        compliers = {agent for plan in outcome.plans for agent in plan.compliers}
        for agent, fine in outcome.fines.items():
            if agent in compliers:
                self._owed[agent] = self._owed.get(agent, 0.0) + fine

    def peek_compensation(self, agents: Iterable[int]) -> dict[int, float]:
        """Вернуть надбавки к выплатам на этот шаг, не меняя долг."""
        rate = self.settings.insurance_rate
        return {agent: rate * self._owed[agent] for agent in agents if self._owed.get(agent, 0.0) > 0}

    def settle(self, paid: dict[int, float]) -> None:
        """Списать выплаченные надбавки."""
        for agent, amount in paid.items():
            self._owed[agent] = max(0.0, self._owed.get(agent, 0.0) - amount)


class LinkSurvival:
    """Вероятность, что группа останется связной на следующем шаге.

    Повторяет логит из evolve_edges. Близости пары в наблюдении нет, поэтому она считается
    равной среднему (нулю). Просьбы агентов о связях тоже неизвестны и не учитываются.
    """

    def __init__(self, *, config: NetworkConfig, settings: PolicySettings) -> None:
        self.config = config
        self.settings = settings

    def keep_probability(self, *, pair: Pair, observation: CoalitionObservation) -> float:
        """Вернуть вероятность сохранить существующую связь."""
        config = self.config
        first, second = pair
        profits = observation.last_profits
        utility = math.tanh((profits.get(first, 0.0) + profits.get(second, 0.0)) / (2.0 * config.utility_scale))
        logit = (
            config.a_keep
            + config.w_history * observation.edges.get(pair, 0.0)
            + config.w_utility * utility
            - config.w_link_cost * observation.link_costs.get(pair, 0.0)
        )
        return 1.0 / (1.0 + math.exp(-logit))

    def group_survival(self, *, group: Iterable[int], observation: CoalitionObservation) -> float:
        """Вернуть вероятность сохранить связность: произведение по самому надёжному остовному дереву.

        Это нижняя оценка: лишние рёбра только повышают надёжность группы.
        """
        if self.settings.network_mode == 'static':
            return 1.0
        members = set(group)
        graph = nx.Graph()
        graph.add_nodes_from(sorted(members))
        for pair in observation.edges:
            if set(pair) <= members:
                graph.add_edge(*pair, weight=self.keep_probability(pair=pair, observation=observation))
        if not nx.is_connected(graph):
            return 0.0
        tree = nx.maximum_spanning_tree(graph)
        return math.prod(weight for _, _, weight in tree.edges(data='weight'))


class Beliefs:
    """Все убеждения одного эпизода."""

    def __init__(self, *, settings: PolicySettings, network: NetworkConfig) -> None:
        self.settings = settings
        self.links = LinkSurvival(config=network, settings=settings)
        self.reset()

    def reset(self) -> None:
        """Забыть эпизод."""
        self.trust = TrustTracker(settings=self.settings)
        self.fines = FineLedger(settings=self.settings)

    def learn(self, outcome: StepOutcome) -> None:
        """Учесть исход шага. При memory=False доверие не меняется: это контрольный режим."""
        if self.settings.memory:
            self.trust.update(outcome)
        self.fines.update(outcome)

    def blocked(self, step: int) -> set[int]:
        """Вернуть агентов, которых нельзя звать в план."""
        return self.trust.banned(step) if self.settings.memory else set()

    def expected_compliance(self, group: Iterable[int]) -> float:
        """Вернуть ожидаемую долю соблюдающих в группе: множитель rho(C) из раздела 4.3."""
        members = list(group)
        return sum(self.trust.trust(agent) for agent in members) / len(members)
