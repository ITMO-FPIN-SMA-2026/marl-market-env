"""Оракул ценности: равновесия рынка с кэшем.

Этап 1 плана. Равновесие стоит 10-80 мс, а `act` должен укладываться примерно в 0.2 с.
Себестоимости внутри seed не меняются, поэтому кэш живёт между шагами и эпизодами
и сбрасывается сам: себестоимости входят в ключ, а выбывание меняет их набор.

Этап 2 добавит сюда внешнюю опцию безбилетника o_i и соблазн предать T_i.
"""

from collections.abc import Iterable

from social_market_env.config import MarketConfig
from social_market_env.core.market import clear_market
from social_market_env.gametheory.equilibrium import Equilibrium, group_equilibrium, nash_equilibrium

CostKey = frozenset[tuple[int, float]]
Group = frozenset[int]
Partition = tuple[Group, ...]


def canonical(partition: Iterable[Group]) -> Partition:
    """Вернуть разбиение в едином порядке, чтобы равные разбиения давали один ключ."""
    return tuple(sorted(partition, key=sorted))


class ValueOracle:
    """Считает и запоминает равновесия для набора себестоимостей."""

    def __init__(self, *, market: MarketConfig) -> None:
        self.market = market
        self._nash: dict[CostKey, Equilibrium] = {}
        self._partitions: dict[tuple[CostKey, Partition], Equilibrium] = {}

    def nash(self, costs: dict[int, float]) -> Equilibrium:
        """Вернуть конкурентное равновесие: опорную точку всех сравнений."""
        key = frozenset(costs.items())
        if key not in self._nash:
            self._nash[key] = nash_equilibrium(costs=costs, market=self.market)
        return self._nash[key]

    def partition_equilibrium(self, *, costs: dict[int, float], partition: Partition) -> Equilibrium:
        """Вернуть равновесие, где каждая группа разбиения действует как один картель.

        Считаем всё разбиение сразу: планы влияют друг на друга. Сумма выгод групп,
        посчитанных поодиночке, сильно занижает разбиения вроде [6, 6] (план, раздел 3.2).
        """
        key = (frozenset(costs.items()), canonical(partition))
        if key not in self._partitions:
            grouped = set().union(*partition) if partition else set()
            singles = [(agent_id,) for agent_id in sorted(costs) if agent_id not in grouped]
            groups = [tuple(sorted(group)) for group in key[1]] + singles
            self._partitions[key] = group_equilibrium(groups=groups, costs=costs, market=self.market)
        return self._partitions[key]

    def gain(self, *, costs: dict[int, float], prices: dict[int, float], members: Iterable[int]) -> float:
        """Вернуть прирост прибыли участников над Нэшем при заданных ценах всех продавцов."""
        profits = clear_market(prices=prices, costs=costs, market=self.market).gross_profits
        nash = self.nash(costs).profits
        return sum(profits[member] - nash[member] for member in members)
