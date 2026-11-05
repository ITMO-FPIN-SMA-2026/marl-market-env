"""Выбор разбиения: какие группы из кандидатов получат планы на этом шаге.

Этап 1 плана. Сначала дешёвый перебор непересекающихся наборов, затем точная оценка
нескольких лучших. Сумма выгод отдельных групп здесь не подходит: планы влияют друг на друга.
"""

from collections.abc import Callable, Sequence

from coalition_policy.settings import PolicySettings
from coalition_policy.value import Group, Partition, canonical


def maximal_packings(candidates: Sequence[Group], *, limit: int) -> list[Partition]:
    """Вернуть до limit непересекающихся наборов, к которым нельзя добавить ни одну группу."""
    # Крупные группы идут первыми: тогда перебор быстрее находит разбиения с большим охватом.
    ordered = sorted(candidates, key=lambda group: (-len(group), sorted(group)))
    found: list[Partition] = []

    def extend(start: int, chosen: list[Group], used: frozenset[int]) -> None:
        if len(found) >= limit:
            return
        free = [index for index in range(start, len(ordered)) if not ordered[index] & used]
        if not free:
            # Пропущенная раньше группа могла ещё поместиться: такой набор не максимальный.
            if chosen and not any(not group & used for group in ordered):
                found.append(canonical(chosen))
            return
        for index in free:
            extend(index + 1, [*chosen, ordered[index]], used | ordered[index])

    extend(0, [], frozenset())
    return list(dict.fromkeys(found))


def coverage_rank(partition: Partition) -> tuple[int, int]:
    """Дешёвая оценка разбиения: охват агентов, затем размер групп.

    Безбилетник зарабатывает больше участника, поэтому важен охват (план, раздел 3.2),
    а ценность растёт быстрее размера, поэтому при равном охвате лучше крупные группы.
    """
    return sum(len(group) for group in partition), sum(len(group) ** 2 for group in partition)


def select_partition(
    *,
    candidates: Sequence[Group],
    previous: Partition,
    score: Callable[[Partition], float],
    settings: PolicySettings,
) -> Partition:
    """Вернуть лучшее разбиение; прошлое остаётся, если новое выгоднее меньше порога."""
    packings = maximal_packings(candidates, limit=settings.max_packings)
    if not packings:
        return ()
    shortlist = sorted(packings, key=coverage_rank, reverse=True)[:settings.max_evaluations]
    best = max(shortlist, key=score)
    current = canonical(previous)
    # Прошлое разбиение годится, только если все его группы всё ещё среди кандидатов.
    if current and set(current) <= set(candidates) and current != best:
        kept = score(current)
        if score(best) <= kept + settings.hysteresis * abs(kept):
            return current
    return best
