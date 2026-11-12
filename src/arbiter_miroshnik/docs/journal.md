# Журнал организации пакета

## 2026-11-12

- Пакет оформлен по структуре `coalitions_badretdinov` из `main`: публичный
  фасад, модули политики, README, именной YAML, `research/` и `docs/`.
- Автор и каноническое имя: Miroshnik, `arbiter_miroshnik`.
- Содержимое `model.py`, `features.py`, `settings.py`, `contracts.py` и
  `reference.py` перенесено без изменения байтов. Исходный алгоритм перенесён
  из `transferred_ppo_gru/policy.py` в `arbiter_miroshnik/ppo.py`.
- Фасад из прежнего `api.py` получил имя `PPOGRUArbiter` и находится в
  `policy.py`. Launcher, работа с источниками, калибровка, конверсия и состояния
  продавцов собраны в `research/`.
- Runtime, инструменты, тесты и документы собраны в `arbiter_miroshnik`.
  Прежние отдельные пакеты и корневой launcher исключены. Запуск:
  `python -m arbiter_miroshnik`; компонент: `arbiter_miroshnik:PPOGRUArbiter`.
  Численный формат checkpoint схемы 4 сохранён.
- Обученные сети и исходные эксперименты сохранены. Другие контроллеры в пакет
  не входят; данные об их результатах не используются для оценки PPO + GRU.

## 2026-11-13

- Локальный `main` возвращён к дереву до слияния отдельным revert-коммитом.
- Последние изменения собраны в локальной ветке `arbiter-miroshnik-ppo-gru`.
  Удалённый репозиторий на этом этапе не изменяется.
