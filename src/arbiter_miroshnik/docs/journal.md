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
- `transferred_ppo_gru` оставлен как набор совместимых импортов, указывающих
  на те же объекты модулей. Старые YAML, launcher и checkpoint схемы 4 работают.
- Обученные сети и исходные эксперименты сохранены. Другие контроллеры в пакет
  не входят; данные об их результатах не используются для оценки PPO + GRU.
