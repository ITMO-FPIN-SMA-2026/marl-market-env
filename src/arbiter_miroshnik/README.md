# arbiter_miroshnik — социальный арбитр PPO + GRU

**Автор:** Miroshnik

Арбитр для `social_market_env` (контракт `2.0.0`), класс
`arbiter_miroshnik:PPOGRUArbiter`. Здесь находится исходный PPO с отдельными
actor и critic и рекуррентной памятью GRU. Архитектура и численные веса при
переименовании пакета не изменялись.

## Идея

Арбитр наблюдает историю публичных цен, продаж и потребительского излишка.
Actor решает, наблюдать дальше или оштрафовать одного продавца. Critic оценивает
ожидаемую награду; завершённые эпизоды используются для обновлений PPO.

| Механизм | Что делает |
| --- | --- |
| Публичная память | Цены и продажи продавцов, изменения рынка, сходство ценовых траекторий |
| Actor и critic | Отдельные сети, попарные признаки, GRU, объединение информации продавцов |
| Действие | Наблюдение либо штраф одному продавцу; сумма штрафа задаётся настройками |
| Ограничения | Минимальная история, cooldown, бюджет продавца и общий бюджет эпизода |
| Обучение | PPO, GAE, Adam, ограничение градиентов и остановка по KL |
| Сохранение | После каждой игры: сети, оптимизаторы, траектории, RNG и SHA-256 состояния |

Входами сетей не служат граф, истинные коалиции, себестоимости, метки вреда или
конкурентный `SW*`. Штатная награда среды используется только как цель обучения.

## Подключение

```yaml
arbiter:
  min_fine: 1.0
  max_fine: 100.0
  policy:
    impl: "arbiter_miroshnik:PPOGRUArbiter"
    kwargs:
      checkpoint_path: "runs/transferred_ppo/initial_model_0.pt"
      device: cpu
      evaluation: greedy
```

Восстановление состояний описано в [инструкции переноса](docs/next_computer.md). Для CUDA и AMD
ROCm используется `device: cuda`. Если GPU явно указан, но недоступен, запуск
завершается ошибкой. Настройки модели сохраняются внутри checkpoint; их список
и ограничения — в [`settings.py`](settings.py).

Готовый конфиг: [`arbiter_miroshnik.yaml`](arbiter_miroshnik.yaml). Это короткая
проверка подключения на четырёх `BestResponseAgent` без коалиций, а не исследование
на командных продавцах. После восстановления архивов:

```bash
python -m social_market_env.run src/arbiter_miroshnik/arbiter_miroshnik.yaml
```

## Командные эксперименты

Подготовка, обучение и оценивание на закреплённых версиях четырёх настоящих
командных агентов находятся в [`research/`](research). Запуск из корня репозитория:

```bash
.deps/rocm-venv/bin/python -m arbiter_miroshnik evaluate \
  --out runs/transferred_ppo_amd --episodes 20 --device cuda
.deps/rocm-venv/bin/python -m arbiter_miroshnik train \
  --out runs/transferred_ppo_amd --games 128 --device cuda
```

Для CPU:

```bash
python -m arbiter_miroshnik evaluate \
  --out runs/transferred_ppo_amd --episodes 20 --device cpu
```

`--games` — целевое общее число игр каждой модели. В одном каталоге одновременно
может работать только один процесс обучения. В конфигурациях используется
`arbiter_miroshnik:PPOGRUArbiter`; checkpoint схемы 4 сохраняют совместимость.

## Результаты

На AMD RX 6600 XT проверены настоящие обновления PPO: три модели в GPU-серии
продолжены с 64 до 72 игр, с восьми до девяти PPO-update. Actor, critic,
оптимизаторы и обе популяции продавцов сохранены.

На 60 новых проверочных играх этой серии разница SW между адаптированными
популяциями составила **+68.343**, 95% интервал **[-84.269; 223.544]**.
Конечная greedy-политика не штрафовала; точность санкций не определена.
Положительный эффект и полная квалификация **не подтверждены**. Результаты
других контроллеров к PPO + GRU не относятся.

Полные ограничения вывода и таблицы прежних серий —
[`docs/experiments.md`](docs/experiments.md), машинная сводка —
[`research/results.json`](research/results.json).

## Состав

| Путь | Что это |
| --- | --- |
| [`policy.py`](policy.py) | Фасад штатного `ArbiterAPI`, применение действий и сохранение этапов |
| [`ppo.py`](ppo.py), [`model.py`](model.py) | Исходные PPO/GAE и actor/critic с GRU |
| [`features.py`](features.py), [`reference.py`](reference.py) | Публичные признаки, память и калибровка |
| [`settings.py`](settings.py), [`contracts.py`](contracts.py) | Параметры и внутренний контракт |
| [`research/`](research) | Launcher, закрепление командных исходников, перенос весов и продавцов |
| [`docs/`](docs/README.md) | Архитектура, результаты, журнал переноса и AMD |
| [`research/tests/test_arbiter_miroshnik.py`](research/tests/test_arbiter_miroshnik.py) | Подключение, обучение и возобновление |
