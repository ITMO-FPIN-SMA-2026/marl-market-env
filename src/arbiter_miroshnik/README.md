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

Из корня репозитория после checkout этой ветки:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[learning]'
python -m social_market_env.run src/arbiter_miroshnik/arbiter_miroshnik.yaml
```

В Windows активируйте окружение командой `.venv\Scripts\Activate.ps1`.
Зависимость `learning` устанавливает PyTorch >= 2.6. Начальный checkpoint
[`checkpoints/initial_model_0.pt`](checkpoints/initial_model_0.pt) включён в Git
и Python-пакет, поэтому для этого запуска архивы и прежние `runs/` не нужны.
Его происхождение и SHA-256 описаны в [checkpoints/README.md](checkpoints/README.md).

```yaml
arbiter:
  min_fine: 1.0
  max_fine: 100.0
  policy:
    impl: "arbiter_miroshnik:PPOGRUArbiter"
    kwargs:
      checkpoint_path: "src/arbiter_miroshnik/checkpoints/initial_model_0.pt"
      device: cpu
      evaluation: greedy
```

Для CUDA и AMD ROCm используется `device: cuda`. Если GPU явно указан, но недоступен, запуск
завершается ошибкой. Настройки модели сохраняются внутри checkpoint; их список
и ограничения — в [`settings.py`](settings.py).

Готовый конфиг: [`arbiter_miroshnik.yaml`](arbiter_miroshnik.yaml). Это короткая
проверка подключения на четырёх `BestResponseAgent` без коалиций, а не исследование
на командных продавцах.

## Командные эксперименты

Подготовка, обучение и оценивание на закреплённых версиях четырёх настоящих
командных агентов описаны в [`research/README.md`](research/README.md).
Команды для AMD ROCm — в [`docs/amd.md`](docs/amd.md).

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
| [`settings.py`](settings.py) | Параметры PPO и ограничений действий |
| [`checkpoint.py`](checkpoint.py) | Совместимость metadata рынка с checkpoint схемы 4 |
| [`checkpoints/`](checkpoints/README.md) | Начальный checkpoint для запуска из чистого checkout |
| [`research/`](research/README.md) | Обучение, закрепление командных исходников и состояния продавцов |
| [`docs/`](docs/README.md) | Архитектура, результаты и AMD |
| [`research/tests/test_arbiter_miroshnik.py`](research/tests/test_arbiter_miroshnik.py) | Подключение, обучение и возобновление |
