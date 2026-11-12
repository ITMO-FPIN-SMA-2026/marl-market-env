# SMA environment

Среда для ЛР «MARL в динамической сетевой рыночной игре с эндогенными коалициями и
социально-ориентированным арбитражем».

## Установка

Windows PowerShell (Python 3.11 или новее):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m social_market_env.run experiments/demo.yaml
```

Linux/macOS:

```bash
$ python3 -m venv .venv 
$ .venv/bin/python -m pip install -e '.[dev]'
```

## Запуск эксперимента

```bash
.venv/bin/python -m social_market_env.run experiments/demo.yaml
```

Результат — папка `runs/<дата>_<имя>/`: `meta.json` (конфиг и версии) и
`seed_<s>/episode_<e>/`: `episode.json` (себестоимости, сид эпизода, реализации компонентов),
`steps.jsonl` (одна строка на шаг), `summary.json`, `timings.json`.

## Как подключить свою модель

1. Сделайте Python-пакет в своём репозитории и реализуйте один из контрактов из
   `social_market_env.contracts`: `MarketAgentAPI`, `CoalitionPolicyAPI` или `ArbiterAPI`.
2. Установите пакет в окружение среды: `uv pip install --python .venv/bin/python git+https://github.com/ITMO-FPIN-SMA-2026/<repo>`.
3. Укажите класс в конфиге: `impl: "your_package.module:YourClass"`, аргументы конструктора — в `kwargs`.

Пример агента:

```python
from social_market_env.contracts import AgentAction, AgentObservation, MarketAgentAPI


class MyAgent(MarketAgentAPI):
    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        self.training = training

    def act(self, observation: AgentObservation) -> AgentAction:
        if observation.offer is not None:
            return AgentAction(price=observation.offer.recommended_price, coalition_choice=observation.offer.plan_id)
        return AgentAction(price=observation.cost + 5.0)

    def observe_transition(self, *, observation, action, reward, next_observation, done) -> None:
        ...  # обучение здесь, если training=True
```

Готовые простые реализации для соперников и тестов — в `social_market_env.baselines`.

Индивидуальный модуль рыночного агента Аскарова Тимура Руслановича:
[`src/agent_askarov/README.md`](src/agent_askarov/README.md) — реализация,
запуск, обучение и результаты сравнения.

Пример политики коалиций — предлагает картель первой паре связанных агентов:

```python
from social_market_env.contracts import CoalitionDecision, CoalitionObservation, CoalitionPlan, CoalitionPolicyAPI


class FirstPairCartel(CoalitionPolicyAPI):
    def reset(self, *, seed: int, training: bool) -> None:
        self.training = training

    def act(self, observation: CoalitionObservation) -> CoalitionDecision:
        eligible = set(observation.active_agents) - set(observation.restricted)
        pairs = [pair for pair in observation.edges if set(pair) <= eligible]
        if not pairs:
            return CoalitionDecision()
        first, second = pairs[0]
        plan = CoalitionPlan(
            plan_id=0,
            members=(first, second),
            recommended_prices={first: 60.0, second: 60.0},
            payout_weights={first: 0.5, second: 0.5},
        )
        return CoalitionDecision(plans=(plan,))

    def observe_transition(self, *, observation, action, reward, next_observation, done) -> None:
        ...  # reward — доход котла соблюдавших участников минус их штрафы
```

Пример арбитра — штрафует самого дорогого продавца после скачка средней цены:

```python
from social_market_env.contracts import ArbiterAction, ArbiterActionKind, ArbiterAPI, ArbiterObservation


class JumpArbiter(ArbiterAPI):
    def reset(self, *, seed: int, training: bool) -> None:
        self.training = training

    def act(self, observation: ArbiterObservation) -> ArbiterAction:
        if observation.delta_average_price > 5.0 and observation.prices:
            target = max(observation.prices, key=observation.prices.get)
            return ArbiterAction(kind=ArbiterActionKind.sanction, target=target, fine=50.0)
        return ArbiterAction()  # observe; поднять стоимость связи:
        # ArbiterAction(kind=ArbiterActionKind.change_link_cost, link_cost_changes={(0, 3): 2.0})
```

Скаляры numpy (`np.int64`, `np.float32`, `np.bool_`, например результат `np.argmax`) в действиях
принимаются и приводятся к обычным `int`/`float`/`bool`.

**Как оценивается арбитр.** Награда `−α·|SW − SW*|/SW* − β·I − λ_FP·FP − λ_FN·FN`. FP — штраф агенту вне
вредной коалиции или повышение стоимости связи хотя бы для одной пары вне вредной коалиции. FN — вредная
коалиция есть, а арбитр не оштрафовал её участника и не поднял стоимость связи внутри неё; повышение
стоимости засчитывается только при `network.mode: evolving`. Штраф меньше `arbiter.min_fine` считается
наблюдением, изменения стоимости связи обрезаются до `±arbiter.max_link_cost_delta`.

## Арбитр Мирошник Виктория Юрьевна — PPO + GRU

Пакет [`src/arbiter_miroshnik`](src/arbiter_miroshnik/README.md) содержит исходный
PPO с отдельными actor и critic и памятью GRU. Внутри — именной YAML,
`research/` с launcher командных экспериментов и `docs/` с архитектурой,
результатами и инструкциями AMD ROCm. Начальный checkpoint включён в пакет;
для запуска примера установите `pip install -e '.[learning]'`.
Результаты и ограничения описаны в
[документации экспериментов](src/arbiter_miroshnik/docs/experiments.md).
Полная квалификация арбитра пока не подтверждена.

## Конфиг

Все ключи и значения по умолчанию описаны в `src/social_market_env/config.py` (докстринги у каждого поля);
неизвестные ключи считаются ошибкой. `mode: eval` — компоненты получают `training=False` и должны работать
замороженными, `mode: train` — `training=True`. `strict: true` (или `--strict` в CLI) останавливает прогон на
первой ошибке в командном коде; по умолчанию ошибка пишется в лог, а вместо действия подставляется безопасное.
Baselines с параметром `market` в конструкторе получают секцию `market` из конфига эксперимента автоматически.

## Режим библиотеки

Для своего цикла обучения среду можно создать напрямую — все аргументы конструктора только именованные:

```python
from social_market_env.environment import SocialMarketEnvironment

env = SocialMarketEnvironment(agents=agents, costs=costs, coalition_policy=policy, arbiter=arbiter, config=config)
```

или собрать всё из конфига так же, как это делает раннер:

```python
from social_market_env.config import config_from_dict
from social_market_env.runner.experiment import build_environment

env = build_environment(config=config_from_dict({'mode': 'train', 'horizon': 200}), seed=0)
for episode in range(100):
    env.reset(seed=episode, training=True)
    while not env.done:
        record = env.step().record  # тот же словарь, что строка в steps.jsonl
```

- Метода `step(actions)` нет: среда сама вызывает `act` у всех ролей. Обучение происходит внутри
  `observe_transition`; он вызывается и при `training=False` (там модель не должна обновляться) и получает
  **санированное** действие — после обрезки цены, ограничений и приведения типов.
- `env.done` — достигнут горизонт или выбыли все агенты; отдельного флага truncation нет.
- Без явного `guard=CallGuard(strict=False, ...)` среда, созданная конструктором, падает на первой ошибке
  в командном коде; `build_environment` берёт `strict` из конфига.
- Для централизованного обучения (MAPPO, QMIX) есть `env.global_state()`.
- `env.agent_state[agent_id]` хранит полную историю `observations`, `expectation_history`,
  `evaluations`, а также текущий прогноз `expectations`. Прогноз по умолчанию повторяет
  результат предыдущего шага; ошибки прогноза записываются в `AgentEvaluation`.
- В `AgentObservation` доступны собственные `expectations`, `last_evaluation` и
  `last_coalition_join_vector`. Последний вектор индексируется по
  `last_coalition_plan_ids`; один элемент равен 1, если агент присоединился к плану.
  Запись шага содержит `coalition_plan_ids` и векторы всех агентов.
- В записи шага `graph_nodes` задаёт порядок строк и столбцов трёх матриц
  `graph_matrices`: `adjacency` (прямые связи), `reachability` (связность по путям)
  и `coalition_membership` (фактическое присоединение к одному плану). Они относятся
  к графу в начале шага `G_t`. Эволюция `G_t` через NetworkX и вероятности из
  `NetworkConfig` формирует граф следующего шага.

## Тесты

```bash
.venv/bin/pytest -q
```
