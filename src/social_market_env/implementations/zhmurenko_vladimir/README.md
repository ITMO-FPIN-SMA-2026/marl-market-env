# Индивидуальный рыночный агент Жмуренко Владимира

Публичный класс: `social_market_env.implementations.zhmurenko_vladimir.agent:ZhmurenkoVladimirAgent`.
Рабочая версия архитектуры — **3**; формат checkpoint — **2**; схема наблюдения — **1 (85 признаков)**.
Все исходники, конфигурации, тесты и инструменты находятся в этом пакете. Родительский
`implementations/` используется как namespace-пакет, без создания `__init__.py` вне нашей области.

## Изменения после диагностики

Исходный checkpoint V1 воспроизвёл нулевую прибыль на 750/750 контрольных шагов:
цена совпадала с собственной себестоимостью при положительных продажах. Внутри старой
ценовой головы logits убывали выше себестоимости; бинарный признак `at_or_above_cost`
создавал скачок, закреплявший моду на границе нулевой маржи. Его удаление из входа старой
головы смещало максимум к минимальной цене. Цены выше себестоимости оставались вероятны,
поэтому стохастическое исполнение приносило прибыль. Это подтверждено реальными
наблюдениями и вмешательством только в собственный признак модели.

Исправление не меняет экономику или набор цен. Генератор кандидатов в `actions.py`
сохранён. Новая нейросетевая голова выводит обучаемые центр и ширину:

```text
center = sigmoid(price_head(actor_features)[0])
scale = 0.02 + (0.30 - 0.02) * sigmoid(price_head(actor_features)[1])
logit(candidate) = -0.5 * ((candidate_price / p_max - center) / scale)^2
```

Softmax по этим logits остаётся категориальной политикой на существующих кандидатах.
Здесь нет фиксированной наценки, формулы спроса, расчёта прибыли или обращения к чужим
данным. Центр и ширина обучаются по реальной награде. Бинарный признак, сохранённый для
совместимости V1, новая ценовая голова не читает. Argmax выбирает кандидата ближе всего
к обученному центру; стохастический режим оценивает то же распределение с собственным RNG.

Actor и critic теперь имеют отдельные MLP и Adam, а также независимый clipping. Это
устраняет подавление actor при большом градиенте critic в общем ограничении нормы.
Critic оптимизируется по Huber loss; отдельно продолжает регистрироваться MSE, чтобы
сравнение с V1 не смешивало разные ошибки.

Промежуточная архитектура 2 всё ещё давала большой терминальный остаток: value 50.61
при target 2.78. Прогон сохранён, не отобран по доходности. Архитектура 3 параметризует
значение через обучаемую ставку возврата и известный оставшийся горизонт:

```text
remaining = clamp(horizon - step, 0, horizon)
discount_mass = sum(gamma**k for k in range(remaining))
V(state) = learned_rate(critic_features) * discount_mass
```

В коде используется эквивалентная формула геометрической суммы с отдельными случаями
`gamma=0` и `gamma=1`. При последнем шаге множитель равен 1, после горизонта — 0.
Это структурное ограничение оценки будущего возврата, а не изменение награды.
CLI проверяет совпадение горизонта checkpoint и эксперимента.

## Нейросеть и признаки

Оба MLP имеют вид `85 → Linear(64) → Tanh → Linear(64) → Tanh`.
Actor управляет участием (2), коалицией (2), видом связи (4), центром/шириной цены (2)
и оценкой динамических целей связи (`64+9+4 → 32 → 1`). Critic имеет отдельный MLP
и линейный выход (1). Всего **22 572 параметра**.

Признаки берутся только из AgentObservation: собственные показатели, история, агрегаты
публичных цен, соседи и стоимости связей, доступное предложение, forecasts/evaluation.
У каждого необязательного числового значения есть отдельная маска наличия. Масштабы
фиксированы и сохраняются; значения ограничиваются [-10,10]. Ни число игроков, ни ID
конкурентов не задают размер сети. `features.FEATURE_NAMES` задаёт точный порядок.

Коалиция выбирается только при участии, допустимом offer и отсутствии ограничения.
Присоединение использует точную цену предложения. Связи выбираются иерархически:
сначала None/add/maintain/remove, затем цель. Недоступные варианты маскируются.
При restricted и cost>p_max продажа запрещена самой политикой. Границы не зависят от
вызова внутреннего санитайзера среды.

## Обучение и обратная связь

```text
target = reward / 100 + gamma * (0 if done else V(next_observation))
delta = target - V(observation)
actor_objective = -log_prob(actual_action) * clip(detach(delta), -20, 20)
                  - 0.01 * H(full_conditional_policy)
critic_objective = 0.5 * Huber(V(observation), detach(target))
```

Adam actor: 0.0003; Adam critic: 0.001; gamma: 0.95; max gradient norm: 1 для
каждого оптимизатора отдельно. Это зафиксированные стартовые настройки, не оптимальные.
Полная энтропия вычисляется по правилу цепочки с вероятностями посещения ветвей.
Градиент log_prob включает только действительно исполненные решения; энтропия является
точной регуляризацией всего допустимого дерева и может затрагивать другие допустимые
ветви. Это отличие от старой суммы энтропий только вдоль выбранного пути явно тестируется.

`reset` очищает ожидающий переход, диагностику эпизода и RNG, не веса/Adam.
`act` хранит одно решение без autograd-графа. `observe_transition` проверяет совпадение
наблюдения/действия, использует реальную награду среды, восстанавливает граф и делает
одно обновление. Bootstrap не дифференцируется. Терминальный переход не использует V(next).

Оценка `greedy` детерминирована; `seeded` воспроизводит последовательность при одинаковом
reset/seed и не обучается. Checkpoint сохраняет веса, оба Adam, настройки, схему признаков,
нормализацию, счётчик и RNG. Встроенный train переносит модель между seed через checkpoint.
Существующие результаты не перезаписываются. Выходные пути внутри репозитория разрешены
только под нашим пакетом; фактические результаты проверок размещаются снаружи репозитория.

Checkpoint V1 загружается точной LegacyActorCritic для сравнения, без конвертации весов.
Продолжение обучения V1 запрещено явно. Checkpoint архитектуры 2 тоже сохраняет прежнюю
семантику critic без нового множителя горизонта. Новая архитектура обучается с новой
инициализации; старые модели и отчёты не меняются.

## Модули

* agent.py — контракт, RNG и соответствие отложенных переходов.
* policy.py — точная LegacyActorCritic и действующая ActorCritic.
* features.py / actions.py — разрешённые признаки и кандидаты.
* learning.py / checkpoint.py / settings.py — обновления, хранение, настройки и пути.
* train.py / evaluate.py / metrics.py — эксперименты, индивидуальные показатели и версии.
* diagnostics.py — реальные наблюдения, logits, вероятности, действия и итоги каждого шага.
* math_audit.py — независимая проверка log_prob, энтропии и атрибуция градиентов.
* plot_results.py — PNG/PDF-графики и CSV по сохранённым результатам.
* experiments/ — train, контрольный eval 100–104 и зафиксированный final_eval 200–209.
* tests/ — математические регрессии, контракт, загрузка и настоящие эпизоды среды.

Не используется global_state(), скрытое состояние, внутренности соперников или другая
реализация рынка. Доступ к env.agents[0] нужен только для нашего объекта и его checkpoint.

## Воспроизведение на Windows

Проверено: Python 3.14.3, torch 2.15.0+cpu, NumPy 2.5.4, PyYAML 6.0.3,
NetworkX 3.7, pytest 9.2.0; matplotlib 3.11.2 нужен только для отчётных графиков.
Среда и её pyproject.toml не изменяются. Для нового внешнего окружения:

```powershell
$agentEnv = Join-Path $env:LOCALAPPDATA 'venvs/itmo-market-agent'
py -3.14 -m venv $agentEnv
$python = Join-Path $agentEnv 'Scripts/python.exe'
$package = 'src/social_market_env/implementations/zhmurenko_vladimir'
& $python -m pip install -r "$package/requirements.txt"
& $python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
$env:PYTHONPATH = (Resolve-Path 'src').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$run = Join-Path $env:LOCALAPPDATA 'ITMO/zhmurenko-new-run'
& $python -B -m social_market_env.implementations.zhmurenko_vladimir.train --output-dir "$run/training"
& $python -B -m social_market_env.implementations.zhmurenko_vladimir.evaluate `
  --checkpoint "$run/training/agent_0_seed_2.pt" --output-dir "$run/control"
& $python -B -m social_market_env.implementations.zhmurenko_vladimir.evaluate `
  "$package/experiments/final_eval.yaml" --checkpoint "$run/training/agent_0_seed_2.pt" `
  --reference-checkpoint '<сохранённый checkpoint V1>' --output-dir "$run/final"
```

Это команды подготовки нового окружения; фактическая проверка использовала существующее
внешнее окружение sma-zhmurenko-venv с указанными версиями.
Для воспроизведения старого дефекта:

```powershell
& $python -B -m social_market_env.implementations.zhmurenko_vladimir.diagnostics `
  --checkpoint '<checkpoint V1>' --history-dir '<каталог исходного обучения>' --output-dir "$run/diagnosis"
& $python -B -m social_market_env.implementations.zhmurenko_vladimir.math_audit `
  --checkpoint '<checkpoint V1>' --trace "$run/diagnosis/trained_sampled/seed_100/episode_0/policy.jsonl" `
  --output "$run/math.json"
```

Тесты (назначайте новый внешний каталог временных файлов):

```powershell
$testRoot = Join-Path $env:TEMP ('itmo-test-' + [guid]::NewGuid())
& $python -B -m pytest -q tests "$package/tests" -p no:cacheprovider `
  --basetemp="$testRoot/data" --junitxml="$testRoot/results.xml"
```

Полные результаты каждого эпизода — steps.jsonl и episode.json; индивидуальные показатели —
episodes.json[l]; история всех обновлений — learning.jsonl; сводка — report.json;
точные зависимости, Git SHA и хеши исходников — metadata.json. Seed-парные разности,
дисперсии и исходные модели сохраняются; неудачные эпизоды не исключаются.

Ограничения: одна инициализация обучения, ограниченный диапазон рынков и соперников,
дискретизация цены, отсутствие рекуррентной памяти. Нулевые расходы в демо не доказывают
устойчивость к платным связям. Сохранённые checkpoints разных архитектур нельзя выдавать
за непрерывное обучение одной и той же модели. Проблемы общей среды не исправляются.
