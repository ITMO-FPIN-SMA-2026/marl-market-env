# MARL Social Market Environment — MVP

Минимальный каркас среды **«MARL в динамической сетевой рыночной игре с эндогенными коалициями и социально-ориентированным арбитражем»**.

## Что уже реализовано

- N рыночных агентов вместо жёстко заданных двух агентов из исходного RL-бенчмарка.
- Строгие Python API-контракты для трёх внешних блоков:
  - `MarketAgentAPI`
  - `CoalitionPolicyAPI`
  - `ArbiterAPI`
- Частичная наблюдаемость арбитра: ему **не передаются** истинный граф и истинные коалиции.
- Порядок шага:
  `G_t -> coalition formation -> agent actions -> market clearing -> SW_t -> arbiter action -> G_{t+1}`.
- Линейный спрос `Q(P)=max(0, A-BP)`.
- Прибыль агента `(price - cost) * quantity - fine`.
- Финансовый резерв и выбывание при длительном отрицательном состоянии.
- Динамический граф связей.
- Простая коалиционная политика-заглушка.
- Три типа действий арбитра: наблюдение, изменение стоимости связей, санкция.
- Награда арбитра с компонентами `|SW-SW*|`, вмешательство, FP и FN.
- История каждого шага для последующей визуализации/экспериментов.

## Запуск

```bash
python run_demo.py
```

Никаких ML-библиотек. 

## Как подключится будущий агент

```python
from social_market_env.contracts import MarketAgentAPI, AgentObservation, AgentAction

class PPOAgent(MarketAgentAPI):
    def reset(self, agent_id, seed=None):
        # load weights / reset hidden state
        ...

    def act(self, observation: AgentObservation) -> AgentAction:
        price = self.model.predict(...)
        return AgentAction(price=float(price), participate=True)
```

Среда не знает, PPO это, Q-learning, Transformer или rule-based политика.


