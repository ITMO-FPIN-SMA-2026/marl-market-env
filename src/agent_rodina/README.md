# agent_rodina — рыночный агент «Демпфированный BR»

**Автор:** Polina Rodina / Родина Полина (`repyevapolina`)

**Модуль:** `src/agent_rodina/`. **Роль:** Market Agent (`MarketAgentAPI`).

## Идея политики

Середина между «всегда соглашайся на сговор» (`CartelAgent`) и «никогда не вступай» (`BestResponseAgent`):

1. **Цена** — damped best response к ценам конкурентов на прошлом шаге (сглаживание, чтобы не прыгать на каждом шаге).
2. **Сговор** — вступаю, только если рекомендованная цена близка к моей одиночной BR (`[-join_slack, +max_cartel_premium]` с обучаемым смещением); иначе отказываюсь и подрезаю картель.
3. **Обучение** — при `training=True` по EMA награды слегка подкручиваю damping и join bias.

Параметры по умолчанию: `damping≈0.35`, `join_slack≈0.08`, `max_cartel_premium≈0.12`, `learn_rate≈0.05`.

## Краткие результаты

Head-to-head (только продавец 0 меняется; остальные — `CartelAgent`; seeds `{0,1,2}`, horizon 40):

| Agent 0 | Средняя награда (agent 0) |
| --- | ---: |
| **RodinaAgent** (демпфированный BR) | **49 294.9** |
| BestResponseAgent | 48 970.0 |
| CartelAgent | 7 838.1 |

На слайдах командной презентации метод назван **«Демпфированный BR»**: среди картельщиков прибыль около **+27%** к обычному best response; в сговор входит примерно в половине шагов.

## Подключение

```yaml
agents:
  overrides:
    0:
      impl: "agent_rodina:RodinaAgent"
      # kwargs: {damping: 0.35, join_slack: 0.08, max_cartel_premium: 0.12, learn_rate: 0.05}
```

Готовый конфиг: [`experiments/agent_rodina.yaml`](experiments/agent_rodina.yaml)
(продавец 0 — наш агент; состав соперников как у Askarov).

## Install / run

Из корня репозитория:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q src/agent_rodina/tests
python -m social_market_env.run src/agent_rodina/experiments/agent_rodina.yaml
```

## Структура

```text
src/agent_rodina/
├── README.md
├── __init__.py
├── agent.py
├── experiments/agent_rodina.yaml
└── tests/test_agent_rodina.py
```

Миграция с устаревшего layout `src/rodina_agent/` / `rodina_agent:RodinaAgent`
(PR #4, ветка `agent-rodina`): используйте `agent_rodina:RodinaAgent` и путь выше.
