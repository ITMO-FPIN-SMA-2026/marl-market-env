# agent_rodina — рыночный агент «Демпфированный BR»

**Автор:** Polina Rodina / Родина Полина (`repyevapolina`)

**Модуль:** `src/agent_rodina/`. **Роль:** Market Agent (`MarketAgentAPI`).

## Идея политики

Середина между «всегда соглашайся на сговор» (`CartelAgent`) и «никогда не вступай» (`BestResponseAgent`):

1. **Цена** — damped best response к ценам конкурентов на прошлом шаге (сглаживание, чтобы не прыгать на каждом шаге).
2. **Сговор** — вступаю, только если **ожидаемая награда в сговоре** не хуже ожидаемой соло-награды (с небольшим допуском `join_slack` и обучаемым смещением). Считаю через модель спроса среды (`clear_market`): соло-прибыль при своей цене vs доля в общем котле при рекомендованной цене.
3. **Если сговор не сложится** — среда **не откатывает** цену: согласие уже означает продажу по рекомендованной цене. Если другие отказались, я получаю только свою валовую прибыль по этой (часто завышенной) цене. В наблюдении нет вероятности распада, поэтому смешиваю два исхода с параметром `formation_prior`:  
   `ожидание = p·доля_в_котле + (1−p)·прибыль_одна_по_картельной_цене`.
4. **Обучение** — при `training=True` по EMA награды слегка подкручиваю damping и join bias (bias влияет и на `p`, и на допуск сравнения).

Параметры по умолчанию: `damping≈0.35`, `join_slack≈0.08`, `formation_prior≈0.7`, `min_payout_weight≈0.05`, `learn_rate≈0.05`.  
(`max_cartel_premium` оставлен для совместимости конфигов, на решение о вступлении больше не влияет.)

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
      # kwargs: {damping: 0.35, join_slack: 0.08, formation_prior: 0.7, learn_rate: 0.05}
```

Готовый конфиг: [`experiments/agent_rodina.yaml`](agent_rodina.yaml)
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
