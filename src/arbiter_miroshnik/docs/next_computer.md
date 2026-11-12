# Продолжение на другом компьютере

Канонический пакет: [`arbiter_miroshnik`](../README.md), компонент
`arbiter_miroshnik:PPOGRUArbiter`. Runtime, инструменты, тесты и документация
находятся внутри пакета. Последние изменения подготовлены в локальной ветке
`arbiter-miroshnik-ppo-gru`; до её публикации перенесите checkout отдельно.

## Установка

Из корня checkout, Python 3.12 или новее:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,learning]'
```

В Windows используйте `.venv\Scripts\Activate.ps1`. Для GPU установите
подходящую сборку PyTorch; проверенная установка AMD RX 6600 XT описана в
[amd.md](amd.md). Для CPU используйте `--device cpu`, для CUDA/ROCm — `cuda`.

## Восстановление архивов

Оба архива находятся в Telegram Downloads этого компьютера:
`/home/harrm/Downloads/Telegram Desktop/`.

- `experiments.tar.gz` — все прежние эксперименты и этапы весов, около 1,26 ГБ.
- `latest_team.tar.gz` — исходная серия на свежих командных коммитах, около 10 МБ.

Перенесите оба файла в корневой `artifacts/` checkout. Manifest с SHA-256
находятся в [`research/handoff/`](../research/handoff). Затем:

```bash
python -m arbiter_miroshnik.research.restore_handoff
```

Утилита проверяет SHA-256 архивов и каждого файла, восстанавливает все этапы
и сохраняет уже существующие состояния. Если состояние изменилось после
восстановления, повторное восстановление в этот каталог прекращается:
используйте отдельный чистый checkout для исходного снимка.
Нужно около 15 ГБ свободного места с учётом зависимостей и распаковки.

Архивные данные сохраняют исторические пути `research/pure_market/runs/`.
Это игнорируемые данные; код прежнего исследовательского проекта не входит
в текущий пакет. Происхождение файлов и старые абсолютные пути оставлены
в manifest без изменения. Начальная штатная серия копируется в
`runs/transferred_ppo`, свежая серия восстанавливается в
`runs/transferred_ppo_latest`.

## Оценивание и продолжение

```bash
python -m arbiter_miroshnik evaluate --episodes 20 --device cpu
python -m arbiter_miroshnik train --games 128 --device cpu
python -m arbiter_miroshnik evaluate --out runs/transferred_ppo_latest --episodes 20 --device cpu
```

`--games` задаёт целевое общее число обучающих игр каждой из трёх моделей.
Повторный запуск продолжает actor, critic, Adam, траектории и состояния четырёх
продавцов в обеих ветвях. Одновременно запускайте только один процесс обучения
на каталог. Новые отчёты и этапы не заменяют прежние.

Архивы содержат исходные снимки: по 64 игры на модель в основной серии и по
8 игр в свежей. В текущем локальном checkout основная серия продолжена до
128, свежая — до 64, отдельная AMD-серия `runs/transferred_ppo_amd` — до 72.
Эти новые состояния в архивы и Git не входят; для продолжения именно их
перенесите соответствующие каталоги `runs/` отдельно.

Командные реализации и настройки закреплены внутри каждой серии в
`team_sources` и `sources.json`. Обновление checkout не заменяет их: для новой
популяции нужны отдельная калибровка и каталог исследования.
Актуальные результаты и ограничения: [experiments.md](experiments.md).
Польза PPO + GRU и точность санкций пока не подтверждены.

## Проверки

```bash
python -m pytest -q
python -m arbiter_miroshnik --help
python -m social_market_env.run src/arbiter_miroshnik/arbiter_miroshnik.yaml
```

Последняя команда требует восстановленного `initial_model_0.pt`; это короткая
проверка подключения без коалиций. Тест обучения и полного возобновления
checkpoint находится в [`research/tests/`](../research/tests).
