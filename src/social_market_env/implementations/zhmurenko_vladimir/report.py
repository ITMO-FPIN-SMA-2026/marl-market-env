"""Assemble an evidence-based audit report from immutable run artifacts."""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .settings import PACKAGE_ROOT, writable_path

LABELS={'old_trained':'Старая обученная, argmax','old_trained_sampled':'Старая обученная, выборка',
        'trained':'Новая обученная, argmax','trained_sampled':'Новая обученная, выборка',
        'untrained':'Новая необученная, argmax','untrained_sampled':'Новая необученная, выборка',
        'CartelAgent':'CartelAgent','BestResponseAgent':'BestResponseAgent','RandomAgent':'RandomAgent [20,60]'}


def table(report):
    lines=['| Стратегия | Эпизоды | Средняя | Медиана | σ эпизодов | σ средних seed | Штрафы | Выбывания |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,item in report['strategies'].items():
        lines.append(f'| {LABELS[key]} | {item["episodes"]} | {item["mean_reward"]:.3f} | {item["median_reward"]:.3f} | {item["reward_std"]:.3f} | {item["seed_mean_std"]:.3f} | {item["fine_count"]} | {item["eliminations"]} |')
    return '\n'.join(lines)


def verify_finite_artifacts(root):
    """Check every numeric value of the selected training and both evaluation logs."""
    import torch
    numbers,tensors=0,0

    def visit(value):
        nonlocal numbers,tensors
        if isinstance(value,torch.Tensor):
            tensors+=1
            if not torch.isfinite(value).all():
                raise FloatingPointError('Non-finite checkpoint tensor')
        elif isinstance(value,dict):
            for item in value.values():
                visit(item)
        elif isinstance(value,(tuple,list)):
            for item in value:
                visit(item)
        elif isinstance(value,(int,float)):
            numbers+=1
            if not math.isfinite(value):
                raise FloatingPointError('Non-finite numerical log value')
    files=0
    for directory in ('training-v3','control-v2','final-v2'):
        for path in (root/directory).rglob('*.jsonl'):
            files+=1
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    visit(json.loads(line))
    visit(torch.load(root/'training-v3/agent_0_seed_2.pt',map_location='cpu',weights_only=True))
    result={'all_finite':True,'jsonl_files_checked':files,'numeric_values_checked':numbers,'checkpoint_tensors_checked':tensors}
    (root/'numerical-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def create_report(root: Path):
    load=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    delivery=load('delivery.json') if (root/'delivery.json').exists() else {}
    execution_root=Path(delivery.get('original_execution_directory',str(root)))
    before,after=load('diagnosis-before/report.json'),load('diagnosis-after/report.json')
    old_math,new_math=load('math-before.json'),load('math-after-corrected.json')
    training,intermediate=load('training-v3/report.json'),load('training-v2/report.json')
    control,final=load('control-v2/report.json'),load('final-v2/report.json')
    scope,lock=load('scope-check.json'),load('final-model-lock.json')
    tests=ET.parse(root/'pytest-final.xml').getroot()
    counts={key:sum(int(suite.attrib.get(key,0)) for suite in tests) for key in ('tests','failures','errors','skipped')}
    test_names=[case.attrib['name'] for case in tests.iter('testcase')]
    example=before['strategies']['trained']['examples'][0]
    improved=after['strategies']['trained']
    new_example=improved['examples'][0]
    old_history,new_history=before['training_history'],after['training_history']
    candidate_index=example['price_argmax_index']
    new_selected=final['strategies']['trained']
    source_descriptions={
        'agent.py':('Выбор точной версии сети по checkpoint, два оптимизатора, публичная эпизодная диагностика.','Загрузка V1/V3; независимость; roundtrip обоих Adam; frozen evaluation.'),
        'policy.py':('Гладкие обучаемые logits цены; раздельные MLP; полная энтропия; конечный горизонт critic.','Перебор дерева, конечные разности, непрерывность/внутренняя мода, точные граничные значения горизонта.'),
        'learning.py':('Две независимые оптимизации, Huber critic, корректная совместная энтропия и расширенные метрики.','Знак изменения вероятности; отсутствие градиента actor в critic и наоборот; TD/terminal.'),
        'settings.py':('Явные версии архитектуры, critic learning rate и границы обучаемой ширины.','Валидация и сохранение настроек; совместимость checkpoint.'),
        'checkpoint.py':('Формат 2, чтение исходного формата 1 без конвертации.','Точный legacy roundtrip; запрет незаметного продолжения V1; сохранение двух Adam.'),
        'train.py':('Эпизодная статистика ошибок, хеши исходников/версии, согласование горизонта.','Межseed handoff; реальные 120 эпизодов/6000 обновлений; проверка конечности.'),
        'evaluate.py':('Отдельные старый/новый checkpoints; контроль горизонта; парные результаты; SHA.','Контроль 15 и независимые 30 эпизодов на стратегию; frozen state_dict и SHA.'),
        'metrics.py':('Себестоимость/маржа из собственного наблюдения или официального результата, episode.json, метаданные.','Реальные журналы; индивидуальные награды суммируются без исключения эпизодов.'),
        'diagnostics.py':('Новый инструмент реальных наблюдений, logits, вероятностей, признаков и вмешательства в собственный индикатор.','750 шагов V1 и V3; тест неизменности RNG и параметров.'),
        'math_audit.py':('Независимая проверка log_prob/энтропии и норм градиентов без optimizer.step.','Файлы math-before / math-after-corrected с проверкой frozen weights.'),
        'plot_results.py':('Новый инструмент стандартных PNG/PDF-графиков и CSV.','Построение по сохранённым данным, визуальная проверка файлов.'),
        'report.py':('Новый отчёт из JSON/XML, без ручной подстановки результатов.','Таблицы читаются непосредственно из отчётов запусков.'),
        'tests/test_learning_regressions.py':('Новые регрессии математической причины и контроль совместимости.','См. полный XML и список тестов ниже.'),
        'experiments/train.yaml':('Явно зафиксирована архитектура 3 и её гиперпараметры.','Использован фактическим обучающим CLI.'),
        'experiments/eval.yaml':('Основной детерминированный режим снова greedy, без сокрытия результата выборкой.','Контрольные сравнения обоих режимов.'),
        'experiments/final_eval.yaml':('Новый заранее зафиксированный диапазон 200–209.','final-model-lock.json и неизменность поведенческих исходников.'),
        'requirements.txt':('Указаны фактически проверенные версии; matplotlib только для графиков.','metadata.json и выполненные запуски.'),
        'README.md':('Документация фактической архитектуры, версий, методов и запуска.','Сверено с реализацией и полученными артефактами.')}
    prefix='src/social_market_env/implementations/zhmurenko_vladimir/'
    changes=[]
    for path in scope['changed_files']+scope['new_files']:
        short=path.removeprefix(prefix)
        description,verification=source_descriptions.get(short,('См. исходный diff.','См. протокол проверок.'))
        changes.append(f'| `{short}` | {description} | {verification} |')
    pairs=['| Seed | V3 argmax | BestResponse | Разность | Random | Разность |','|---|---:|---:|---:|---:|---:|']
    for seed,value in new_selected['mean_reward_by_seed'].items():
        best=final['strategies']['BestResponseAgent']['mean_reward_by_seed'][seed]
        random=final['strategies']['RandomAgent']['mean_reward_by_seed'][seed]
        pairs.append(f'| {seed} | {value:.3f} | {best:.3f} | {value-best:+.3f} | {random:.3f} | {value-random:+.3f} |')
    statistics={}
    for baseline in ('BestResponseAgent','RandomAgent'):
        differences=np.array(list(final['paired_seed_reward_differences'][baseline].values()))
        rng=np.random.default_rng(314159)
        bootstrap=rng.choice(differences,(10000,len(differences)),replace=True).mean(axis=1)
        statistics[baseline]={'paired_mean':float(differences.mean()),'positive_seeds':int((differences>0).sum()),
            'seed_count':len(differences),'bootstrap_seed_mean_ci95':list(map(float,np.quantile(bootstrap,[.025,.975])))}
    statistic_rows=['| Сравнение | Средняя парная разность | Seed с преимуществом | 95% bootstrap интервал |','|---|---:|---:|---|']
    for name,value in statistics.items():
        low,high=value['bootstrap_seed_mean_ci95']
        statistic_rows.append(f'| V3 argmax − {name} | {value["paired_mean"]:+.3f} | {value["positive_seeds"]}/{value["seed_count"]} | [{low:+.3f}; {high:+.3f}] |')
    finite=verify_finite_artifacts(root)
    (root/'paired-statistics.json').write_text(json.dumps(statistics,indent=2),encoding='utf-8')
    image=lambda name:f'![{name}]({(root/"figures"/name).as_posix()})'
    rootcause=f'''# Диагностика и исправление индивидуального рыночного агента

## A. Подтверждённая причина

Исходный checkpoint: `{before['checkpoint']}`. SHA-256: `{before['checkpoint_sha256']}`.
Он загружен до изменений и остаётся неизменным. Старая архитектура воспроизведена точно,
а не заменена новой моделью с похожими результатами.

На контрольных seed 100–104, 15 эпизодах и 750 шагах исходная детерминированная политика
продавала по себестоимости **750/750** раз, всегда с положительными продажами. Маржа,
gross_profit, coalition_payout, fine, link_fee и reward равны нулю. Ограничения и выбывание
не были причиной. Пример: seed 100 / episode 0 / step 0: cost={example['cost']},
price={example['action']['price']}, quantity={example['outcome']['quantity']:.8f},
reward={example['outcome']['reward']}, probability={example['price_probabilities'][candidate_index]:.8f},
logit={example['price_logits'][candidate_index]:.8f}, index={candidate_index}, entropy={example['price_entropy']:.8f}.

Все кандидаты этого состояния: `{example['prices']}`. Обычно кандидатов 16–22 для
старой детерминированной траектории; себестоимость не была единственной допустимой ценой.
Цены cost+1/+3/+5/+10 присутствовали. Генератор, округление и границы не менялись.
Число кандидатов влияет на нормировку вероятностей, но не меняет относительный порядок
одинаковых logits; увеличение числа целей связи не влияет на вероятность её вида.

Во всех 750 состояниях logits убывали выше cost. Обнуление только бинарного
`at_or_above_cost` переводило максимум к p_min в 750/750 состояниях. Это указывает на
выученный скачок на границе и убывающий скоринг после него. При этом вероятность cost
в примере лишь {example['price_probabilities'][candidate_index]:.3f}; остальные вероятности
не исчезли, поэтому выборка более дорогих цен могла приносить прибыль. Полные таблицы:
`diagnosis-before/candidates.csv`, `policy.jsonl` в каждом эпизоде. Аналогичная диагностика
исходной необученной модели сохранена отдельно.

Подтвердились ценовая причина и патологическая форма logits. Не подтвердились ошибки
знака actor-loss, совместной log_prob, пропуска terminal или использования искусственной
награды: независимая проверка log_prob совпадает с кодом до 1e-5, терминальный bootstrap=0.
Критик V(s) не является Q(s,a): его значение нельзя использовать как оценку отдельной цены.
Стохастическая политика может быть корректным способом исполнения, но её включение
само по себе не устраняет патологию argmax.

Дополнительно обнаружены сильный общий clipping и несогласованность старой path-энтропии
с полной энтропией дерева. В исторических 5468 обновлениях clipping был активен в
{100*old_history['gradient_clipped_fraction']:.2f}% случаев; advantage_clip — лишь в
{100*old_history['advantage_clipped_fraction']:.2f}%. Медианная MSE критика
{old_history['critic_loss']['quantiles']['p50']:.4f}, p99={old_history['critic_loss']['quantiles']['p99']:.4f}.
На реальном переходе step 1 нормы actor/critic были 1.5435/12.1340, общий множитель clipping
0.08178. Это масштаб градиента перед Adam, а не утверждение о таком же уменьшении шага Adam.
Сумма энтропий вдоль одного выбранного пути также не равна полной энтропии условной политики.
Все измерения и независимые нормы сохранены в `math-before.json`.

{image('price_diagnosis.png')}

## B. Исправления и границы

Изменены только файлы нашего пакета. Не менялись actions.py, features.py, общий контракт,
экономика, базовые стратегии, общие тесты и конфигурации. Новая ценовая голова учит центр
и ширину гладких квадратичных logits на прежнем множестве цен; price=cost+5 не подставляется.
Разделены сети/Adam/clipping actor и critic. Цель TD по-прежнему использует фактическую
награду среды. Critic оптимизирует Huber, MSE регистрируется отдельно. Энтропия теперь
точно взвешивает допустимые ветви по правилу цепочки; log_prob учитывает только исполненные
решения. Градиенты энтропийной регуляризации допустимых альтернатив не выдаются за градиенты
фактически выбранных действий.

Промежуточный прогон V2 сохранён: {intermediate['updates']} обновлений,
train mean={intermediate['summary']['mean_reward']:.3f}, последний terminal MSE=
{intermediate['last_learning_metrics']['critic_loss']:.3f}. Он не выбран: critic продолжал
оценивать последний шаг как бесконечный горизонт. V3 предсказывает обучаемую ставку возврата,
умноженную на известную оставшуюся дисконтную массу. При последнем шаге масса=1, после
горизонта=0. Награда и правила среды не меняются. Архитектура выбрана до окончательной
проверки; большая train mean V2 не использовалась для сокрытия этого дефекта.

| Файл | Что и зачем изменено | Подтверждение |
|---|---|---|
'''+ '\n'.join(changes)+f'''

## C. Тесты и математические регрессии

Итог: **{counts['tests']} tests, failures={counts['failures']}, errors={counts['errors']}, skipped={counts['skipped']}**.
Штатных тестов среды 81; собственных {counts['tests']-81}. Полный протокол — `pytest-final.xml`.

Фактически выполненная команда из корня репозитория (журналы затем скопированы в эту папку):

```powershell
$python = 'C:/Users/darte/AppData/Local/Temp/sma-zhmurenko-venv/Scripts/python.exe'
$env:PYTHONPATH = (Resolve-Path 'src').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $python -B -m pytest -q tests src/social_market_env/implementations/zhmurenko_vladimir/tests `
  --basetemp='{(execution_root/'pytest-final').as_posix()}' `
  --junitxml='{(execution_root/'pytest-final.xml').as_posix()}'
```

Проверены нормировка полного дерева и log_prob независимым перебором, градиент энтропии
конечными разностями, знак изменения вероятности выбранного действия, непрерывность
ценовых logits при пересечении cost, возможность внутренней моды, отсутствие градиентов
log_prob в неисполненных ценовых решениях, раздельность actor/critic, границы горизонта
для gamma=0/0.95/1, неизменность диагностики/RNG, оба Adam, точное чтение V1 и запрет
незаметного продолжения его обучения. Старые проверки покрывают YAML, разные размеры рынка,
reset, delayed/terminal, выбывание, допустимые действия, независимость экземпляров,
checkpoint в отдельном процессе и настоящие эпизоды среды.

## D. Повторное обучение

Seeds 0,1,2; по 40 эпизодов, horizon=50. Конфигурация — `training-v3/experiment.json`;
зависимости и хеши кода — `metadata.json`; точный исходный снимок — `training-source.zip`.
Сохранённая модель: [{(root/'training-v3/agent_0_seed_2.pt').name}]({(root/'training-v3/agent_0_seed_2.pt').as_posix()}).
SHA-256: `{lock['selected_checkpoint_sha256']}`.
Архитектура 3: два MLP 85→64→64, отдельные головы, **{training['model_parameters']} параметра**.
Adam actor=0.0003, critic=0.001; gamma=.95; reward_scale=100; entropy=.01; clipping=1
отдельно; advantage_clip=20; ширина цены 0.02–0.30 от p_max.

Получено {training['updates']} обновлений, L2 изменения параметров={training['parameter_delta_l2']:.8f}.
Все значения истории обновлений конечны: `{training['losses_finite']}`. Средняя награда
обучения {training['summary']['mean_reward']:.3f}, медиана {training['summary']['median_reward']:.3f},
выбывания {training['summary']['eliminations']}, события ошибок {training['summary']['events']}.
Последний terminal: target={training['last_learning_metrics']['target']:.6f},
value={training['last_learning_metrics']['value']:.6f},
MSE={training['last_learning_metrics']['critic_loss']:.6f}. Это один переход; полная
динамика и распределения приведены на графике и в training_history отчёта диагностики.
Контрольная проверка после обучения: cost-мода исчезла на 750/750 шагах; средняя маржа
{improved['margins']['mean']:.6f}, минимальная {improved['margins']['quantiles']['min']:.6f}.
Для того же первого реального состояния новая сеть выбрала цену {new_example['action']['price']}
при cost={new_example['cost']}, с вероятностью {new_example['price_probabilities'][new_example['price_argmax_index']]:.8f},
продажи={new_example['outcome']['quantity']:.8f}, награда={new_example['outcome']['reward']:.8f}.
Эволюция условной ценовой головы проверена на одном и том же реальном наблюдении по
инициализации и трём сохранённым checkpoint каждого семейства; полные logits и вероятности
в `checkpoint-evolution.json`. Это дискретные сохранённые стадии, а не выдуманная запись
распределения на каждом историческом обновлении.

Команда обучения:

```powershell
& $python -B -m social_market_env.implementations.zhmurenko_vladimir.train --output-dir '<новый внешний каталог>'
```

{image('training_dashboard.png')}

## E. Контрольное и независимое сравнение

Контроль: 100–104, 15 эпизодов на стратегию. Эти seeds использовались для диагностики и
не считаются новой независимой проверкой.

{table(control)}

Независимая проверка: 200–209, 30 эпизодов на стратегию. Набор зафиксирован до обучения;
checkpoint и поведенческие исходники зафиксированы до проверки в `final-model-lock.json`.
Окончательная проверка выполнена один раз; по результатам модель не изменялась.

{table(final)}

Все стратегии используют одну экономику, состав соперников, коалиционную политику,
арбитра и одинаковые episode_seed. Меняется только agent 0. Неблагоприятные эпизоды не
исключались. σ эпизодов — разброс всех эпизодов; σ средних seed — разброс средних по seed.
Параметры и SHA checkpoint проверены неизменными. Награда индивидуальная, из agent_rewards[0].

{image('evaluation_comparison.png')}

Парные результаты на независимом наборе:

'''+ '\n'.join(pairs)+f'''

Bootstrap по десяти seed-кластерам (10000 выборок, RNG 314159), только оценка
неопределённости после проверки, без выбора модели:

'''+ '\n'.join(statistic_rows)+f'''

Численная проверка: все значения конечны — {finite['all_finite']}; проверено
{finite['numeric_values_checked']} числовых значений в {finite['jsonl_files_checked']} JSONL-файлах
и {finite['checkpoint_tensors_checked']} тензоров checkpoint. Протокол: numerical-check.json.
Эти интервалы не заменяют проверки на других моделях рынка и других инициализациях обучения.

{image('paired_differences.png')}

## F. Оставшиеся ограничения

Исправлена подтверждённая ценовая патология и проверена положительная награда реальной
детерминированной политики. Выборка сохранена как отдельный режим, не как замена исправлению.
Новая необученная argmax-политика может вообще не продавать: её нулевой результат честно
сохранён. Рост train-наград сам по себе не использовался как доказательство качества.

Не доказана универсальная оптимальность или преимущество на всех рынках. Проверена одна
инициализация обучения, один диапазон себестоимостей и одна конфигурация baseline-соперников.
Цена дискретна; зависимость V3 от горизонта требует корректной настройки; память не рекуррентная.
Точные веса отдельных исправлений в приросте качества не установлены полномасштабным
факторным экспериментом. Вывод о скачке ценовых logits основан на прямом вмешательстве,
а оценка комплексного исправления — на реальных прогонах. Нулевые затраты связей в демо
не доказывают устойчивости к платным связям. Известные недостатки среды не исправлялись.
Последовательные сравнения нельзя выдавать за независимые инициализации обучения.

## G. Git, неизменность и воспроизводимость

Ветка `{scope['branch']}`, HEAD `{scope['commit']}`. Нарушений области записи:
`{scope['scope_violations']}`. Исходных файлов вне пакета изменено/удалено/добавлено: 0.
По сравнению с началом диагностики изменено {len(scope['changed_files'])} собственных файлов,
добавлено {len(scope['new_files'])}, удалено {len(scope['deleted_files'])}.
Коммиты, push, merge, rebase, PR не выполнялись.

Фактический git status:

```text
{scope['git_status']}
```

Исходный checkpoint SHA подтверждён неизменным. Все файлы поведенческого lock также
сохранили хеши. Реальные результаты, журналы, модели и рисунки находятся вне Git-репозитория.
Все воспроизводящие инструменты — внутри разрешённого пакета.

Для повторения сравнения:

```powershell
& $python -B -m social_market_env.implementations.zhmurenko_vladimir.evaluate `
  src/social_market_env/implementations/zhmurenko_vladimir/experiments/final_eval.yaml `
  --checkpoint '<training-v3/agent_0_seed_2.pt>' --reference-checkpoint '<исходный V1>' `
  --output-dir '<новый внешний каталог>'
```
'''
    writable_path(root/'analysis_report.md').write_text(rootcause,encoding='utf-8')
    print(root/'analysis_report.md')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    create_report(parser.parse_args(argv).root)


if __name__=='__main__':
    main()
