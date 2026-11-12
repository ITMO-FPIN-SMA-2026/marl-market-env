"""Deployment selection and a short, honest report for the reduced Q-only study."""

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np

from social_market_env.config import load_config
from ..agent import PACKAGE, output_path
from .components import MarkupStrategy
from .experiment import (BASE_CONFIG, EVAL_SCENARIOS, FINAL_SEEDS, IMPL, REPOSITORY, TRAIN_SCENARIOS,
                         VALIDATION_SEEDS, aggregate, digest, model_variants, play, scenario_config, write_json)

LABELS = {'cold':'До обучения', 'economic':'Простая наценка', 'flat':'Обученная общая Q', 'regime':'Q с режимной специализацией'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def group(policy):
    return policy.rsplit('_',1)[0] if policy.startswith(('flat_','regime_')) else policy


def save_csv(path, rows):
    if not rows:
        return
    fields=sorted({k for r in rows for k,v in r.items() if not isinstance(v,(dict,list))})
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def economic_validation(root):
    path=root/'economic-validation.json'
    if path.exists():
        return load(path)
    base=load_config(BASE_CONFIG); agent=MarkupStrategy(market=asdict(base.market));rows=[]
    for i,seed in enumerate(VALIDATION_SEEDS):
        for scenario in TRAIN_SCENARIOS:
            config,pos=scenario_config(base,scenario=scenario,index=i*6,position_index=i,
                focal={'impl':IMPL,'kwargs':{'fresh':True}},fast=True)
            rows.append(dict(play(agent,config,position=pos,seed=seed,training=False),scenario=scenario))
    result={s:aggregate([r for r in rows if r['scenario']==s]) for s in TRAIN_SCENARIOS}
    write_json(path,result)
    return result


def bundle(root):
    if (root/'final-lock.json').exists():
        raise ValueError('Deployment selection must happen before final evaluation')
    for rep in range(3):
        if not (root/f'repeat_{rep}'/'summary.json').exists():
            raise ValueError(f'Training repeat {rep} is still running')
    reference=economic_validation(root); ranked=[]
    for rep in range(3):
        for name in ('flat','regime'):
            path=root/f'repeat_{rep}'/name/'model.json';payload=load(path)
            validation=payload['metadata']['last_validation']
            ratios=[validation['per_scenario'][s]['reward']/max(reference[s]['reward'],1) for s in TRAIN_SCENARIOS]
            ranked.append(dict(policy=f'{name}_{rep}',path=str(path),worst_ratio=min(ratios),
                               mean_ratio=mean(ratios),survival=validation['survival'],
                               validation_reward=validation['reward'],episodes=payload['metadata']['episodes']))
    ranked.sort(key=lambda r:(r['survival'],r['worst_ratio'],r['mean_ratio']),reverse=True)
    selected=ranked[0];destination=PACKAGE/'models'/'q_agent.json';destination.parent.mkdir(exist_ok=True)
    shutil.copy2(selected['path'],destination)
    published_ranking = [{key: value for key, value in item.items() if key != 'path'} for item in ranked]
    provenance=dict(selected=published_ranking[0],ranking=published_ranking,
                    selection='validation_only_maximin_relative_reward_then_mean',
                    model_sha256=digest(destination),validation_seeds=VALIDATION_SEEDS,
                    final_seeds_not_used=True)
    write_json(PACKAGE/'models'/'provenance.json',provenance)
    write_json(root/'deployment-selection.json',provenance)
    print(f"Bundled {selected['policy']}: {destination}; SHA={provenance['model_sha256']}",flush=True)


def paired(rows, first, second, scenarios=None):
    selected=[r for r in rows if scenarios is None or r['scenario'] in scenarios]
    by=defaultdict(list)
    for r in selected:
        by[(r['policy'],r['seed'])].append(r['total_reward'])
    seeds=sorted({r['seed'] for r in selected})
    matrix=[]
    for rep in range(3):
        a=f'{first}_{rep}'
        b=f'{second}_{rep}' if second in ('flat','regime') else second
        matrix.append([mean(by[(a,s)])-mean(by[(b,s)]) for s in seeds])
    delta=np.array(matrix,dtype=float);rng=np.random.default_rng(20261106)
    r=rng.integers(0,3,size=(5000,3));s=rng.integers(0,len(seeds),size=(5000,len(seeds)))
    samples=delta[r[:,:,None],s[:,None,:]].mean(axis=(1,2))
    ci=np.quantile(samples,[.025,.975]).tolist()
    return dict(first=first,second=second,mean_difference=float(delta.mean()),ci95=ci,
                per_training_repeat=delta.mean(axis=1).tolist(),seeds=len(seeds),training_repeats=3,
                method='crossed bootstrap of paired training repeats and root-seed clusters',
                category='positive' if ci[0]>0 else 'negative' if ci[1]<0 else 'not_established')


def scope(root):
    initial=load(root/'initial-state.json')
    violations=[p for p,v in initial['protected'].items() if digest(REPOSITORY/p)!=v]
    status=subprocess.check_output(['git','status','--short','--untracked-files=all'],cwd=REPOSITORY,text=True)
    for line in status.splitlines():
        path=line[3:].replace('\\','/')
        if not path.startswith(('src/zhmurenko_agent/', 'src/social_market_env/implementations/zhmurenko_vladimir/')) and path!='experiments/zhmurenko_agent.yaml':
            violations.append(path)
    old=load(root/'v4-preservation.json')
    old_violations=[p for p,v in old.items() if digest(p)!=v]
    result=dict(outside_scope_changes=violations,V4_preserved=not old_violations,
                V4_changed_files=old_violations,protected_files=len(initial['protected']),git_status=status,
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPOSITORY,text=True).strip(),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPOSITORY,text=True).strip())
    write_json(root/'scope-check.json',result)
    if violations or old_violations or result['commit']!=initial['base']:
        raise AssertionError('Scope or previous results changed')
    return result


def render_html(root):
    command="$body=[IO.File]::ReadAllText($env:Q_REPORT); ($body | ConvertFrom-Markdown).Html"
    completed=subprocess.run(['pwsh','-NoProfile','-Command',command],capture_output=True,text=True,encoding='utf-8',
        env=dict(os.environ,Q_REPORT=str(root/'REPORT.md')),check=True)
    body=completed.stdout
    for path in (root/'figures').glob('*.png'):
        body=body.replace('figures/'+path.name,'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode())
    html='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Q-learning Жмуренко</title><style>body{font:16px/1.6 Segoe UI,sans-serif;max-width:1100px;margin:30px auto;padding:0 24px}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #bbb;padding:6px 9px}img{max-width:100%}pre{overflow:auto;background:#f4f4f4;padding:12px}</style>'+body+'</html>'
    (root/'REPORT.html').write_text(html,encoding='utf-8')


def figures(root,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for rep in range(3):
        history=load(root/f'repeat_{rep}'/'validation.json')
        for name,color in (('flat','#b87318'),('regime','#277e52')):
            data=[r for r in history if r['model']==name]
            for ax,key,title in ((axes[0],'reward','Награда валидации'),(axes[1],'policy_drift','Изменение выбранных действий')):
                ax.plot([r['trained_episodes'] for r in data],[r[key] for r in data],color=color,
                        linestyle=('-','--',':')[rep],label=f'{name}, запуск {rep}')
                ax.set_title(title);ax.set_xlabel('Эпизоды обучения');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(folder/'training.png',dpi=150);plt.close(fig)
    effect=stats['mechanism'];fig,ax=plt.subplots(figsize=(7,3.5))
    ax.bar(range(3),effect['per_training_repeat'],color='#277e52');ax.axhline(0,color='black',lw=.8)
    ax.set_xticks(range(3),['Запуск 0','Запуск 1','Запуск 2']);ax.set_ylabel('regime − flat, награда')
    ax.set_title('Эффект механизма в меняющихся ценах');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(folder/'mechanism.png',dpi=150);plt.close(fig)


def report(root):
    rows=load(root/'final'/'episodes.json')
    expected={name for name,_ in model_variants(root)}
    assert {r['policy'] for r in rows}==expected and len(rows)==672
    hashes=defaultdict(set)
    for r in rows:
        hashes[(r['seed'],r['scenario'])].add(r['initial_conditions'])
    assert len(hashes)==84 and all(len(v)==1 for v in hashes.values())
    stats=dict(mechanism=paired(rows,'regime','flat',('wave','wave2')),
               overall_mechanism=paired(rows,'regime','flat'),
               regime_learning=paired(rows,'regime','economic'),flat_learning=paired(rows,'flat','economic'))
    write_json(root/'statistics.json',stats);save_csv(root/'episodes.csv',rows)
    strata=[]
    for scenario in EVAL_SCENARIOS:
        for name in LABELS:
            strata.append(dict(scenario=scenario,policy=name,**aggregate([r for r in rows if group(r['policy'])==name and r['scenario']==scenario])))
    save_csv(root/'by-scenario.csv',strata)
    training=[]
    for rep in range(3):
        summary=load(root/f'repeat_{rep}'/'summary.json')
        for name in ('flat','regime'):
            payload=load(root/f'repeat_{rep}'/name/'model.json')
            training.append(dict(repeat=rep,policy=name,episodes=payload['metadata']['episodes'],updates=payload['updates'],
                shared_states=len(payload['shared']),regime_states=len(payload['regimes']),
                coalition_samples=sum(r['visits'][10] for r in payload['shared']),
                stability_reached=summary['stability_reached']))
    save_csv(root/'training-summary.csv',training)
    check=scope(root);figures(root,stats)
    provenance=load(root/'deployment-selection.json');selected=provenance['selected']['policy']
    delivered=aggregate([r for r in rows if r['policy']==selected])
    simple=aggregate([r for r in rows if r['policy']=='economic'])
    testroot=ET.parse(root/'tests-final.xml').getroot()
    testcounts={key:sum(int(s.attrib.get(key,0)) for s in testroot) for key in ('tests','failures','errors','skipped')}
    replay=load(root/'reproduction/result.json')
    tables=['| Политика | Эпизоды | Награда | Цена | Продажи/шаг | Наценка | Коалиционные шаги |',
            '|---|---:|---:|---:|---:|---:|---:|']
    coalition=[]
    for name in LABELS:
        subset=[r for r in rows if group(r['policy'])==name];a=aggregate(subset)
        tables.append(f'| {LABELS[name]} | {a["episodes"]} | {a["reward"]:.2f} | {a["price"]:.2f} | {a["quantity"]:.2f} | {a["margin"]:.2f} | {a["coalition_steps"]} |')
        if name in ('flat','regime'):
            coalition.append(dict(policy=name,known=a['known_fraction'],learned=a['learned_fraction'],changed=a['changed_from_cold'],
                specialized=sum(r.get('specialized_decisions',0) for r in subset)/sum(r['decisions'] for r in subset),
                regime_changed=sum(r.get('regime_changed_decisions',0) for r in subset)/sum(r['decisions'] for r in subset),
                high_offers=sum(r.get('high_weight_offers',0) for r in subset),high_joins=sum(r.get('high_weight_joins',0) for r in subset),
                low_offers=sum(r.get('low_weight_offers',0) for r in subset),low_joins=sum(r.get('low_weight_joins',0) for r in subset)))
    save_csv(root/'coalition-and-learning.csv',coalition)
    scenario_table=['| Состав соперников | flat | regime | До обучения | Простая стратегия |','|---|---:|---:|---:|---:|']
    for scenario in EVAL_SCENARIOS:
        scenario_table.append('| '+scenario+' | '+' | '.join(f'{next(r["reward"] for r in strata if r["policy"]==name and r["scenario"]==scenario):.2f}' for name in ('flat','regime','cold','economic'))+' |')
    effect=stats['mechanism'];category={'positive':'положительный эффект','negative':'ухудшение','not_established':'преимущество не установлено'}[effect['category']]
    text=f'''# Краткий отчёт: Q-Learning агента Жмуренко

Основной вывод о структурном механизме: **{category}**.
На меняющихся ценах regime−flat = **{effect['mean_difference']:+.2f}** награды за эпизод;
95% crossed-bootstrap интервал **[{effect['ci95'][0]:+.2f}; {effect['ci95'][1]:+.2f}]**.
Разности трёх независимых обучений: {', '.join(f'{v:+.2f}' for v in effect['per_training_repeat'])}.

Для команды поставляется **{selected}**, выбранная по валидации до итоговой проверки.
Её средняя итоговая награда **{delivered['reward']:.2f}**, у простой стратегии в тех же
условиях **{simple['reward']:.2f}**. Это результат одной поставляемой таблицы,
а ниже также приведены средние всех трёх обучений.
SHA таблицы: `{provenance['model_sha256']}`. Обычный запуск: `python -m social_market_env.run experiments/zhmurenko_agent.yaml --runs-dir <внешний каталог> --strict`.

## Что улучшалось по требованию Word

В классе Market Agents исследуется адаптация к меняющемуся конкурентному окружению
в повторяющейся игре. Новизной не объявляется само применение Q-learning.
Предложена небольшая специализация Q по падающему/стабильному/растущему рынку,
с обученной общей Q-оценкой для недостаточно изученных режимов. Один вес доверия
используется для всей строки после минимум 8 проб каждого допустимого действия.
Это защищает сравнение действий от различий абсолютного уровня ценности состояния.
Всего максимум 168 общих и 504 режимных строки, по 12 действий.

Контроль flat отключает механизм. Состояния текущего рынка, действия, реальные награды,
seed и бюджеты обучения внутри каждого повтора совпадают. Причинная история и маски
проверены тестами. Подробные признаки и границы описаны в README агента.

## Обучение и ограничения

{json.dumps(training,ensure_ascii=False,indent=2)}

Достижение предела эпизодов не означает сходимости. Признак stability_reached приведён
явно; нестабильные прогоны не удалены. Медиана цен скрывает разброс и отдельных дешёвых
конкурентов; грубая дискретизация и постоянный шаг Q могут ограничивать устойчивость.
После фиксации серии алгоритм не менялся; дополнительные обучения не запускались.
Сохранены промежуточные модели и неудачные прототипы с раздельным доверием к действиям.

В 9 из 10 обучающих эпизодов использован штатный параметр сети w_coop=0, в остальных — 1.
Это ускоряет расчёт, не меняя исходники среды; история/полезность связей остаются включены.
Итоговая оценка использует исходный w_coop=1. Таблицы других агентов обучены нами на
неизменённых исходниках веток: Аскаров и Твердохлеб получили по 125000 обновлений.
Родина использует штатную стратегию, без Q-таблицы. Авторские готовые модели не предоставлялись.
Полноценное взаимное переобучение всех участников не заявляется.

![Обучение](figures/training.png)

## Сокращённая итоговая проверка

12 новых seed 70000–70011, все 12 позиций, четыре топологии, static/evolving по шесть seed.
Семь составов, восемь конкретных моделей/политик: **672 эпизода**. Совпадение начальных
условий проверено по SHA для всех 84 парных условий. V4 и PyTorch исключены.
Для flat/regime показано среднее по трём обученным таблицам, для остальных — одна политика.
Поэтому суммарное число коалиционных шагов нельзя напрямую сравнивать между группами
с разным числом эпизодов. Цены/наценки считаются при фактическом участии.

{chr(10).join(tables)}

{chr(10).join(scenario_table)}

До обучения и простая стратегия оба выбирают cost+tau, поэтому их совпадение ожидаемо.
Реальный вклад обучения оценивается относительно этой политики, а не по наличию файла Q.
Статистика эффекта обучения: regime−экономическая {stats['regime_learning']['mean_difference']:+.2f},
flat−экономическая {stats['flat_learning']['mean_difference']:+.2f}. Полные интервалы — statistics.json.

## Использование Q и коалиций

{json.dumps(coalition,ensure_ascii=False,indent=2)}

known — доля решений в известных общих состояниях; learned — доля выбранных действий
с реальными посещениями; changed — отличие исполненного решения от начальной стратегии.
specialized и regime_changed отдельно показывают применение и влияние режимной таблицы.
high/low — доля выплат выше 1.25 или ниже 0.75 равной доли, а не обещанная прибыль.
Принятие предложения обучается по настоящим выплатам; принудительного вступления нет.
PairOffers — контролируемые допустимые предложения подключённых пар с весами 0.85/0.15,
при обычном расчёте рынка и распределении прибыли среды.

![Эффект механизма](figures/mechanism.png)

Интервал получен 5000 повторениями bootstrap по парным обучающим запускам и seed-кластерам.
Шаги одного эпизода не считаются независимыми. Три обучения и 12 итоговых seed — небольшая
выборка; вывод относится к этим условиям. Это не универсальный рейтинг архитектур.

## Передача и воспроизведение

Корневой YAML: experiments/zhmurenko_agent.yaml. Класс автоматически загружает
models/q_agent.json внутри своего пакета. Отсутствующий/пустой checkpoint вызывает ошибку;
fresh=True — только явно необученный контроль. Модель переносима как обычный JSON.
Полные конфигурации и сжатые пошаговые журналы сохранены рядом с результатами.
Файлы: episodes.csv, by-scenario.csv, training-summary.csv, coalition-and-learning.csv,
statistics.json, deployment-selection.json, final-lock.json.

Ветка {check['branch']}; HEAD {check['commit']}. Чужие/общие изменённые файлы:
{check['outside_scope_changes']}. Проверено {check['protected_files']} исходных файлов.
V4 сохранена: {check['V4_preserved']}. Единственное разрешённое дополнение вне нашего
пакета — собственный корневой YAML. Commit/push не выполнялись.

Тесты: {testcounts['tests']}, ошибок {testcounts['errors']}, падений {testcounts['failures']},
пропусков {testcounts['skipped']}. Штатный раннер успешно выполнил три эпизода с автоматически
загруженной поставляемой таблицей. Повтор выбранного итогового случая воспроизвёл
распакованные записи шагов: {replay['same_decompressed_records']}. PyTorch при повторе
загружен: {replay['PyTorch_loaded']}.
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8');render_html(root)
    (root/'DEFENSE.md').write_text(f'''# Вывод для защиты

Механизм: режимная специализация Q с обученным общим backoff и общим весом доверия строки.
Мотивация: одинаковые последние цены могут соответствовать росту и падению рынка;
редкая история не должна сразу вытеснять надёжно обученную общую оценку.
Проверка: три парных обучения flat/regime и 12 новых seed, одинаковые позиции/cost,
дополнительно необученная таблица и простая стратегия, три реализации коллег среди соперников.
Результат: {category}; regime−flat {effect['mean_difference']:+.2f},
95% CI [{effect['ci95'][0]:+.2f}; {effect['ci95'][1]:+.2f}].
Ограничения: малая итоговая выборка, возможная нестабильность Q и неполная взаимная адаптация.
''',encoding='utf-8')
    print(root/'REPORT.html',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('bundle','report','scope'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=output_path(args.root)
    if args.stage=='bundle':bundle(root)
    elif args.stage=='scope':scope(root)
    else:report(root)


if __name__=='__main__':main()
