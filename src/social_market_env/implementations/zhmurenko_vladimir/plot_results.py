"""Standalone scientific figures and CSV tables from saved experiment artifacts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .settings import writable_path


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def rolling(values,window):
    return [float(np.mean(values[max(0,index-window+1):index+1])) for index in range(len(values))]


def training_series(directory):
    episodes=read_lines(directory/'episodes.jsonl')
    learning=read_lines(directory/'learning.jsonl')
    cursor=0
    for episode in episodes:
        block=learning[cursor:cursor+episode['updates']]
        cursor+=episode['updates']
        for field in ('critic_loss','td_error','entropy','gradient_norm','actor_gradient_norm','critic_gradient_norm'):
            episode[field]=float(np.mean([row[field] for row in block])) if block and field in block[0] else None
        records=read_lines(directory/f'seed_{episode["seed"]}'/f'episode_{episode["episode"]}'/'steps.jsonl')
        if episode.get('mean_margin') is None:
            margins=[]
            for record in records:
                evaluation=record['agent_evaluations'].get('0')
                if evaluation and evaluation['price'] is not None and evaluation['quantity']>1e-12:
                    margins.append(evaluation['gross_profit']/evaluation['quantity'])
            episode['mean_margin']=float(np.mean(margins)) if margins else None
        episode['clip_fraction']=float(np.mean([row.get('actor_gradient_norm',row['gradient_norm'])>1 for row in block])) if block else 0
    assert cursor==len(learning)
    return episodes,learning


def create_plots(*, root: Path, old_training: Path):
    output=writable_path(root/'figures')
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.2,'savefig.facecolor':'white'})
    diagnosis=json.loads((root/'diagnosis-before/report.json').read_text(encoding='utf-8'))
    trained=diagnosis['strategies']['trained']['examples'][0]
    untrained=diagnosis['strategies']['untrained']['examples'][0]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for example,label,color in ((trained,'Обученная V1','#be3d31'),(untrained,'Исходная V1','#4477aa')):
        axes[0].plot(example['prices'],example['price_probabilities'],'o-',label=label,color=color,markersize=3)
    axes[0].axvline(trained['cost'],color='black',linestyle='--',label=f'Себестоимость {trained["cost"]:.2f}')
    axes[0].set(xlabel='Цена',ylabel='Вероятность',title='Реальное состояние: seed 100, эпизод 0, шаг 0')
    logits=np.asarray(trained['price_logits'])
    normalized=logits-logits.max()-np.log(np.exp(logits-logits.max()).sum())
    axes[1].plot(trained['prices'],normalized,'o-',label='Исходные признаки',color='#be3d31',markersize=3)
    axes[1].plot(trained['prices'],trained['indicator_ablation']['logits'],'o-',label='Обнулён индикатор price ≥ cost',color='#4477aa',markersize=3)
    axes[1].axvline(trained['cost'],color='black',linestyle='--')
    axes[1].set(xlabel='Цена',ylabel='Нормированный logit',title='Вмешательство только в признак нашей сети')
    for axis in axes:
        axis.legend(fontsize=8)
    fig.savefig(output/'price_diagnosis.png',dpi=160)
    fig.savefig(output/'price_diagnosis.pdf')
    plt.close(fig)

    old_episodes,old_learning=training_series(old_training)
    new_episodes,new_learning=training_series(root/'training-v3')
    columns=('seed','episode','total_reward','critic_loss','td_error','entropy','gradient_norm','actor_gradient_norm',
             'critic_gradient_norm','participation_rate','mean_price','mean_margin','join_rate','fine_count','eliminated','clip_fraction')
    with (output/'training_timeseries.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=('version','episode_number',*columns),extrasaction='ignore')
        writer.writeheader()
        for label,episodes in (('V1',old_episodes),('V3',new_episodes)):
            for index,episode in enumerate(episodes):
                writer.writerow(dict(episode,version=label,episode_number=index+1))
    fig,axes=plt.subplots(4,3,figsize=(17,15),layout='constrained')
    old_color,new_color='#999999','#176b9b'
    for label,episodes,learning,color in (('V1',old_episodes,old_learning,old_color),('V3',new_episodes,new_learning,new_color)):
        x=np.arange(1,len(episodes)+1)
        rewards=[e['total_reward'] for e in episodes]
        axes[0,0].plot(x,rewards,color=color,alpha=.25)
        axes[0,0].plot(x,rolling(rewards,10),color=color,label=label)
        for axis,key,title in ((axes[0,1],'critic_loss','MSE критика, средняя по эпизоду'),
                              (axes[0,2],'td_error','TD-ошибка, средняя по эпизоду'),
                              (axes[1,0],'entropy','V1: энтропия пути; V3: полная энтропия'),
                              (axes[1,2],'participation_rate','Доля участия'),
                              (axes[2,0],'mean_price','Средняя цена'),(axes[2,1],'mean_margin','Средняя цена минус себестоимость'),
                              (axes[2,2],'join_rate','Доля принятия предложений'),(axes[3,0],'fine_count','Количество штрафов'),
                              (axes[3,2],'clip_fraction','Доля clipping: V1 общий / V3 actor')):
            axis.plot(x,[e.get(key) for e in episodes],color=color,alpha=.8,label=label)
            axis.set_title(title)
        axes[3,1].plot(x,np.cumsum([e['eliminated'] for e in episodes]),color=color,label=label)
    axes[0,0].set_title('Награда эпизода и скользящая средняя (10)')
    axes[3,1].set_title('Накопленное число выбываний')
    for label,episodes,key,color in (('V1 общий',old_episodes,'gradient_norm',old_color),
                                     ('V3 actor',new_episodes,'actor_gradient_norm','#176b9b'),
                                     ('V3 critic',new_episodes,'critic_gradient_norm','#c88514')):
        axes[1,1].plot(np.arange(1,len(episodes)+1),[e[key] for e in episodes],label=label,color=color)
    axes[1,1].set_title('Нормы градиентов до clipping')
    axes[0,1].set_yscale('symlog',linthresh=.1)
    axes[1,1].set_yscale('symlog',linthresh=.1)
    for axis in axes.flat:
        axis.set_xlabel('Эпизод')
        axis.legend(fontsize=8)
    fig.savefig(output/'training_dashboard.png',dpi=150)
    fig.savefig(output/'training_dashboard.pdf')
    plt.close(fig)

    labels={'old_trained':'V1\nargmax','old_trained_sampled':'V1\nвыборка','trained':'V3\nargmax','trained_sampled':'V3\nвыборка',
            'untrained':'V3 без\nобучения','untrained_sampled':'V3 без обучения\nвыборка','CartelAgent':'Cartel','BestResponseAgent':'BestResponse','RandomAgent':'Random'}
    reports={name:json.loads((root/name/'report.json').read_text(encoding='utf-8')) for name in ('control-v2','final-v2')}
    fig,axes=plt.subplots(2,1,figsize=(14,10),layout='constrained')
    for axis,(name,report) in zip(axes,reports.items()):
        strategies=report['strategies']
        keys=list(strategies)
        colors=['#176b9b' if key=='trained' else '#3b9b86' if key=='trained_sampled' else '#999999' for key in keys]
        axis.bar(np.arange(len(keys)),[strategies[k]['mean_reward'] for k in keys],
                 yerr=[strategies[k]['reward_std'] for k in keys],color=colors,capsize=3)
        axis.set_xticks(np.arange(len(keys)),[labels[key] for key in keys])
        axis.axhline(0,color='black',linewidth=.7)
        axis.set(ylabel='Индивидуальная награда за эпизод',title=('Контроль: seeds 100–104' if name=='control-v2' else 'Независимая проверка: seeds 200–209')+'; ошибки — стандартное отклонение эпизодов')
        with (output/(name+'-comparison.csv')).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(('strategy','episodes','mean_reward','median_reward','reward_std','seed_mean_std','fines','eliminations'))
            for key,value in strategies.items():
                writer.writerow((key,*(value[field] for field in ('episodes','mean_reward','median_reward','reward_std','seed_mean_std','fine_count','eliminations'))))
    fig.savefig(output/'evaluation_comparison.png',dpi=160)
    fig.savefig(output/'evaluation_comparison.pdf')
    plt.close(fig)
    final=reports['final-v2']
    fig,axis=plt.subplots(figsize=(11,4.5),layout='constrained')
    for key,color in (('BestResponseAgent','#c88514'),('RandomAgent','#176b9b'),('old_trained_sampled','#888888')):
        differences=final['paired_seed_reward_differences'][key]
        axis.plot(list(map(int,differences)),list(differences.values()),'o-',label='V3 argmax − '+labels[key].replace('\n',' '),color=color)
    axis.axhline(0,color='black',linewidth=.8)
    axis.set(xlabel='Seed',ylabel='Разность средних наград',title='Парное сравнение на зафиксированном независимом наборе')
    axis.legend()
    fig.savefig(output/'paired_differences.png',dpi=160)
    plt.close(fig)
    print(output)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--old-training',type=Path,required=True)
    args=parser.parse_args(argv)
    create_plots(root=args.root,old_training=args.old_training)


if __name__=='__main__':
    main()
