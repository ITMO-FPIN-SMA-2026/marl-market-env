"""Read-only policy traces on real public observations and recorded outcomes."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from social_market_env.config import config_from_dict, config_to_dict, load_config
from social_market_env.runner.experiment import build_environment

from .actions import LINK_KINDS, PRICE_FEATURES, build_candidates
from .agent import ZhmurenkoVladimirAgent
from .checkpoint import read_checkpoint
from .metrics import agent_value, episode_metrics, play_episode, summarize
from .policy import Choice, categorical
from .settings import PACKAGE_ROOT, writable_path


def policy_trace(agent, observation):
    """Inspect our own network; neither sample actions nor modify a parameter/RNG."""
    o = observation
    encoded = agent.encoder.encode(o)
    candidates = build_candidates(o, agent.settings, agent.market)
    with torch.no_grad():
        hidden, value = agent.model.state(encoded.values)
        price_logits = agent.model.price_logits(hidden, candidates.price_features)
        price_distribution = categorical(price_logits)
        part = categorical(agent.model.participation(hidden), (True, candidates.can_sell))
        coalition = categorical(agent.model.coalition(hidden))
        kind = categorical(agent.model.link_kind(hidden), (True, *(bool(g) for g in candidates.links[1:])))
        target_heads = []
        for index, group in enumerate(candidates.links):
            if group:
                distribution = categorical(agent.model.target_logits(hidden, candidates.link_features[index], index))
                target_heads.append({'kind': str(LINK_KINDS[index]), 'targets': [a.target for a in group],
                                     'probabilities': distribution.probs.tolist(), 'entropy': float(distribution.entropy())})
        price_index = int(price_distribution.probs.argmax())
        learned_price_parameters = None
        if hasattr(agent.model,'price_parameters'):
            center,scale=agent.model.price_parameters(hidden)
            learned_price_parameters={'center':float(center)*agent.market.p_max,'scale':float(scale)*agent.market.p_max}
        # An intervention on OUR feature only: does the discontinuous indicator cause the mode?
        indicator_ablation = None
        if 'at_or_above_cost' in PRICE_FEATURES:
            ablated = candidates.price_features.copy()
            ablated[:, PRICE_FEATURES.index('at_or_above_cost')] = 0
            alternative = categorical(agent.model.price_logits(hidden, ablated))
            indicator_ablation = {'argmax_price': candidates.prices[int(alternative.probs.argmax())],
                                   'logits': alternative.logits.tolist()}
    return {'observation': asdict(o), 'cost': o.cost, 'step': o.step, 'features': encoded.values.tolist(),
            'prices': list(candidates.prices), 'candidate_features': candidates.price_features.tolist(),
            'price_feature_names': list(PRICE_FEATURES), 'price_logits': price_logits.tolist(),
            'price_probabilities': price_distribution.probs.tolist(), 'price_argmax_index': price_index,
            'price_argmax': candidates.prices[price_index], 'price_entropy': float(price_distribution.entropy()),
            'learned_price_parameters':learned_price_parameters,
            'critic_value': float(value), 'participation_probabilities': part.probs.tolist(),
            'coalition_available': candidates.can_join, 'coalition_probabilities': coalition.probs.tolist(),
            'link_kind_probabilities': kind.probs.tolist(), 'link_target_heads': target_heads,
            'indicator_ablation': indicator_ablation}


class DiagnosticAgent(ZhmurenkoVladimirAgent):
    def __init__(self, *, market=None, **kwargs):
        super().__init__(market=market, **kwargs)
        self.traces = []

    def reset(self, *, agent_id, seed, training):
        if training:
            raise ValueError('Diagnostic traces are read-only; training must be False')
        super().reset(agent_id=agent_id, seed=seed, training=False)
        self.traces.clear()

    def act(self, observation):
        trace = policy_trace(self, observation)
        action = super().act(observation)
        trace['action'] = asdict(action)
        trace['margin'] = action.price-observation.cost
        trace['selected_price_probability'] = next((prob for price,prob in zip(trace['prices'],trace['price_probabilities'])
                                                  if abs(price-action.price)<1e-7), None)
        self.traces.append(trace)
        return action


def numeric_summary(values):
    array = np.asarray(values, dtype=float)
    return {'count': len(array), 'mean': float(array.mean()), 'std': float(array.std()),
            'quantiles': dict(zip(('min','p50','p90','p95','p99','max'),
                                 map(float, np.quantile(array, (0,.5,.9,.95,.99,1)))))} if array.size else {}


def checkpoint_evolution(*, checkpoint_paths, observation_data, output):
    """Compare saved stages on ONE real public observation, not different trajectories."""
    from .math_audit import observation_from_dict
    observation=observation_from_dict(observation_data)
    rows=[]
    for family,paths in checkpoint_paths.items():
        last=ZhmurenkoVladimirAgent(checkpoint_path=str(paths[-1]))
        initial=ZhmurenkoVladimirAgent(settings=asdict(last.settings),model_seed=last.model_seed)
        agents=[('initial',None,initial)]
        agents.extend((path.name,path,ZhmurenkoVladimirAgent(checkpoint_path=str(path))) for path in paths)
        for name,path,agent in agents:
            before=hashlib.sha256(path.read_bytes()).hexdigest() if path else None
            trace=policy_trace(agent,observation)
            cost_probability=next((p for price,p in zip(trace['prices'],trace['price_probabilities']) if abs(price-observation.cost)<1e-7),0)
            rows.append({'family':family,'checkpoint':str(path) if path else None,'name':name,'updates':agent.updates,
                         'argmax_price':trace['price_argmax'],'cost_probability':cost_probability,
                         'price_entropy':trace['price_entropy'],'participation_probability':trace['participation_probabilities'][1],
                         'logits':trace['price_logits'],'probabilities':trace['price_probabilities'],'prices':trace['prices']})
            if path:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    writable_path(output).write_text(json.dumps({'observation':observation_data,'stages':rows},indent=2),encoding='utf-8')
    return rows


def training_history(directory: Path):
    updates = [json.loads(line) for line in (directory/'learning.jsonl').read_text(encoding='utf-8').splitlines()]
    episodes = [json.loads(line) for line in (directory/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    result = {key:numeric_summary([row[key] for row in updates]) for key in updates[0]}
    result['gradient_clipped_fraction'] = sum(row['gradient_norm']>1 for row in updates)/len(updates)
    result['advantage_clipped_fraction'] = sum(abs(row['td_error'])>20 for row in updates)/len(updates)
    if 'actor_gradient_norm' in updates[0]:
        result['combined_norm_above_1_fraction']=result.pop('gradient_clipped_fraction')
        result['actor_clipped_fraction']=sum(row['actor_gradient_norm']>1 for row in updates)/len(updates)
        result['critic_clipped_fraction']=sum(row['critic_gradient_norm']>1 for row in updates)/len(updates)
    result['blocks'] = [{key:numeric_summary([row[key] for row in block]) for key in ('critic_loss','td_error','entropy','gradient_norm')}
                        for block in np.array_split(np.asarray(updates,dtype=object), 6)]
    result['episodes'] = episodes
    return result


def diagnose(*, checkpoint: Path, output_dir: Path, config_path: Path, history_dir: Path | None = None):
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    raw = config_to_dict(load_config(config_path))
    raw.update(mode='eval', strict=True)
    output = writable_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'experiment.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
    report = {'checkpoint':str(checkpoint),'checkpoint_sha256':before_hash,'strategies':{}}
    checkpoint_settings=dict(read_checkpoint(checkpoint)['settings'])
    checkpoint_settings.setdefault('architecture_version',1)
    with (output/'candidates.csv').open('w',encoding='utf-8',newline='') as candidate_stream:
        writer=csv.writer(candidate_stream)
        writer.writerow(('strategy','seed','episode','step','cost','candidate_index','price','margin','logit','probability','argmax'))
        for strategy in ('trained','trained_sampled','untrained'):
            kwargs={'evaluation_mode':'seeded' if strategy.endswith('_sampled') else 'greedy','settings':checkpoint_settings}
            if strategy.startswith('trained'):
                kwargs['checkpoint_path']=str(checkpoint.resolve())
            config_data=json.loads(json.dumps(raw))
            config_data['agents']['overrides']['0']={'impl':__name__+':DiagnosticAgent','kwargs':kwargs}
            # __main__ cannot be imported by the factory; always use the canonical module.
            config_data['agents']['overrides']['0']['impl']='social_market_env.implementations.zhmurenko_vladimir.diagnostics:DiagnosticAgent'
            episodes, rows = [], []
            for seed in raw['seeds']:
                env=build_environment(config=config_from_dict(config_data),seed=seed)
                own=env.agents[0]
                weights={name:value.clone() for name,value in own.model.state_dict().items()}
                for episode in range(raw['episodes_per_seed']):
                    directory=output/strategy/f'seed_{seed}'/f'episode_{episode}'
                    records,_,reset_seed=play_episode(env,seed=seed,episode=episode,training=False,directory=directory)
                    with (directory/'policy.jsonl').open('w',encoding='utf-8') as stream:
                        for trace,record in zip(own.traces,records):
                            result=record['agent_evaluations']['0'] if '0' in record['agent_evaluations'] else record['agent_evaluations'][0]
                            trace['outcome']=result
                            trace['balance']=agent_value(record['balances'],0)
                            trace['active_after']=agent_value(record['active'],0)
                            trace['restricted_after']=agent_value(record['restricted'],0)
                            stream.write(json.dumps(trace)+'\n')
                            for index,price in enumerate(trace['prices']):
                                writer.writerow((strategy,seed,episode,trace['step'],trace['cost'],index,price,price-trace['cost'],
                                                 trace['price_logits'][index],trace['price_probabilities'][index],index==trace['price_argmax_index']))
                            rows.append(trace)
                    episodes.append(dict(episode_metrics(records,agent_id=0),seed=seed,episode=episode,reset_seed=reset_seed))
                assert all(torch.equal(value,weights[name]) for name,value in own.model.state_dict().items())
                print(f'diagnose {strategy} seed={seed}',flush=True)
            above = [trace for trace in rows if abs(trace['action']['price']-trace['cost'])<1e-7]
            monotonically_decreasing = []
            for trace in rows:
                logits=[logit for price,logit in zip(trace['prices'],trace['price_logits']) if price>=trace['cost']-1e-8]
                monotonically_decreasing.append(all(a>=b-1e-6 for a,b in zip(logits,logits[1:])))
            report['strategies'][strategy]={'summary':summarize(episodes),'steps':len(rows),'at_cost_steps':len(above),
                'margins':numeric_summary([trace['margin'] for trace in rows]),
                'positive_sales_at_cost':sum(trace['outcome']['quantity']>0 for trace in above),
                'decreasing_logits_above_cost_fraction':sum(monotonically_decreasing)/len(rows),
                'indicator_removed_selects_min_fraction':sum(abs(trace['indicator_ablation']['argmax_price']-own.market.p_min)<1e-7 for trace in rows)/len(rows),
                'candidate_counts':numeric_summary([len(trace['prices']) for trace in rows]),
                'examples':[rows[index] for index in (0,min(1,len(rows)-1),min(49,len(rows)-1))]}
    if history_dir:
        report['training_history']=training_history(history_dir)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before_hash
    report['frozen_weights_verified']=True
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:{key:value for key,value in item.items() if key!='examples'} for name,item in report['strategies'].items()},indent=2),flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PACKAGE_ROOT/'experiments/eval.yaml')
    parser.add_argument('--history-dir',type=Path)
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    diagnose(checkpoint=args.checkpoint,output_dir=args.output_dir,config_path=args.config,history_dir=args.history_dir)


if __name__=='__main__':
    main()
