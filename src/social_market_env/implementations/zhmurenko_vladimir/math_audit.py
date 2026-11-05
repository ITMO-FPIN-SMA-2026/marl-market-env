"""Independent probability checks and gradient attribution on OUR frozen policy."""

import argparse
import json
import math
from pathlib import Path

import torch

from social_market_env.contracts import AgentEvaluation, AgentExpectations, AgentObservation, CoalitionOffer, MarketInfo

from .actions import LINK_KINDS, build_candidates
from .agent import ZhmurenkoVladimirAgent
from .policy import Choice
from .settings import writable_path


def observation_from_dict(data):
    data=dict(data)
    market=dict(data['market'])
    market['last_prices']={int(key):value for key,value in market['last_prices'].items()}
    data['market']=MarketInfo(**market)
    data['neighbors']={int(key):value for key,value in data['neighbors'].items()}
    data['link_costs']={int(key):value for key,value in data['link_costs'].items()}
    data['expectations']=AgentExpectations(**data['expectations'])
    if data['last_evaluation']:
        data['last_evaluation']=AgentEvaluation(**data['last_evaluation'])
    if data['offer']:
        data['offer']=CoalitionOffer(**data['offer'])
    return AgentObservation(**data)


def reconstruct_choice(trace, candidates):
    action=trace['action']
    participation=int(action['participate'])
    coalition=int(action['coalition_choice'] is not None) if participation and candidates.can_join else None
    price=next((i for i,p in enumerate(candidates.prices) if abs(p-action['price'])<1e-7),None) if participation and coalition!=1 else None
    kind=LINK_KINDS.index(action['link_action']['kind']) if action['link_action'] else 0
    target=next(i for i,a in enumerate(candidates.links[kind]) if a.target==action['link_action']['target']) if kind else None
    return Choice(participation,coalition,price,kind,target)


def independent_log_probability(trace, choice):
    components={'participation':math.log(trace['participation_probabilities'][choice.participation])}
    if choice.coalition is not None:
        components['coalition']=math.log(trace['coalition_probabilities'][choice.coalition])
    if choice.price is not None:
        components['price']=math.log(trace['price_probabilities'][choice.price])
    components['link_kind']=math.log(trace['link_kind_probabilities'][choice.link_kind])
    if choice.link_target is not None:
        head=next(item for item in trace['link_target_heads'] if item['kind']==str(LINK_KINDS[choice.link_kind]))
        components['link_target']=math.log(head['probabilities'][choice.link_target])
    return components


def joint_entropy_from_trace(trace):
    entropy=lambda values:-sum(p*math.log(p) for p in values if p>0)
    sell=trace['participation_probabilities'][1]
    join=trace['coalition_probabilities'][1] if trace['coalition_available'] else 0
    result=entropy(trace['participation_probabilities'])+sell*(
        (entropy(trace['coalition_probabilities']) if trace['coalition_available'] else 0)+(1-join)*trace['price_entropy'])
    result+=entropy(trace['link_kind_probabilities'])
    result+=sum(trace['link_kind_probabilities'][LINK_KINDS.index(head['kind'])]*head['entropy'] for head in trace['link_target_heads'])
    return result


def path_entropy_from_trace(trace,choice):
    entropy=lambda values:-sum(p*math.log(p) for p in values if p>0)
    result=entropy(trace['participation_probabilities'])+entropy(trace['link_kind_probabilities'])
    if choice.coalition is not None:
        result+=entropy(trace['coalition_probabilities'])
    if choice.price is not None:
        result+=trace['price_entropy']
    if choice.link_target is not None:
        result+=next(head['entropy'] for head in trace['link_target_heads'] if head['kind']==str(LINK_KINDS[choice.link_kind]))
    return result


def audit(*, checkpoint: Path, trace_path: Path, output: Path):
    traces=[json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
    agent=ZhmurenkoVladimirAgent(checkpoint_path=str(checkpoint))
    weights={key:value.clone() for key,value in agent.model.state_dict().items()}
    rows=[]
    for index in (0,1,len(traces)-1):
        trace=traces[index]
        o=observation_from_dict(trace['observation'])
        features=agent.encoder.encode(o).values
        candidates=build_candidates(o,agent.settings,agent.market)
        choice=reconstruct_choice(trace,candidates)
        log_probability,path_entropy,value=agent.model.evaluate_choice(features,candidates,choice)
        terminal=index==len(traces)-1
        with torch.no_grad():
            bootstrap=torch.zeros(()) if terminal else agent.model.state(agent.encoder.encode(observation_from_dict(traces[index+1]['observation'])).values)[1]
            target=trace['outcome']['reward']/agent.settings.reward_scale+agent.settings.gamma*bootstrap
        delta=target-value
        actor=-log_probability*delta.detach().clamp(-agent.settings.advantage_clip,agent.settings.advantage_clip)
        critic=agent.settings.value_coefficient*(delta.square() if agent.settings.architecture_version==1
                                                else torch.nn.functional.smooth_l1_loss(value,target))
        regularizer=-agent.settings.entropy_coefficient*path_entropy
        parameters=list(agent.model.named_parameters())
        vectors={}
        for name,loss in (('actor',actor),('critic',critic),('entropy',regularizer)):
            gradients=torch.autograd.grad(loss,[parameter for _,parameter in parameters],allow_unused=True,retain_graph=True)
            vectors[name]=torch.cat([torch.zeros_like(parameter).flatten() if gradient is None else gradient.flatten()
                                     for (_,parameter),gradient in zip(parameters,gradients)])
        total=sum(vectors.values())
        norms={key:float(vector.norm()) for key,vector in vectors.items()}
        components=independent_log_probability(trace,choice)
        rows.append({'step':trace['step'],'done':terminal,'reward':trace['outcome']['reward'],
                     'value':float(value.detach()),'target':float(target),'td_error':float(delta.detach()),
                     'log_probability':float(log_probability.detach()),'independent_log_probability':sum(components.values()),
                     'log_probability_components':components,'path_entropy':path_entropy_from_trace(trace,choice),
                     'reported_entropy':float(path_entropy.detach()),
                     'joint_entropy':joint_entropy_from_trace(trace),'gradient_norms':norms,
                     'total_gradient_norm':float(total.norm()),'global_clip_multiplier':min(1,agent.settings.max_gradient_norm/(float(total.norm())+1e-6)),
                     'actor_critic_cosine':float(torch.dot(vectors['actor'],vectors['critic'])/(vectors['actor'].norm()*vectors['critic'].norm()+1e-12))})
        rows[-1]['critic_objective']='MSE' if agent.settings.architecture_version==1 else 'Huber'
        rows[-1]['clip_scheme']='global' if agent.settings.architecture_version==1 else 'separate'
        if agent.settings.architecture_version>=2:
            rows[-1]['counterfactual_global_clip_multiplier']=rows[-1].pop('global_clip_multiplier')
            rows[-1]['actor_clip_multiplier']=min(1,agent.settings.max_gradient_norm/(float((vectors['actor']+vectors['entropy']).norm())+1e-6))
            rows[-1]['critic_clip_multiplier']=min(1,agent.settings.max_gradient_norm/(norms['critic']+1e-6))
        assert abs(rows[-1]['log_probability']-rows[-1]['independent_log_probability'])<1e-5
    assert all(torch.equal(value,weights[key]) for key,value in agent.model.state_dict().items())
    report={'samples':rows,'frozen_weights_verified':True}
    writable_path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    audit(checkpoint=args.checkpoint,trace_path=args.trace,output=args.output)


if __name__=='__main__':
    main()
