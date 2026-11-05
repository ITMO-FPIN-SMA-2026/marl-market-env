"""Mathematical regressions for the measured price-mode and gradient failures."""

import itertools
import math
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from social_market_env.contracts import AgentExpectations, AgentObservation, CoalitionOffer, MarketInfo
from social_market_env.implementations.zhmurenko_vladimir.actions import build_candidates
from social_market_env.implementations.zhmurenko_vladimir.agent import ZhmurenkoVladimirAgent
from social_market_env.implementations.zhmurenko_vladimir.features import FEATURE_NAMES, FEATURE_VERSION
from social_market_env.implementations.zhmurenko_vladimir.learning import update
from social_market_env.implementations.zhmurenko_vladimir.policy import ActorCritic, Choice, LegacyActorCritic, categorical
from social_market_env.implementations.zhmurenko_vladimir.settings import Settings
from social_market_env.implementations.zhmurenko_vladimir.diagnostics import policy_trace


def observed(offer=None):
    return AgentObservation(agent_id=0,step=0,cost=18,balance=1000,active=True,restricted=False,
        last_price=None,last_quantity=0,last_profit=0,market=MarketInfo({},None,None),
        neighbors={1:.2},link_costs={1:0,2:1},offer=offer)


def context(agent,offer=None):
    o=observed(offer)
    return agent.encoder.encode(o).values,build_candidates(o,agent.settings,agent.market)


def enumerate_joint(model,features,candidates):
    """Independent enumeration, not the network's entropy implementation."""
    h,_=model.state(features)
    part=categorical(model.participation(h),(True,candidates.can_sell)).probs
    coal=categorical(model.coalition(h)).probs
    price=categorical(model.price_logits(h,candidates.price_features)).probs
    kind=categorical(model.link_kind(h),(True,*(bool(g) for g in candidates.links[1:]))).probs
    values=[]
    for selling in (0,1):
        if part[selling]==0:
            continue
        joining=(0,1) if selling and candidates.can_join else (None,)
        for join in joining:
            base=part[selling]*(coal[join] if join is not None else 1)
            price_indices=range(len(candidates.prices)) if selling and join!=1 else (None,)
            for price_index in price_indices:
                base_price=base*(price[price_index] if price_index is not None else 1)
                for link_kind in range(4):
                    if kind[link_kind]==0:
                        continue
                    target=categorical(model.target_logits(h,candidates.link_features[link_kind],link_kind)).probs if link_kind else torch.ones(1)
                    for target_index in range(len(target)):
                        values.append((Choice(selling,join,price_index,link_kind,target_index if link_kind else None),
                                       base_price*kind[link_kind]*target[target_index]))
    return values


def test_conditional_probabilities_and_joint_entropy_match_enumeration():
    agent=ZhmurenkoVladimirAgent()
    features,candidates=context(agent,CoalitionOffer(0,(0,1),60,.5))
    candidates=replace(candidates,prices=candidates.prices[:2],price_features=candidates.price_features[:2])
    combinations=enumerate_joint(agent.model,features,candidates)
    probabilities=torch.stack([prob for _,prob in combinations])
    assert float(probabilities.sum().detach())==pytest.approx(1,abs=2e-6)
    entropy=-(probabilities*probabilities.log()).sum()
    assert float(agent.model.joint_entropy(features,candidates).detach())==pytest.approx(float(entropy.detach()),abs=2e-6)
    for choice,prob in combinations[::11]:
        log_probability,_,_=agent.model.evaluate_choice(features,candidates,choice)
        assert float(log_probability.detach())==pytest.approx(float(prob.log().detach()),abs=2e-6)


def test_entropy_branch_gradient_matches_finite_difference():
    agent=ZhmurenkoVladimirAgent()
    features,candidates=context(agent,CoalitionOffer(0,(0,1),60,.5))
    parameter=agent.model.participation.bias
    analytic=torch.autograd.grad(agent.model.joint_entropy(features,candidates),parameter)[0][1]
    epsilon=.001
    with torch.no_grad():
        parameter[1]+=epsilon
        plus=agent.model.joint_entropy(features,candidates).item()
        parameter[1]-=2*epsilon
        minus=agent.model.joint_entropy(features,candidates).item()
        parameter[1]+=epsilon
    assert float(analytic)==pytest.approx((plus-minus)/(2*epsilon),abs=.001)


def test_new_price_logits_have_no_jump_at_cost_and_have_interior_mode():
    agent=ZhmurenkoVladimirAgent()
    features,candidates=context(agent)
    with torch.no_grad():
        agent.model.price.weight.zero_()
        agent.model.price.bias.copy_(torch.tensor([math.log(.25/.75),0.]))
    rows=np.zeros((5,9),dtype=np.float32)
    rows[:,0]=np.array([.17,.18,.25,.30,.40])
    rows[:,-1]=np.array([0,1,1,1,1])
    hidden,_=agent.model.state(features)
    logits=agent.model.price_logits(hidden,rows)
    assert int(logits.argmax())==2
    ablated=rows.copy()
    ablated[:,-1]=1-ablated[:,-1]
    assert torch.equal(logits,agent.model.price_logits(hidden,ablated))
    equal_grid=np.zeros((3,9),dtype=np.float32)
    equal_grid[:,0]=[.20,.25,.30]
    scores=agent.model.price_logits(hidden,equal_grid)
    assert float((scores[0]-2*scores[1]+scores[2]).detach())<0


def test_action_log_probability_has_no_unused_price_gradient():
    agent=ZhmurenkoVladimirAgent()
    features,candidates=context(agent,CoalitionOffer(0,(0,1),60,.5))
    log_probability,_,_=agent.model.evaluate_choice(features,candidates,Choice(1,1,None,0,None))
    gradients=torch.autograd.grad(log_probability,tuple(agent.model.price.parameters()),allow_unused=True)
    assert all(gradient is None for gradient in gradients)


@pytest.mark.parametrize('reward',[100.,-100.])
def test_actor_loss_moves_selected_action_probability_in_correct_direction(reward):
    agent=ZhmurenkoVladimirAgent(settings={'gamma':0.,'entropy_coefficient':0.,'learning_rate':.003})
    features,candidates=context(agent)
    with torch.no_grad():
        agent.model.critic.weight.zero_()
        agent.model.critic.bias.zero_()
    index=min(range(len(candidates.prices)),key=lambda i:abs(candidates.prices[i]-40))
    h,_=agent.model.state(features)
    before=categorical(agent.model.price_logits(h,candidates.price_features)).probs[index].item()
    update(agent.model,agent.optimizer,critic_optimizer=agent.critic_optimizer,settings=agent.settings,
           features=features,candidates=candidates,choice=Choice(1,0 if candidates.can_join else None,index,0,None),
           reward=reward,next_features=None,done=True)
    h,_=agent.model.state(features)
    after=categorical(agent.model.price_logits(h,candidates.price_features)).probs[index].item()
    assert (after-before)*reward>0


def test_critic_and_actor_gradients_are_disjoint():
    agent=ZhmurenkoVladimirAgent()
    features,candidates=context(agent)
    choice=Choice(1,None,0,0,None)
    log_probability,entropy,value=agent.model.evaluate_choice(features,candidates,choice)
    actor_loss=-log_probability*(1-value).detach()-.01*entropy
    actor_to_critic=torch.autograd.grad(actor_loss,agent.model.critic_parameters(),allow_unused=True,retain_graph=True)
    critic_to_actor=torch.autograd.grad((1-value).square(),agent.model.actor_parameters(),allow_unused=True)
    assert all(gradient is None for gradient in actor_to_critic+critic_to_actor)


def test_old_format_is_loaded_exactly_and_not_silently_resumed(tmp_path):
    settings=Settings(architecture_version=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model=LegacyActorCritic(settings)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings.learning_rate)
    original_settings={key:value for key,value in asdict(settings).items() if key not in
                       ('architecture_version','critic_learning_rate','price_scale_min','price_scale_max')}
    path=tmp_path/'v1.pt'
    torch.save({'checkpoint_version':1,'feature_version':FEATURE_VERSION,'feature_names':list(FEATURE_NAMES),
                'settings':original_settings,'market':{'p_min':1.,'p_max':100.,'quantity_scale':120.,'price_offset':5.},
                'model_seed':0,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'updates':123,
                'rng_state':torch.Generator().manual_seed(7).get_state()},path)
    agent=ZhmurenkoVladimirAgent(checkpoint_path=str(path))
    assert agent.settings.architecture_version==1 and agent.updates==123
    assert all(torch.equal(value,agent.model.state_dict()[key]) for key,value in model.state_dict().items())
    agent.reset(agent_id=0,seed=0,training=False)
    assert math.isfinite(agent.act(observed()).price)
    with pytest.raises(ValueError,match='evaluation-only'):
        agent.reset(agent_id=0,seed=0,training=True)


def test_v2_checkpoint_restores_both_optimizer_states(tmp_path):
    first=ZhmurenkoVladimirAgent()
    first.reset(agent_id=0,seed=0,training=True)
    o=observed()
    action=first.act(o)
    first.observe_transition(observation=o,action=action,reward=100,next_observation=None,done=True)
    checkpoint=first.save_checkpoint(tmp_path/'v2.pt')
    second=ZhmurenkoVladimirAgent(checkpoint_path=str(checkpoint))
    assert second.optimizer.state_dict()['state'] and second.critic_optimizer.state_dict()['state']
    for agent in (first,second):
        agent.reset(agent_id=0,seed=5,training=True)
        action=agent.act(o)
        agent.observe_transition(observation=o,action=action,reward=200,next_observation=None,done=True)
    assert first.last_metrics==second.last_metrics
    assert all(torch.equal(value,second.model.state_dict()[name]) for name,value in first.model.state_dict().items())


@pytest.mark.parametrize('gamma',[0.,.95,1.])
def test_critic_remaining_horizon_has_exact_boundary_values(gamma):
    agent=ZhmurenkoVladimirAgent(settings={'gamma':gamma})
    with torch.no_grad():
        agent.model.critic.weight.zero_()
        agent.model.critic.bias.fill_(2.)
    features,_=context(agent)
    start=agent.model.state(features)[1].item()
    last=features.copy()
    last[FEATURE_NAMES.index('step')]=49/50
    end=features.copy()
    end[FEATURE_NAMES.index('step')]=1.
    expected=2*sum(gamma**index for index in range(50))
    assert start==pytest.approx(expected,abs=1e-4)
    assert agent.model.state(last)[1].item()==pytest.approx(2.,abs=1e-4)
    assert agent.model.state(end)[1].item()==0


def test_policy_diagnostics_do_not_consume_rng_or_change_parameters():
    agent=ZhmurenkoVladimirAgent(evaluation_mode='seeded')
    agent.reset(agent_id=0,seed=1,training=False)
    weights={name:value.clone() for name,value in agent.model.state_dict().items()}
    rng=agent.generator.get_state().clone()
    trace=policy_trace(agent,observed(CoalitionOffer(0,(0,1),60,.5)))
    assert sum(trace['price_probabilities'])==pytest.approx(1,abs=1e-6)
    assert trace['learned_price_parameters']['scale']>0
    assert torch.equal(rng,agent.generator.get_state())
    assert all(torch.equal(value,weights[name]) for name,value in agent.model.state_dict().items())
