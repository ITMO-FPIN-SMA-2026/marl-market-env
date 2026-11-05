import copy
import importlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from social_market_env.config import config_from_dict, load_config
from social_market_env.contracts import (AgentAction, AgentExpectations, AgentObservation, CoalitionOffer,
                                         LinkActionKind, MarketInfo)
from social_market_env.runner.experiment import build_environment
from social_market_env.implementations.zhmurenko_vladimir import ZhmurenkoVladimirAgent
from social_market_env.implementations.zhmurenko_vladimir.actions import LINK_FEATURES, build_candidates
from social_market_env.implementations.zhmurenko_vladimir.checkpoint import read_checkpoint
from social_market_env.implementations.zhmurenko_vladimir.features import FEATURE_NAMES, FeatureEncoder
from social_market_env.implementations.zhmurenko_vladimir.metrics import play_episode
from social_market_env.implementations.zhmurenko_vladimir.policy import Choice, categorical
from social_market_env.implementations.zhmurenko_vladimir.settings import MarketSettings, PACKAGE_ROOT, Settings, writable_path
from social_market_env.implementations.zhmurenko_vladimir.train import AGENT_IMPL, train


@pytest.fixture(scope='session', autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(**changes):
    base = AgentObservation(agent_id=0, step=0, cost=18.0, balance=1000.0, active=True, restricted=False,
                            last_price=None, last_quantity=0.0, last_profit=0.0,
                            market=MarketInfo({}, None, None), neighbors={}, link_costs={}, offer=None)
    return replace(base, **changes)


def weights(agent):
    return {name: value.clone() for name, value in agent.model.state_dict().items()}


def same_weights(first, second):
    return first.keys() == second.keys() and all(torch.equal(value, second[name]) for name, value in first.items())


def force_head(agent, head, chosen):
    with torch.no_grad():
        layer = getattr(agent.model, head)
        layer.weight.zero_()
        layer.bias.fill_(-100)
        layer.bias[chosen] = 100


def small_config(horizon=4, **changes):
    raw = {'horizon': horizon, 'strict':True, 'agents': {'count':3, 'overrides':{0:{'impl':AGENT_IMPL}}},
           'network':{'topology':'path','mode':'static','w_coop':0}}
    raw.update(changes)
    return config_from_dict(raw)


def test_real_yaml_loader_and_namespace_import():
    assert not (PACKAGE_ROOT.parent/'__init__.py').exists()
    assert importlib.import_module(AGENT_IMPL.split(':')[0]).ZhmurenkoVladimirAgent is ZhmurenkoVladimirAgent
    config = load_config(PACKAGE_ROOT/'experiments/train.yaml')
    env = build_environment(config=config, seed=0)
    assert isinstance(env.agents[0], ZhmurenkoVladimirAgent)
    assert env.agents[0].market.p_max == config.market.p_max
    assert env.agents[0].settings.horizon == config.horizon


@pytest.mark.parametrize('count', [0, 1, 2, 12, 25])
def test_features_are_fixed_finite_and_do_not_mutate_input(count):
    o = observation(step=2, market=MarketInfo({i:20+i for i in range(count)},30.0,90.0),
                    neighbors={i:0.2 for i in range(1,count)}, link_costs={i:0.5 for i in range(1,count)})
    original = copy.deepcopy(o)
    encoded = FeatureEncoder(Settings(), MarketSettings()).encode(o)
    assert encoded.names == FEATURE_NAMES and encoded.values.shape == (85,)
    assert np.isfinite(encoded.values).all()
    assert o == original


def test_missing_is_distinct_from_an_observed_zero():
    encoder = FeatureEncoder(Settings(), MarketSettings())
    missing = encoder.encode(observation()).values
    zero = encoder.encode(observation(last_price=0.0)).values
    index = FEATURE_NAMES.index('last_price_present')
    assert missing[index] == 0 and zero[index] == 1
    o = observation(balance=float('inf'), cost=float('nan'), market=MarketInfo({1:float('inf')},float('nan'),None))
    assert np.isfinite(encoder.encode(o).values).all()


@pytest.mark.parametrize(('restricted','cost','active'), [(False,18,True),(True,18,True),(True,150,True),(False,18,False)])
def test_legal_first_step_action(restricted, cost, active):
    agent = ZhmurenkoVladimirAgent()
    agent.reset(agent_id=0,seed=0,training=False)
    action = agent.act(observation(restricted=restricted,cost=cost,active=active))
    assert type(action) is AgentAction and type(action.price) is float and math.isfinite(action.price)
    assert 1 <= action.price <= 100 and type(action.participate) is bool
    assert action.coalition_choice is None and action.link_action is None
    if not active or (restricted and cost>100):
        assert not action.participate
    elif restricted:
        assert action.price>=cost


def test_offer_is_followed_exactly_and_restricted_offer_is_masked():
    agent = ZhmurenkoVladimirAgent()
    force_head(agent,'participation',1)
    force_head(agent,'coalition',1)
    o = observation(offer=CoalitionOffer(7,(0,1),61.23456789,0.5))
    agent.reset(agent_id=0,seed=1,training=False)
    action=agent.act(o)
    assert action.participate and action.coalition_choice == 7 and action.price == o.offer.recommended_price
    assert agent.act(replace(o,restricted=True)).coalition_choice is None
    assert agent.act(replace(o,offer=replace(o.offer,recommended_price=101))).coalition_choice is None
    force_head(agent,'participation',0)
    assert agent.act(o).coalition_choice is None


def test_link_candidates_and_hierarchical_none_probability():
    agent=ZhmurenkoVladimirAgent()
    o=observation(neighbors={1:0.3},link_costs={1:0,2:1})
    candidates=build_candidates(o,agent.settings,agent.market)
    assert [a.target for a in candidates.links[1]] == [2]
    assert candidates.links[2][0].kind == LinkActionKind.maintain
    assert candidates.links[3][0].kind == LinkActionKind.remove
    for kind in (1,2,3):
        force_head(agent,'link_kind',kind)
        agent.reset(agent_id=0,seed=3,training=False)
        action=agent.act(o)
        assert action.link_action == candidates.links[kind][0]
    logits=torch.zeros(4)
    first=categorical(logits,(True,True,True,True)).probs[0]
    # Adding 100 target choices does not enter the kind distribution at all.
    many=build_candidates(replace(o,link_costs={i:1.0 for i in range(1,102)}),agent.settings,agent.market)
    assert len(many.links[1])==100
    assert categorical(logits,(True,*(bool(g) for g in many.links[1:]))).probs[0] == first


def test_training_changes_parameters_and_reset_preserves_them():
    agent=ZhmurenkoVladimirAgent()
    agent.reset(agent_id=0,seed=2,training=True)
    before=weights(agent)
    for step in range(3):
        o=observation(step=step)
        action=agent.act(o)
        agent.observe_transition(observation=o,action=action,reward=100.0,
                                 next_observation=observation(step=step+1) if step<2 else None,done=step==2)
    after=weights(agent)
    assert agent.updates==3 and not same_weights(before,after)
    assert all(math.isfinite(value) for value in agent.last_metrics.values())
    assert agent.last_metrics['target']==pytest.approx(1.0)
    agent.reset(agent_id=0,seed=9,training=True)
    assert same_weights(after,weights(agent)) and agent._pending is None


def test_td_bootstraps_without_updating_through_next_state():
    agent=ZhmurenkoVladimirAgent()
    agent.reset(agent_id=0,seed=0,training=True)
    o,next_o=observation(),observation(step=1,balance=1050.0)
    action=agent.act(o)
    with torch.no_grad():
        expected=0.5+agent.settings.gamma*float(agent.model.state(agent.encoder.encode(next_o).values)[1])
    agent.observe_transition(observation=o,action=action,reward=50,next_observation=next_o,done=False)
    assert agent.last_metrics['target']==pytest.approx(expected,abs=1e-6)
    assert agent._pending is None


def test_unused_price_and_coalition_heads_do_not_learn_when_not_selling():
    agent=ZhmurenkoVladimirAgent()
    force_head(agent,'participation',0)
    agent.reset(agent_id=0,seed=0,training=True)
    o=observation(offer=CoalitionOffer(0,(0,1),60,0.5))
    before=weights(agent)
    action=agent.act(o)
    assert not action.participate
    agent.observe_transition(observation=o,action=action,reward=50,next_observation=None,done=True)
    after=weights(agent)
    for name in before:
        if name.startswith(('price.','coalition.')):
            assert torch.equal(before[name],after[name])


def test_price_head_does_not_learn_when_joining():
    agent=ZhmurenkoVladimirAgent()
    force_head(agent,'participation',1)
    force_head(agent,'coalition',1)
    agent.reset(agent_id=0,seed=0,training=True)
    o=observation(offer=CoalitionOffer(0,(0,1),60,0.5))
    before=weights(agent)
    action=agent.act(o)
    agent.observe_transition(observation=o,action=action,reward=50,next_observation=None,done=True)
    for name,value in agent.model.state_dict().items():
        if name.startswith('price.'):
            assert torch.equal(value,before[name])


def test_bad_transition_and_missing_feedback_are_detected():
    agent=ZhmurenkoVladimirAgent()
    agent.reset(agent_id=0,seed=0,training=True)
    o=observation()
    action=agent.act(o)
    with pytest.raises(RuntimeError,match='waiting'):
        agent.act(o)
    with pytest.raises(ValueError,match='differs'):
        agent.observe_transition(observation=o,action=replace(action,price=action.price+1),reward=1,next_observation=None,done=True)
    assert agent.updates==0
    agent.reset(agent_id=0,seed=0,training=True)
    action=agent.act(o)
    with pytest.raises(ValueError,match='nonterminal'):
        agent.observe_transition(observation=o,action=action,reward=1,next_observation=None,done=False)


def test_evaluation_is_reproducible_and_never_updates_weights():
    agent=ZhmurenkoVladimirAgent()
    agent.reset(agent_id=0,seed=0,training=False)
    before=weights(agent)
    first=agent.act(observation())
    for seed in (0,0,7):
        agent.reset(agent_id=0,seed=seed,training=False)
        action=agent.act(observation())
        assert action==first
        agent.observe_transition(observation=observation(),action=action,reward=100,next_observation=None,done=True)
    assert same_weights(before,weights(agent)) and agent.updates==0


def test_instances_and_random_streams_are_independent():
    first,second=ZhmurenkoVladimirAgent(),ZhmurenkoVladimirAgent()
    first.reset(agent_id=0,seed=10,training=True)
    second.reset(agent_id=0,seed=10,training=True)
    assert first.act(observation())==second.act(observation())
    before=weights(second)
    action=first._pending.action
    first.observe_transition(observation=observation(),action=action,reward=100,next_observation=None,done=True)
    assert same_weights(before,weights(second))
    torch.manual_seed(123)
    expected=torch.rand(3)
    torch.manual_seed(123)
    ZhmurenkoVladimirAgent()
    assert torch.equal(expected,torch.rand(3))


def test_seeded_evaluation_repeats_its_sequence_without_learning():
    agent=ZhmurenkoVladimirAgent(evaluation_mode='seeded')
    before=weights(agent)
    sequences=[]
    for _ in range(2):
        agent.reset(agent_id=0,seed=123,training=False)
        sequences.append([agent.act(observation()) for _ in range(20)])
    assert sequences[0]==sequences[1] and len(set(a.price for a in sequences[0]))>1
    assert same_weights(before,weights(agent)) and agent.updates==0


def test_resumed_optimizer_and_reset_give_identical_next_update(tmp_path):
    first=ZhmurenkoVladimirAgent(model_seed=13)
    first.reset(agent_id=0,seed=1,training=True)
    o=observation()
    action=first.act(o)
    first.observe_transition(observation=o,action=action,reward=30,next_observation=None,done=True)
    checkpoint=first.save_checkpoint(tmp_path/'resume.pt')
    second=ZhmurenkoVladimirAgent(model_seed=99)
    second.load_checkpoint(checkpoint)
    for item in (first,second):
        item.reset(agent_id=0,seed=9,training=True)
        action=item.act(o)
        item.observe_transition(observation=o,action=action,reward=60,next_observation=None,done=True)
    assert first.model_seed==second.model_seed==13
    assert same_weights(weights(first),weights(second)) and first.last_metrics==second.last_metrics


def test_checkpoint_roundtrip_and_fresh_process(tmp_path):
    agent=ZhmurenkoVladimirAgent(model_seed=17)
    agent.reset(agent_id=0,seed=1,training=True)
    o=observation()
    action=agent.act(o)
    agent.observe_transition(observation=o,action=action,reward=123,next_observation=None,done=True)
    path=agent.save_checkpoint(tmp_path/'agent_0.pt')
    with pytest.raises(FileExistsError):
        agent.save_checkpoint(path)
    restored=ZhmurenkoVladimirAgent(checkpoint_path=str(path))
    assert same_weights(weights(agent),weights(restored)) and restored.updates==1
    assert restored.optimizer.state_dict()['state']
    for item in (agent,restored):
        item.reset(agent_id=0,seed=8,training=False)
    assert agent.act(o)==restored.act(o)
    obs_path=tmp_path/'observation.json'
    obs_path.write_text(json.dumps(asdict(o)),encoding='utf-8')
    code='''import json,sys
from dataclasses import asdict
from social_market_env.contracts import AgentObservation,MarketInfo,AgentExpectations
from social_market_env.implementations.zhmurenko_vladimir import ZhmurenkoVladimirAgent
data=json.load(open(sys.argv[2],encoding='utf-8'))
data['market']=MarketInfo(**data['market'])
data['expectations']=AgentExpectations(**data['expectations'])
agent=ZhmurenkoVladimirAgent(checkpoint_path=sys.argv[1])
agent.reset(agent_id=0,seed=8,training=False)
print(json.dumps(asdict(agent.act(AgentObservation(**data)))))
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(path),str(obs_path)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==json.loads(json.dumps(asdict(agent.act(o))))


def test_checkpoint_rejects_schema_and_market_mismatch(tmp_path):
    agent=ZhmurenkoVladimirAgent()
    path=agent.save_checkpoint(tmp_path/'agent.pt')
    with pytest.raises(ValueError,match='Market'):
        ZhmurenkoVladimirAgent(market={'p_max':80},checkpoint_path=str(path))
    payload=read_checkpoint(path)
    payload['feature_version']=999
    bad=tmp_path/'bad.pt'
    torch.save(payload,bad)
    with pytest.raises(ValueError,match='schema'):
        ZhmurenkoVladimirAgent(checkpoint_path=str(bad))


def test_real_delayed_terminal_and_frozen_episode(tmp_path):
    env=build_environment(config=small_config(),seed=0)
    own=env.agents[0]
    records,learning,_=play_episode(env,seed=0,episode=0,training=True,directory=tmp_path/'train',learner=own)
    assert len(records)==4 and own.updates==4 and len(learning)==4 and own._pending is None
    assert not any(record['events'] for record in records)
    before=weights(own)
    play_episode(env,seed=0,episode=1,training=False,directory=tmp_path/'eval')
    assert own.updates==4 and same_weights(before,weights(own))


def test_real_elimination_releases_our_terminal_transition():
    env=build_environment(config=small_config(finance={'initial_balance':-100,'max_negative_steps':0}),seed=0)
    own=env.agents[0]
    env.reset(seed=0,training=True)
    force_head(own,'participation',0)
    result=env.step()
    assert not result.record['active'][0] and own.updates==1 and own._pending is None


def test_real_episode_with_standard_coalitions_and_arbiter():
    env=build_environment(config=small_config(coalitions={'policy':{'impl':'social_market_env.baselines.coalitions:ConnectedComponentsCoalitions'}},
        arbiter={'policy':{'impl':'social_market_env.baselines.arbiters:ThresholdArbiter'}}),seed=0)
    env.reset(seed=0,training=True)
    while not env.done:
        result=env.step()
        assert result.record['events']==[]
    assert env.agents[0].updates==4


def test_training_hands_off_weights_between_seeds(tmp_path):
    raw={'horizon':2,'seeds':[0,1],'episodes_per_seed':1,'agents':{'count':2,'overrides':{0:{'impl':AGENT_IMPL}}},
         'network':{'mode':'static','w_coop':0}}
    path=tmp_path/'train.yaml'
    path.write_text(yaml.safe_dump(raw),encoding='utf-8')
    report=train(config_path=path,output_dir=tmp_path/'output')
    assert report['updates']==4 and report['parameter_delta_l2']>0 and report['learning_observations']==4
    assert report['losses_finite']
    assert read_checkpoint(tmp_path/'output/agent_0_seed_0.pt')['updates']==2
    assert read_checkpoint(Path(report['checkpoint']))['updates']==4


def test_output_guard_rejects_other_repository_directories():
    repository=next(parent for parent in PACKAGE_ROOT.parents if (parent/'.git').exists())
    with pytest.raises(ValueError,match='must be under'):
        writable_path(repository/'tests'/'should_not_exist.pt')
