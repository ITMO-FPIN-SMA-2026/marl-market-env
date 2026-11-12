import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from social_market_env.contracts import AgentAction, AgentObservation, CoalitionOffer, MarketInfo
from social_market_env.config import load_config
from zhmurenko_agent.agent import (
    ACTIONS, JOIN, WAIT, PACKAGE, QSettings, ZhmurenkoVladimirAgent, output_path,
)
from zhmurenko_agent.research.experiment import (
    BASE_CONFIG, IMPL, play, scenario_config,
)


def observation(step=0, rival=25, **kwargs):
    fields = dict(agent_id=0, step=step, cost=20.0, balance=1000.0, active=True, restricted=False,
                  last_price=25.0 if step else None, last_quantity=8.0, last_profit=40.0,
                  market=MarketInfo(last_prices={1:rival}, average_price=rival, total_quantity=90),
                  neighbors={1:0.0}, link_costs={1:0.0}, offer=None)
    return AgentObservation(**(fields | kwargs))


def learner(**settings):
    agent = ZhmurenkoVladimirAgent(fresh=True, settings=settings)
    agent.reset(agent_id=0, seed=9, training=True)
    return agent


@pytest.mark.parametrize('settings', [dict(gamma=1.1), dict(learning_rate=0), dict(epsilon_start=-1),
    dict(reward_scale=0), dict(use_regime_context=1), dict(context_min_visits=0), dict(gamma=float('nan'))])
def test_settings_reject_invalid_values(settings):
    with pytest.raises(ValueError):
        QSettings(**settings)


def test_missing_checkpoint_never_silently_creates_untrained_agent(tmp_path):
    with pytest.raises(FileNotFoundError, match='Trained Q-table missing'):
        ZhmurenkoVladimirAgent(model_path=tmp_path/'absent.json')


def test_cold_policy_is_explicit_and_cost_relative():
    agent = learner(epsilon_start=0, epsilon_end=0)
    first = agent.act(observation())
    assert first.price == 25 and first.participate
    assert len(agent.available_actions(observation())) == len(ACTIONS)-1  # no offer
    second = agent.act(observation(cost=30))
    assert second.price == 35


def test_action_masks_and_exact_offer_price():
    agent = learner()
    offer = CoalitionOffer(plan_id=91, members=(0,1), recommended_price=25.125, payout_weight=.8)
    o = observation(offer=offer)
    assert agent.available_actions(o)[JOIN] == AgentAction(price=25.125, coalition_choice=91)
    assert JOIN not in agent.available_actions(replace(o, restricted=True))
    assert JOIN not in agent.available_actions(replace(o, offer=replace(offer, recommended_price=19)))
    assert set(agent.available_actions(replace(o, active=False))) == {WAIT}
    assert set(agent.available_actions(replace(o, cost=101))) == {WAIT}


def test_terminal_bellman_update_uses_real_reward_and_never_bootstraps():
    agent = learner(epsilon_start=0, epsilon_end=0, learning_rate=.5, gamma=.9, reward_scale=100)
    o = observation(); action = agent.act(o)
    agent.observe_transition(observation=o, action=action, reward=80, next_observation=observation(1), done=True)
    key, context = agent.state(o)
    assert agent.q[key][2] == pytest.approx(.4)
    assert agent.visits[key][2] == 1 and agent.updates == 1
    assert sum(sum(row) for row in agent.context_visits.values()) == 1


def test_next_maximum_excludes_unavailable_coalition():
    agent = learner(epsilon_start=0, epsilon_end=0, learning_rate=1, gamma=.5)
    o = observation(); key, _ = agent.state(o)
    agent.q[key] = [0.0]*len(ACTIONS); agent.q[key][JOIN] = 1000
    agent.visits[key] = [0]*len(ACTIONS)
    action = agent.act(o)
    agent.observe_transition(observation=o, action=action, reward=100, next_observation=observation(1), done=False)
    assert agent.q[key][2] == pytest.approx(1.0)


def test_same_current_market_different_history_has_different_context():
    down, up = learner(), learner()
    down.state(observation(0, rival=35), commit=True)
    up.state(observation(0, rival=20), commit=True)
    current = observation(1, rival=25)
    assert down.state(current)[0] == up.state(current)[0]
    assert down.state(current)[1][-1] == 0
    assert up.state(current)[1][-1] == 2


def test_history_preview_is_pure_idempotent_and_reset_clears_it():
    agent = learner(); agent.state(observation(0), commit=True)
    before = agent._step, agent._median, agent._direction
    first = agent.state(observation(1, rival=30))
    assert before == (agent._step, agent._median, agent._direction)
    assert first == agent.state(observation(1, rival=30), commit=True)
    assert first == agent.state(observation(1, rival=30), commit=True)
    agent.reset(agent_id=0, seed=0, training=False)
    assert agent.state(observation(0, rival=30))[1][-1] == 1


def test_sparse_regime_uses_learned_shared_values_then_qualified_specialization():
    agent = learner(context_min_visits=3)
    o = observation(); key, context = agent.state(o)
    agent.q[key] = [0.0]*len(ACTIONS); agent.q[key][3] = 5
    agent.visits[key] = [0]*len(ACTIONS)
    agent.context_q[context] = [0.0]*len(ACTIONS); agent.context_q[context][4] = 10
    agent.context_visits[context] = [0]*len(ACTIONS)
    values, _ = agent._values(key, context)
    assert values[3] == 5 and values[4] == 0
    agent.context_visits[context][4] = 3
    assert agent._values(key, context)[0] == agent.q[key]
    agent.context_visits[context] = [3]*len(ACTIONS)
    assert 0 < agent._values(key, context)[0][4] < 10


def test_context_confidence_cannot_change_ranking_for_a_common_value_offset():
    agent=learner(context_min_visits=3)
    key,context=agent.state(observation())
    shared=[0.0]*len(ACTIONS);shared[3]=5.0
    agent.q[key]=shared;agent.visits[key]=[10]*len(ACTIONS)
    agent.context_q[context]=[v+100 for v in shared]
    agent.context_visits[context]=[3]*len(ACTIONS)
    agent.context_visits[context][4]=1000
    values,_=agent._values(key,context)
    assert max(range(len(values)),key=lambda i:values[i])==3


def test_disabled_specialization_cannot_change_shared_policy():
    agent = learner(use_regime_context=False)
    o = observation(); action = agent.act(o)
    agent.observe_transition(observation=o, action=action, reward=10, next_observation=None, done=True)
    assert agent.context_q == {} and agent.context_visits == {}
    assert agent.stats['regime_changed_decisions'] == 0


def test_coalition_can_be_selected_from_learned_values_without_an_eval_rule():
    agent = learner(epsilon_start=0, epsilon_end=0, learning_rate=1, gamma=0)
    offer = CoalitionOffer(plan_id=3, members=(0,1), recommended_price=25.0, payout_weight=.85)
    o = observation(offer=offer)
    # Sampling the join action, then delivering a real-reward API transition,
    # tests attribution. Market-generated coalition experience is tested below.
    key, context = agent.state(o)
    action = agent.available_actions(o)[JOIN]
    agent._pending = (o.step, key, context, JOIN, action)
    agent.observe_transition(observation=o, action=action, reward=100, next_observation=None, done=True)
    agent.reset(agent_id=0, seed=2, training=False)
    assert agent.act(o).coalition_choice == 3


def test_changed_actions_are_not_credited_to_another_q_entry():
    agent = learner(); o=observation(); action=agent.act(o)
    with pytest.raises(ValueError, match='changed action'):
        agent.observe_transition(observation=o, action=replace(action, price=action.price+.1), reward=10,
                                 next_observation=None, done=True)
    assert agent.updates == 0


def trained_model(tmp_path):
    agent = learner(epsilon_start=0, epsilon_end=0)
    o = observation(); action = agent.act(o)
    agent.observe_transition(observation=o, action=action, reward=100, next_observation=None, done=True)
    path = tmp_path/'agent.json'; agent.save(path)
    return agent, path


def test_json_roundtrip_and_frozen_evaluation(tmp_path):
    original, path = trained_model(tmp_path)
    restored = ZhmurenkoVladimirAgent(model_path=path)
    assert restored.q == original.q and restored.visits == original.visits
    assert restored.context_q == original.context_q and restored.updates == original.updates
    before = copy.deepcopy((restored.q, restored.visits, restored.context_q, restored.updates))
    restored.reset(agent_id=0, seed=0, training=False)
    o=observation(); action=restored.act(o)
    restored.observe_transition(observation=o, action=action, reward=10000, next_observation=None, done=True)
    assert before == (restored.q, restored.visits, restored.context_q, restored.updates)


@pytest.mark.parametrize('field, change', [('format_version',99), ('actions',[]), ('updates',0), ('state_sizes',[99])])
def test_bad_checkpoint_is_rejected(tmp_path, field, change):
    _,path=trained_model(tmp_path)
    payload=json.loads(path.read_text(encoding='utf-8'));payload[field]=change
    path.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError): ZhmurenkoVladimirAgent(model_path=path)


def test_refuse_saving_in_other_repository_folders():
    with pytest.raises(ValueError, match='must stay within'):
        output_path(BASE_CONFIG.parent/'forbidden-output.json')


def test_position_rotation_and_single_own_slot():
    base=load_config(BASE_CONFIG)
    seen=set()
    for i in range(12):
        config,pos=scenario_config(base,scenario='mixed',index=i*6,position_index=i,
                                  focal={'impl':IMPL,'kwargs':{'fresh':True}})
        seen.add(pos)
        assert [j for j,spec in config.agents.overrides.items() if spec.impl==IMPL] == [pos]
    assert seen==set(range(12))


def test_real_environment_delivers_learning_and_coalition_experience(tmp_path):
    base=load_config(BASE_CONFIG);base.horizon=12
    agent=learner(epsilon_start=1,epsilon_end=1)
    count=0
    for i in range(4):
        config,pos=scenario_config(base,scenario='pairs',index=i,
                                  focal={'impl':IMPL,'kwargs':{'fresh':True}},fast=True)
        metrics=play(agent,config,position=pos,seed=300+i,training=True)
        count+=metrics['updates']
    assert agent.updates==count==48
    assert sum(row[JOIN] for row in agent.visits.values()) > 0
    agent.save(tmp_path/'market.json')
    assert ZhmurenkoVladimirAgent(model_path=tmp_path/'market.json').updates==48


def test_block_resume_matches_uninterrupted_q_learning(tmp_path):
    from types import SimpleNamespace
    from zhmurenko_agent.research.experiment import train_pair
    base=load_config(BASE_CONFIG);base.horizon=2
    settings=dict(repeat=0,variants=['flat','regime'],block=2,min_episodes=2,patience=99)
    train_pair(SimpleNamespace(root=tmp_path/'full',max_episodes=4,resume=False,**settings),base)
    train_pair(SimpleNamespace(root=tmp_path/'resumed',max_episodes=2,resume=False,**settings),base)
    train_pair(SimpleNamespace(root=tmp_path/'resumed',max_episodes=4,resume=True,**settings),base)
    for name in ('flat','regime'):
        first=json.loads((tmp_path/'full/repeat_0'/name/'model.json').read_text(encoding='utf-8'))
        second=json.loads((tmp_path/'resumed/repeat_0'/name/'model.json').read_text(encoding='utf-8'))
        assert first==second


def test_reduced_final_plan_excludes_v4_and_pytorch(tmp_path):
    import sys
    from zhmurenko_agent.research.experiment import FINAL_SEEDS,EVAL_SCENARIOS,model_variants
    names=[name for name,_ in model_variants(tmp_path)]
    assert len(FINAL_SEEDS)==12 and len(EVAL_SCENARIOS)==7 and len(names)==8
    assert len(FINAL_SEEDS)*len(EVAL_SCENARIOS)*len(names)==672
    assert all('V4' not in name for name in names)
    assert 'torch' not in sys.modules
