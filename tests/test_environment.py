from dataclasses import fields
from typing import Any

from social_market_env.baselines.agents import BestResponseAgent, CartelAgent
from social_market_env.baselines.arbiters import AlwaysObserve, PunishAll
from social_market_env.baselines.coalitions import ConnectedComponentsCoalitions, NoCoalitions
from social_market_env.config import ExperimentConfig, config_from_dict
from social_market_env.contracts import AgentAction, AgentObservation, ArbiterObservation, MarketAgentAPI
from social_market_env.core.guard import CallGuard
from social_market_env.environment import SocialMarketEnvironment

COSTS = {agent_id: 18.0 for agent_id in range(4)}


def small_config(horizon: int = 8) -> ExperimentConfig:
    return config_from_dict({'horizon': horizon, 'network': {'topology': 'complete', 'mode': 'static', 'w_coop': 0.0}})


def make_env(**overrides: Any) -> SocialMarketEnvironment:
    parts = {
        'agents': {agent_id: BestResponseAgent() for agent_id in COSTS},
        'costs': COSTS,
        'coalition_policy': NoCoalitions(),
        'arbiter': AlwaysObserve(),
        'config': small_config(),
    }
    parts.update(overrides)
    return SocialMarketEnvironment(**parts)


def run_to_end(env: SocialMarketEnvironment) -> list[dict[str, Any]]:
    env.reset(seed=0, training=False)
    records = []
    while not env.done:
        records.append(env.step().record)
    return records


class CrashingAgent(MarketAgentAPI):
    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        return None

    def act(self, observation: AgentObservation) -> AgentAction:
        raise RuntimeError('boom')


class RecordingAgent(BestResponseAgent):
    def __init__(self) -> None:
        super().__init__()
        self.transitions: list[dict[str, Any]] = []

    def observe_transition(self, *, observation, action, reward, next_observation, done) -> None:
        self.transitions.append({'step': observation.step, 'next': next_observation, 'done': done})


def test_episode_runs_until_horizon_and_logs_every_step() -> None:
    records = run_to_end(make_env())

    assert [record['step'] for record in records] == list(range(8))
    assert set(records[0]['prices']) == set(COSTS)


def test_cartel_on_complete_graph_is_harmful_and_missed_by_passive_arbiter() -> None:
    env = make_env(
        agents={agent_id: CartelAgent() for agent_id in COSTS}, coalition_policy=ConnectedComponentsCoalitions()
    )

    records = run_to_end(env)

    assert all(record['plans'][0]['harmful'] for record in records)
    assert all(record['arbiter']['false_negative'] for record in records)
    assert records[-1]['average_price'] > 60.0


def test_punish_all_scores_worse_than_observing() -> None:
    observe = run_to_end(make_env())
    punish = run_to_end(make_env(arbiter=PunishAll()))

    assert sum(r['arbiter']['reward'] for r in punish) < sum(r['arbiter']['reward'] for r in observe)


def test_crashing_agent_falls_back_when_not_strict() -> None:
    agents = {agent_id: BestResponseAgent() for agent_id in COSTS} | {0: CrashingAgent()}

    records = run_to_end(make_env(agents=agents, guard=CallGuard(strict=False, time_limit=1.0)))

    assert records[0]['events'][0]['role'] == 'agent:0'
    assert 0 not in records[0]['prices']


def test_arbiter_observation_has_no_hidden_fields() -> None:
    names = {field.name for field in fields(ArbiterObservation)}

    assert names.isdisjoint({'edges', 'plans', 'coalitions', 'link_costs', 'offers'})


def test_learning_agent_gets_next_observation_and_final_done() -> None:
    recorder = RecordingAgent()
    agents = {agent_id: BestResponseAgent() for agent_id in COSTS} | {1: recorder}

    run_to_end(make_env(agents=agents, config=small_config(horizon=3)))

    assert [t['step'] for t in recorder.transitions] == [0, 1, 2]
    assert recorder.transitions[0]['next'].step == 1
    assert (recorder.transitions[-1]['next'], recorder.transitions[-1]['done']) == (None, True)


class FailingLearnerCoalitions(NoCoalitions):
    def observe_transition(self, *, observation, action, reward, next_observation, done) -> None:
        raise RuntimeError('learn boom')


def test_terminal_learning_failures_do_not_leak_into_next_episode() -> None:
    env = make_env(
        coalition_policy=FailingLearnerCoalitions(),
        config=small_config(horizon=2),
        guard=CallGuard(strict=False, time_limit=1.0),
    )
    run_to_end(env)

    second_episode = run_to_end(env)

    assert second_episode[0]['events'] == []


class SeedRecordingAgent(BestResponseAgent):
    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        self.seed = seed


class SeedRecordingCoalitions(NoCoalitions):
    def reset(self, *, seed: int, training: bool) -> None:
        self.seed = seed


class SeedRecordingArbiter(AlwaysObserve):
    def reset(self, *, seed: int, training: bool) -> None:
        self.seed = seed


def reset_seeds(seed: int) -> list[int]:
    env = make_env(
        agents={agent_id: SeedRecordingAgent() for agent_id in COSTS},
        coalition_policy=SeedRecordingCoalitions(),
        arbiter=SeedRecordingArbiter(),
    )
    env.reset(seed=seed, training=False)
    return [*(env.agents[agent_id].seed for agent_id in sorted(COSTS)), env.coalition_policy.seed, env.arbiter.seed]


def test_components_get_distinct_plain_int_seeds_that_repeat_for_the_same_reset_seed() -> None:
    first = reset_seeds(7)
    again = reset_seeds(7)

    assert len(set(first)) == len(first)
    assert {type(seed) for seed in first} == {int}
    assert first == again
    assert first != reset_seeds(8)
