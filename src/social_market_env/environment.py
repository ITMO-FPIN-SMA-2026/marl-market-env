"""The referee of the market game: owns the state and calls the roles in order."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from social_market_env.config import ExperimentConfig
from social_market_env.contracts import (
    AgentAction,
    AgentObservation,
    ArbiterAction,
    ArbiterActionKind,
    ArbiterAPI,
    CoalitionDecision,
    CoalitionPolicyAPI,
    MarketAgentAPI,
    Pair,
)
from social_market_env.core.coalitions import (
    assess_compliance,
    build_offers,
    cooperation_signals,
    distribute_pool,
    validate_plans,
)
from social_market_env.core.finance import Account, settle
from social_market_env.core.guard import CallGuard
from social_market_env.core.market import clear_market
from social_market_env.core.network import (
    apply_link_cost_changes,
    collect_link_requests,
    create_network,
    evolve_edges,
    link_fees,
    update_pair_memory,
)
from social_market_env.core.observations import (
    build_agent_observation,
    build_arbiter_observation,
    build_coalition_observation,
)
from social_market_env.core.records import build_step_record
from social_market_env.core.sanitize import (
    NO_PARTICIPATION,
    sanitize_agent_action,
    sanitize_arbiter_action,
    sanitize_coalition_decision,
)
from social_market_env.core.scoring import arbiter_reward, find_harm, judge_arbiter, social_welfare
from social_market_env.core.seeding import spawn_seeds
from social_market_env.core.state import EnvState, StepContext, remember_step
from social_market_env.core.transitions import TransitionBuffer
from social_market_env.gametheory.equilibrium import (
    CharacteristicFunction,
    Equilibrium,
    coalition_equilibrium,
    nash_equilibrium,
)
from social_market_env.gametheory.graphs import all_pairs
from social_market_env.gametheory.payouts import coalition_surplus


@dataclass(frozen=True)
class StepResult:
    """What one step produced."""

    record: dict[str, Any]
    """JSON-friendly step record."""
    done: bool
    """True when the episode is over."""


class SocialMarketEnvironment:
    """Runs the step G_t -> coalitions -> agents -> market -> SW -> arbiter -> G_{t+1}."""

    def __init__(
        self,
        *,
        agents: dict[int, MarketAgentAPI],
        costs: dict[int, float],
        coalition_policy: CoalitionPolicyAPI,
        arbiter: ArbiterAPI,
        config: ExperimentConfig,
        guard: CallGuard | None = None,
    ) -> None:
        if set(agents) != set(costs):
            raise ValueError('agents and costs must have the same ids')
        self.agents = agents
        self.costs = dict(costs)
        self.coalition_policy = coalition_policy
        self.arbiter = arbiter
        self.config = config
        self.guard = guard or CallGuard(strict=True, time_limit=config.act_seconds_limit)
        self.transitions = TransitionBuffer(guard=self.guard)
        self._state: EnvState | None = None
        self._nash_cache: dict[frozenset[int], Equilibrium] = {}
        self._cartel_cache: dict[frozenset[int], Equilibrium] = {}
        self._value_cache: dict[frozenset[int], CharacteristicFunction] = {}

    @property
    def state(self) -> EnvState:
        """Current episode state."""
        if self._state is None:
            raise RuntimeError('Call reset() before using the environment')
        return self._state

    @property
    def done(self) -> bool:
        """True when the horizon is reached or nobody is active."""
        return self.state.step >= self.config.horizon or not self.state.active_ids()

    def reset(self, *, seed: int, training: bool) -> None:
        """Start a new episode: new network, full reserves, components reset with seeds derived from seed."""
        ids = sorted(self.agents)
        env_seed, coalition_seed, arbiter_seed, *agent_seeds = spawn_seeds(seed=seed, count=3 + len(ids))
        rng = np.random.default_rng(env_seed)
        self._state = EnvState(
            step=0,
            rng=rng,
            training=training,
            accounts={agent_id: Account(balance=self.config.finance.initial_balance) for agent_id in ids},
            network=create_network(agent_ids=ids, costs=self.costs, config=self.config.network, rng=rng),
        )
        self.transitions.clear()
        self.guard.drain_events()
        for agent_id, agent_seed in zip(ids, agent_seeds):
            reset_agent = partial(self.agents[agent_id].reset, agent_id=agent_id, seed=agent_seed, training=training)
            self.guard.call(role=f'agent:{agent_id}', method='reset', fn=reset_agent, fallback=None)
        reset_coalition = partial(self.coalition_policy.reset, seed=coalition_seed, training=training)
        self.guard.call(role='coalition', method='reset', fn=reset_coalition, fallback=None)
        reset_arbiter = partial(self.arbiter.reset, seed=arbiter_seed, training=training)
        self.guard.call(role='arbiter', method='reset', fn=reset_arbiter, fallback=None)

    def step(self) -> StepResult:
        """Play one step and return its record."""
        state = self.state
        if self.done:
            raise RuntimeError('Episode is over, call reset()')
        context = StepContext()
        self._coalition_phase(state=state, context=context)
        self._agent_phase(state=state, context=context)
        self._market_phase(state=state, context=context)
        self._arbiter_phase(state=state, context=context)
        self._network_phase(state=state, context=context)
        self._finance_phase(state=state, context=context)
        record = build_step_record(state=state, context=context, events=self.guard.drain_events())
        state.step += 1
        done = self.done
        self._learning_phase(state=state, context=context, done=done)
        return StepResult(record=record, done=done)

    def global_state(self) -> dict[str, Any]:
        """Return the full hidden state for centralized training in library mode."""
        state = self.state
        active = set(state.active_ids())
        return {
            'step': state.step,
            'costs': dict(self.costs),
            'balances': {agent_id: account.balance for agent_id, account in state.accounts.items()},
            'active': sorted(active),
            'edges': state.network.edge_weights(active=active),
            'link_costs': dict(state.network.link_costs),
            'last_prices': dict(state.last_prices),
        }

    def _coalition_phase(self, *, state: EnvState, context: StepContext) -> None:
        active = set(state.active_ids())
        context.edges_snapshot = state.network.edge_weights(active=active)
        observation = build_coalition_observation(state=state, costs=self.costs)
        self.transitions.release(key='coalition', next_observation=observation, done=False)
        decision = self.guard.call(
            role='coalition',
            method='act',
            fn=lambda: sanitize_coalition_decision(self.coalition_policy.act(observation)),
            fallback=CoalitionDecision(),
        )
        eligible = {agent_id for agent_id in active if not state.accounts[agent_id].restricted}
        validation = validate_plans(
            decision=decision, eligible=eligible, edges=set(context.edges_snapshot), market=self.config.market
        )
        context.coalition_observation = observation
        context.decision = decision
        context.plans = validation.accepted
        context.rejections = validation.rejected

    def _agent_phase(self, *, state: EnvState, context: StepContext) -> None:
        offers = build_offers(context.plans)
        for agent_id in state.active_ids():
            observation = build_agent_observation(
                state=state, agent_id=agent_id, cost=self.costs[agent_id], offer=offers.get(agent_id)
            )
            self.transitions.release(key=f'agent:{agent_id}', next_observation=observation, done=False)
            context.agent_observations[agent_id] = observation
            context.actions[agent_id] = self._agent_action(state=state, agent_id=agent_id, observation=observation)

    def _agent_action(self, *, state: EnvState, agent_id: int, observation: AgentObservation) -> AgentAction:
        agent = self.agents[agent_id]
        restricted = state.accounts[agent_id].restricted
        return self.guard.call(
            role=f'agent:{agent_id}',
            method='act',
            fn=lambda: sanitize_agent_action(
                action=agent.act(observation),
                cost=self.costs[agent_id],
                restricted=restricted,
                market=self.config.market,
            ),
            fallback=NO_PARTICIPATION,
        )

    def _market_phase(self, *, state: EnvState, context: StepContext) -> None:
        market = self.config.market
        participating = {agent_id for agent_id, action in context.actions.items() if action.participate}
        prices = {agent_id: context.actions[agent_id].price for agent_id in sorted(participating)}
        context.outcome = clear_market(prices=prices, costs=self.costs, market=market)
        context.compliance = assess_compliance(
            plans=context.plans,
            actions=context.actions,
            participating=participating,
            tolerance=self.config.coalitions.compliance_tolerance,
        )
        context.received = distribute_pool(
            plans=context.plans, compliance=context.compliance, profits=context.outcome.gross_profits
        )
        self._welfare_phase(state=state, context=context, prices=prices)

    def _welfare_phase(self, *, state: EnvState, context: StepContext, prices: dict[int, float]) -> None:
        market = self.config.market
        active = frozenset(state.active_ids())
        nash = self._nash(active)
        context.welfare = social_welfare(total_quantity=context.outcome.total_quantity, market=market)
        context.target_welfare = social_welfare(total_quantity=nash.total_quantity, market=market)
        context.nash_average_price = _mean_price(nash)
        context.cartel_average_price = _mean_price(self._cartel(active))
        context.harm = find_harm(
            compliance=context.compliance,
            prices=prices,
            costs=self.costs,
            nash_prices=nash.prices,
            market=market,
            margin=self.config.scoring.harm_margin,
        )

    def _arbiter_phase(self, *, state: EnvState, context: StepContext) -> None:
        observation = build_arbiter_observation(state=state, context=context)
        self.transitions.release(key='arbiter', next_observation=observation, done=False)
        active = set(state.active_ids())
        action = self.guard.call(
            role='arbiter',
            method='act',
            fn=lambda: sanitize_arbiter_action(
                action=self.arbiter.act(observation), active=active, config=self.config.arbiter
            ),
            fallback=ArbiterAction(),
        )
        apply_link_cost_changes(
            state=state.network, changes=action.link_cost_changes, decay=self.config.network.link_cost_decay
        )
        context.arbiter_observation = observation
        context.arbiter_action = action
        is_sanction = action.kind == ArbiterActionKind.sanction and action.target is not None
        context.fines = {action.target: action.fine} if is_sanction else {}
        self._score_arbiter(context)

    def _score_arbiter(self, context: StepContext) -> None:
        context.verdict = judge_arbiter(
            action=context.arbiter_action,
            harmful_members=context.harm.members,
            harmful_pairs=context.harm.pairs,
            prevention_counts=self.config.network.mode == 'evolving',
        )
        context.arbiter_reward = arbiter_reward(
            welfare=context.welfare,
            target_welfare=context.target_welfare,
            verdict=context.verdict,
            config=self.config.scoring,
        )

    def _network_phase(self, *, state: EnvState, context: StepContext) -> None:
        network_config = self.config.network
        update_pair_memory(
            state=state.network,
            cooperation=cooperation_signals(context.compliance),
            utility=self._utility_signals(context),
            rate=network_config.memory_rate,
        )
        requests = collect_link_requests(context.actions)
        changes = evolve_edges(
            state=state.network,
            active=set(state.active_ids()),
            requests=requests,
            expected_gain=self._expected_gains(state),
            config=network_config,
            rng=state.rng,
        )
        context.link_fees = link_fees(formed=changes.formed, requests=requests, link_costs=state.network.link_costs)

    def _finance_phase(self, *, state: EnvState, context: StepContext) -> None:
        for agent_id in context.actions:
            reward = (
                context.received.get(agent_id, 0.0)
                - context.fines.get(agent_id, 0.0)
                - context.link_fees.get(agent_id, 0.0)
            )
            context.agent_rewards[agent_id] = reward
            settle(
                account=state.accounts[agent_id],
                amount=reward,
                max_negative_steps=self.config.finance.max_negative_steps,
            )
        context.coalition_reward = sum(
            context.received.get(member, 0.0) - context.fines.get(member, 0.0)
            for record in context.compliance
            for member in record.compliers
        )
        remember_step(state=state, context=context)

    def _learning_phase(self, *, state: EnvState, context: StepContext, done: bool) -> None:
        for agent_id, action in context.actions.items():
            key = f'agent:{agent_id}'
            self.transitions.hold(
                key=key,
                learner=self.agents[agent_id],
                observation=context.agent_observations[agent_id],
                action=action,
                reward=context.agent_rewards[agent_id],
            )
            if not state.accounts[agent_id].active:
                self.transitions.release(key=key, next_observation=None, done=True)
        self.transitions.hold(
            key='coalition',
            learner=self.coalition_policy,
            observation=context.coalition_observation,
            action=context.decision,
            reward=context.coalition_reward,
        )
        self.transitions.hold(
            key='arbiter',
            learner=self.arbiter,
            observation=context.arbiter_observation,
            action=context.arbiter_action,
            reward=context.arbiter_reward,
        )
        if done:
            self.transitions.release_all()

    def _utility_signals(self, context: StepContext) -> dict[Pair, float]:
        scale = self.config.network.utility_scale
        received = context.received
        return {
            pair: math.tanh((received[pair[0]] + received[pair[1]]) / (2.0 * scale)) for pair in all_pairs(received)
        }

    def _expected_gains(self, state: EnvState) -> dict[Pair, float]:
        if self.config.network.w_coop == 0:
            return {}
        active = frozenset(state.active_ids())
        value = self._value_function(active)
        scale = self.config.network.utility_scale
        return {
            pair: math.tanh(coalition_surplus(members=frozenset(pair), value=value) / scale)
            for pair in all_pairs(active)
        }

    def _nash(self, active: frozenset[int]) -> Equilibrium:
        if active not in self._nash_cache:
            costs = {agent_id: self.costs[agent_id] for agent_id in active}
            self._nash_cache[active] = nash_equilibrium(costs=costs, market=self.config.market)
        return self._nash_cache[active]

    def _cartel(self, active: frozenset[int]) -> Equilibrium:
        if active not in self._cartel_cache:
            costs = {agent_id: self.costs[agent_id] for agent_id in active}
            self._cartel_cache[active] = coalition_equilibrium(members=active, costs=costs, market=self.config.market)
        return self._cartel_cache[active]

    def _value_function(self, active: frozenset[int]) -> CharacteristicFunction:
        if active not in self._value_cache:
            costs = {agent_id: self.costs[agent_id] for agent_id in active}
            self._value_cache[active] = CharacteristicFunction(costs=costs, market=self.config.market)
        return self._value_cache[active]


def _mean_price(equilibrium: Equilibrium) -> float:
    return sum(equilibrium.prices.values()) / len(equilibrium.prices)
