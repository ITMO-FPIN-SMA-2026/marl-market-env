from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .config import EnvironmentConfig
from .contracts import (
    AgentAction,
    AgentEvaluation,
    AgentExpectations,
    AgentObservation,
    ArbiterAction,
    ArbiterActionType,
    ArbiterAPI,
    ArbiterObservation,
    CoalitionDecision,
    CoalitionObservation,
    CoalitionPolicyAPI,
    MarketAgentAPI,
)
from .myerson import graph_matrices, myerson_value


@dataclass
class AgentState:
    cost: float
    balance: float
    active: bool = True
    negative_steps: int = 0
    last_profit: float = 0.0
    last_price: Optional[float] = None
    observations: List[AgentObservation] = field(default_factory=list)
    expectations: AgentExpectations = field(default_factory=AgentExpectations)
    expectation_history: List[AgentExpectations] = field(default_factory=list)
    evaluations: List[AgentEvaluation] = field(default_factory=list)
    coalition_join_vector: Tuple[int, ...] = ()


class SocialMarketEnvironment:

    def __init__(
        self,
        agents: Dict[int, MarketAgentAPI],
        agent_costs: Dict[int, float],
        coalition_policy: CoalitionPolicyAPI,
        arbiter: ArbiterAPI,
        config: Optional[EnvironmentConfig] = None,
    ) -> None:
        if set(agents) != set(agent_costs):
            raise ValueError("agents and agent_costs must have identical IDs")

        self.agents = agents
        self.agent_costs = dict(agent_costs)
        self.coalition_policy = coalition_policy
        self.arbiter = arbiter
        self.config = config or EnvironmentConfig()
        self.rng = random.Random(self.config.seed)

        self.step_index = 0
        self.agent_state: Dict[int, AgentState] = {}
        self.graph = nx.Graph()
        self.edge_similarity: Dict[Tuple[int, int], float] = {}
        self.pair_history: Dict[Tuple[int, int], float] = {}
        self.link_cost_pressure: float = 0.0
        self.current_coalitions: Tuple[Tuple[int, ...], ...] = ()
        self._candidate_coalitions: Tuple[Tuple[int, ...], ...] = ()
        self.last_average_price: float = 0.0
        self.last_social_welfare: float = 0.0
        self.history: List[dict] = []
        self._last_agent_obs: Dict[int, AgentObservation] = {}
        self._last_arbiter_obs: Optional[ArbiterObservation] = None

    @property
    def edges(self) -> set[Tuple[int, int]]:
        return {tuple(sorted(edge)) for edge in self.graph.edges}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[int, AgentObservation]:
        if seed is not None:
            self.config.seed = seed
        self.rng.seed(self.config.seed)
        self.step_index = 0
        self.history.clear()
        self.link_cost_pressure = 0.0
        self.current_coalitions = ()
        self._candidate_coalitions = ()
        self.last_average_price = 0.0
        self.last_social_welfare = 0.0
        self._last_arbiter_obs = None

        self.agent_state = {
            agent_id: AgentState(
                cost=self.agent_costs[agent_id],
                balance=self.config.initial_balance,
            )
            for agent_id in self.agents
        }
        for aid, agent in self.agents.items():
            agent.reset(aid, self.config.seed)
        self.coalition_policy.reset(self.config.seed)
        self.arbiter.reset(self.config.seed)

        self._initialize_graph()
        obs = {aid: self._agent_observation(aid) for aid in self.agents}
        for aid, observation in obs.items():
            self.agent_state[aid].observations.append(observation)
            self.agent_state[aid].expectation_history.append(observation.expectations)
        self._last_agent_obs = obs
        return obs

    def step(self) -> Tuple[Dict[int, AgentObservation], Dict[int, float], bool, dict]:
        if self.step_index >= self.config.horizon:
            raise RuntimeError("Episode finished. Call reset().")

        # 1) G_t -> coalition formation
        coalition_obs = self._coalition_observation()
        decision = self.coalition_policy.propose(coalition_obs)
        self._candidate_coalitions = self._sanitize_coalition_candidates(decision)

        # 2) agent actions
        observations = {aid: self._agent_observation(aid) for aid in self.agents}
        actions: Dict[int, AgentAction] = {}
        for aid, agent in self.agents.items():
            self.agent_state[aid].observations.append(observations[aid])
            action = agent.act(observations[aid])
            actions[aid] = self._sanitize_agent_action(aid, action)

        self.current_coalitions = self._resolve_coalitions(actions)
        # Keep matrix dimensions stable for downstream MARL models even after
        # agents are eliminated. Inactive nodes are isolated after evolution.
        graph_nodes = tuple(sorted(self.agents))
        matrices = graph_matrices(self.graph, self.current_coalitions, graph_nodes)
        edges_before = sorted(self.edges)

        # 3) market clearing
        prices, quantities, profits, total_quantity, average_price = self._clear_market(actions)

        # The graph-restricted characteristic function gives each connected
        # coalition a surplus; Myerson allocates it by marginal contribution.
        coalition_bonus = self._coalition_bonus(profits)

        # 4) SW_t. Exact SW formula is not explicit in the supplied text, so
        # this MVP uses negative average procurement price as a replaceable proxy.
        social_welfare = self._social_welfare(average_price, total_quantity)

        # 5) arbiter action (partial observation only)
        arbiter_obs = ArbiterObservation(
            step=self.step_index,
            active_agents=tuple(a for a, s in self.agent_state.items() if s.active),
            prices=prices,
            quantities=quantities,
            total_quantity=total_quantity,
            average_price=average_price,
            social_welfare=social_welfare,
            delta_average_price=average_price - self.last_average_price,
            delta_social_welfare=social_welfare - self.last_social_welfare,
        )
        arbiter_action = self._sanitize_arbiter_action(self.arbiter.act(arbiter_obs))
        fines = self._apply_arbiter_action(arbiter_action)

        # Agent reward r_i,t = profit + coalition bonus - sanction.
        rewards: Dict[int, float] = {}
        for aid in self.agents:
            reward = profits.get(aid, 0.0) + coalition_bonus.get(aid, 0.0) - fines.get(aid, 0.0)
            rewards[aid] = reward
            self._update_balance(aid, reward, profits.get(aid, 0.0), prices.get(aid))
            self._evaluate_agent(aid, observations[aid], prices, quantities, profits, coalition_bonus, fines, reward)

        harmful_agents = self._harmful_coalition_agents(prices, average_price)
        arbiter_reward, fp, fn = self._arbiter_reward(
            social_welfare=social_welfare,
            action=arbiter_action,
            harmful_agents=harmful_agents,
        )

        # 6) G_t -> G_{t+1}
        self._update_pair_history(profits)
        self._evolve_graph(profits, actions)
        self._candidate_coalitions = ()

        self.step_index += 1
        done = self.step_index >= self.config.horizon or not any(s.active for s in self.agent_state.values())
        self.last_average_price = average_price
        self.last_social_welfare = social_welfare

        next_obs = {aid: self._agent_observation(aid) for aid in self.agents}
        for aid, observation in next_obs.items():
            self.agent_state[aid].observations.append(observation)
        for aid, agent in self.agents.items():
            agent.observe_transition(
                observations[aid], actions[aid], rewards[aid], next_obs[aid] if not done else None, done
            )

        next_arbiter_obs = None if done else self._arbiter_observation_from_current_state()
        self.arbiter.observe_transition(
            arbiter_obs, arbiter_action, arbiter_reward, next_arbiter_obs, done
        )

        record = {
            "step": self.step_index - 1,
            "prices": prices,
            "quantities": quantities,
            "profits": profits,
            "coalition_bonus": coalition_bonus,
            "agent_rewards": rewards,
            "balances": {a: s.balance for a, s in self.agent_state.items()},
            "active": {a: s.active for a, s in self.agent_state.items()},
            "edges": sorted(self.edges),
            "edges_before": edges_before,
            "graph_nodes": graph_nodes,
            "graph_matrices": matrices,
            "coalitions": [list(c) for c in self.current_coalitions],
            "coalition_join_vectors": {a: s.coalition_join_vector for a, s in self.agent_state.items()},
            "agent_expectations": {a: asdict(s.expectations) for a, s in self.agent_state.items()},
            "agent_evaluations": {a: asdict(s.evaluations[-1]) for a, s in self.agent_state.items()},
            "average_price": average_price,
            "total_quantity": total_quantity,
            "social_welfare": social_welfare,
            "arbiter_action": asdict(arbiter_action),
            "arbiter_reward": arbiter_reward,
            "false_positive": fp,
            "false_negative": fn,
        }
        self.history.append(record)
        self._last_agent_obs = next_obs
        self._last_arbiter_obs = arbiter_obs

        info = {"record": record, "arbiter_reward": arbiter_reward}
        return next_obs, rewards, done, info

    def run(self, steps: Optional[int] = None) -> List[dict]:
        if not self.agent_state:
            self.reset()
        max_steps = self.config.horizon if steps is None else min(steps, self.config.horizon)
        while self.step_index < max_steps:
            _, _, done, _ = self.step()
            if done:
                break
        return self.history

    # ------------------------------------------------------------------
    # Market
    # ------------------------------------------------------------------
    def _clear_market(self, actions: Dict[int, AgentAction]):
        active_actions = {
            aid: action
            for aid, action in actions.items()
            if self.agent_state[aid].active and action.participate
        }
        if not active_actions:
            return {}, {}, {}, 0.0, 0.0

        prices = {aid: action.price for aid, action in active_actions.items()}
        average_price = mean(prices.values())

        # From the assignment: Q(P)=max(0, A-BP).
        total_quantity = max(0.0, self.config.demand_a - self.config.demand_b * average_price)

        # The DOCX extraction shows q_i(p_i) proportional to (1 - p_i/sum p).
        # We normalize those non-negative weights to make them sum to Q.
        sum_prices = sum(prices.values())
        raw_weights = {
            aid: max(0.0, 1.0 - (price / sum_prices if sum_prices > 0 else 0.0))
            for aid, price in prices.items()
        }
        weight_sum = sum(raw_weights.values())
        if weight_sum <= 0:
            quantities = {aid: total_quantity / len(prices) for aid in prices}
        else:
            quantities = {aid: total_quantity * w / weight_sum for aid, w in raw_weights.items()}

        profits = {
            aid: (prices[aid] - self.agent_state[aid].cost) * quantities[aid]
            for aid in prices
        }
        return prices, quantities, profits, total_quantity, average_price

    def _social_welfare(self, average_price: float, total_quantity: float) -> float:
        # Replaceable proxy. The assignment says harmful coalitions worsen social
        # satisfaction through higher average procurement price, but no complete SW
        # equation is recoverable from the provided file.
        return -average_price

    # ------------------------------------------------------------------
    # Coalition layer
    # ------------------------------------------------------------------
    def _coalition_bonus(self, profits: Dict[int, float]) -> Dict[int, float]:
        bonus = {aid: 0.0 for aid in self.agents}
        for coalition in self.current_coalitions:
            members = [a for a in coalition if a in profits]
            if len(members) < 2:
                continue
            def worth(group: frozenset[int]) -> float:
                return (
                    self.config.coalition_bonus_scale * max(0.0, sum(profits[a] for a in group))
                    if len(group) >= 2 else 0.0
                )

            allocation = myerson_value(
                self.graph,
                members,
                worth,
                max_exact_players=self.config.max_exact_coalition_size,
                samples=self.config.myerson_samples,
                rng=self.rng,
            )
            for aid, value in allocation.items():
                bonus[aid] += value
        return bonus

    def _sanitize_coalition_candidates(self, decision: CoalitionDecision) -> Tuple[Tuple[int, ...], ...]:
        active = {a for a, s in self.agent_state.items() if s.active}
        result = []
        for c in decision.coalitions:
            members = tuple(sorted(set(c) & active))
            if len(members) >= 2 and members not in result:
                result.append(members)
        return tuple(result)

    def _resolve_coalitions(self, actions: Dict[int, AgentAction]) -> Tuple[Tuple[int, ...], ...]:
        """Apply each agent's one-hot join choice, then enforce graph connectivity."""
        selected: dict[int, set[int]] = {index: set() for index in range(len(self._candidate_coalitions))}
        for aid, state in self.agent_state.items():
            choices = [i for i, candidate in enumerate(self._candidate_coalitions) if aid in candidate]
            requested = actions[aid].coalition_choice
            chosen = choices[0] if requested is None and choices else requested
            if not state.active or chosen not in choices:
                chosen = None
            state.coalition_join_vector = tuple(int(i == chosen) for i in range(len(self._candidate_coalitions)))
            if chosen is not None:
                selected[chosen].add(aid)

        result: list[Tuple[int, ...]] = []
        for members in selected.values():
            for component in nx.connected_components(self.graph.subgraph(members)):
                if len(component) >= 2:
                    result.append(tuple(sorted(component)))
        return tuple(sorted(result))

    # ------------------------------------------------------------------
    # Arbiter
    # ------------------------------------------------------------------
    def _apply_arbiter_action(self, action: ArbiterAction) -> Dict[int, float]:
        fines = {aid: 0.0 for aid in self.agents}
        if action.kind == ArbiterActionType.SANCTION:
            if action.target_agent in self.agent_state and self.agent_state[action.target_agent].active:
                fines[action.target_agent] = max(0.0, action.fine)
        elif action.kind == ArbiterActionType.CHANGE_LINK_COST:
            self.link_cost_pressure += action.link_cost_delta
        return fines

    def _harmful_coalition_agents(self, prices: Dict[int, float], market_avg: float) -> set[int]:
        harmful = set()
        if not prices:
            return harmful
        threshold = market_avg * (1.0 + self.config.harmful_coalition_margin)
        for coalition in self.current_coalitions:
            member_prices = [prices[a] for a in coalition if a in prices]
            if member_prices and mean(member_prices) > threshold:
                harmful.update(a for a in coalition if a in prices)
        return harmful

    def _arbiter_reward(self, social_welfare: float, action: ArbiterAction, harmful_agents: set[int]):
        intervention = int(action.kind != ArbiterActionType.OBSERVE)
        fp = 0
        fn = 0
        if action.kind == ArbiterActionType.SANCTION:
            if action.target_agent not in harmful_agents:
                fp = 1
        elif harmful_agents:
            fn = 1

        cfg = self.config
        reward = (
            -cfg.welfare_deviation_penalty * abs(social_welfare - cfg.target_social_welfare)
            -cfg.intervention_penalty * intervention
            -cfg.false_positive_penalty * fp
            -cfg.false_negative_penalty * fn
        )
        return reward, fp, fn

    # ------------------------------------------------------------------
    # Dynamic network
    # ------------------------------------------------------------------
    def _initialize_graph(self) -> None:
        self.graph.clear()
        self.graph.add_nodes_from(self.agents)
        self.edge_similarity.clear()
        self.pair_history.clear()
        ids = sorted(self.agents)
        for i, u in enumerate(ids):
            for v in ids[i + 1 :]:
                key = (u, v)
                similarity = self.rng.random()
                self.edge_similarity[key] = similarity
                self.pair_history[key] = 0.0
                p = min(1.0, max(0.0, self.config.initial_edge_probability * (0.5 + similarity)))
                if self.rng.random() < p:
                    self.graph.add_edge(u, v)

    def _evolve_graph(self, profits: Dict[int, float], actions: Dict[int, AgentAction]) -> None:
        active = {a for a, s in self.agent_state.items() if s.active}
        ids = sorted(active)
        requested = {
            aid: action.link_action
            for aid, action in actions.items()
            if aid in active and action.link_action is not None
        }
        coalition_pairs = {
            tuple(sorted((u, v)))
            for coalition in self.current_coalitions
            for i, u in enumerate(coalition)
            for v in coalition[i + 1 :]
        }

        next_graph = nx.Graph()
        next_graph.add_nodes_from(self.agents)
        for i, u in enumerate(ids):
            for v in ids[i + 1 :]:
                key = (u, v)
                u_request = requested.get(u)
                v_request = requested.get(v)
                u_op = u_request[0] if u_request and u_request[1] == v else None
                v_op = v_request[0] if v_request and v_request[1] == u else None
                if "remove" in (u_op, v_op):
                    continue
                if u_op == v_op == "add" or (self.graph.has_edge(u, v) and u_op == v_op == "maintain"):
                    next_graph.add_edge(u, v)
                    continue
                similarity = self.edge_similarity.get(key, 0.5)
                hist = self.pair_history.get(key, 0.0)
                utility = self._pair_utility_signal(u, v, profits)
                same_coalition = 1.0 if key in coalition_pairs else 0.0

                # Assignment only specifies P(e_ij,t+1=1)=f(z_ij,H,U,C,I_A).
                # This is a transparent MVP heuristic for f, intentionally easy
                # to replace later with a learned or research-specific function.
                score = (
                    self.config.base_link_probability
                    + self.config.similarity_weight * similarity
                    + self.config.history_weight * hist
                    + self.config.utility_weight * utility
                    + self.config.coalition_weight * same_coalition
                    - self.config.arbiter_link_cost_weight * max(0.0, self.link_cost_pressure)
                )
                p = min(0.98, max(0.02, score))
                if self.rng.random() < p:
                    next_graph.add_edge(u, v)
        self.graph = next_graph

    def _update_pair_history(self, profits: Dict[int, float]) -> None:
        for key in self.pair_history:
            u, v = key
            both_active = self.agent_state[u].active and self.agent_state[v].active
            if not both_active:
                self.pair_history[key] *= 0.9
                continue
            same_coalition = any(u in c and v in c for c in self.current_coalitions)
            target = 1.0 if same_coalition else 0.0
            self.pair_history[key] = 0.8 * self.pair_history[key] + 0.2 * target

    def _pair_utility_signal(self, u: int, v: int, profits: Dict[int, float]) -> float:
        vals = [profits.get(u, 0.0), profits.get(v, 0.0)]
        avg = mean(vals)
        return 1.0 / (1.0 + math.exp(-avg / 100.0))

    # ------------------------------------------------------------------
    # Observations and state updates
    # ------------------------------------------------------------------
    def _agent_observation(self, aid: int) -> AgentObservation:
        state = self.agent_state[aid]
        neighbors = sorted(self.graph.neighbors(aid))
        return AgentObservation(
            agent_id=aid,
            step=self.step_index,
            cost=state.cost,
            balance=state.balance,
            last_profit=state.last_profit,
            last_price=state.last_price,
            market_average_price=self.last_average_price if self.step_index > 0 else None,
            local_neighbors=tuple(sorted(neighbors)),
            active=state.active,
            candidate_coalitions=tuple(
                (index, coalition)
                for index, coalition in enumerate(self._candidate_coalitions)
                if aid in coalition
            ),
            coalition_join_vector=state.coalition_join_vector,
            expectations=state.expectations,
            last_evaluation=state.evaluations[-1] if state.evaluations else None,
        )

    def _coalition_observation(self) -> CoalitionObservation:
        return CoalitionObservation(
            step=self.step_index,
            active_agents=tuple(a for a, s in self.agent_state.items() if s.active),
            edges=tuple(sorted(self.edges)),
            recent_profits={a: s.last_profit for a, s in self.agent_state.items()},
            recent_prices={a: s.last_price for a, s in self.agent_state.items()},
        )

    def _arbiter_observation_from_current_state(self) -> ArbiterObservation:
        # Used only as a next-state placeholder for learning hooks.
        prices = {a: s.last_price for a, s in self.agent_state.items() if s.active and s.last_price is not None}
        avg = mean(prices.values()) if prices else 0.0
        return ArbiterObservation(
            step=self.step_index,
            active_agents=tuple(a for a, s in self.agent_state.items() if s.active),
            prices=prices,
            quantities={},
            total_quantity=0.0,
            average_price=avg,
            social_welfare=self.last_social_welfare,
            delta_average_price=avg - self.last_average_price,
            delta_social_welfare=0.0,
        )

    def _evaluate_agent(
        self,
        aid: int,
        observation: AgentObservation,
        prices: Dict[int, float],
        quantities: Dict[int, float],
        profits: Dict[int, float],
        bonuses: Dict[int, float],
        fines: Dict[int, float],
        reward: float,
    ) -> None:
        state = self.agent_state[aid]
        expected = observation.expectations
        profit = profits.get(aid, 0.0)
        evaluation = AgentEvaluation(
            step=self.step_index,
            price=prices.get(aid),
            quantity=quantities.get(aid, 0.0),
            profit=profit,
            coalition_bonus=bonuses.get(aid, 0.0),
            fine=fines.get(aid, 0.0),
            reward=reward,
            profit_error=None if expected.profit is None else profit - expected.profit,
            reward_error=None if expected.reward is None else reward - expected.reward,
            joined_coalition=next((c for c in self.current_coalitions if aid in c), None),
        )
        state.evaluations.append(evaluation)

        alpha = min(1.0, max(0.0, self.config.expectation_smoothing))

        def update(previous: Optional[float], realized: float) -> float:
            return realized if previous is None else (1.0 - alpha) * previous + alpha * realized

        state.expectations = AgentExpectations(
            market_average_price=update(expected.market_average_price, mean(prices.values()) if prices else 0.0),
            quantity=update(expected.quantity, evaluation.quantity),
            profit=update(expected.profit, profit),
            coalition_bonus=update(expected.coalition_bonus, evaluation.coalition_bonus),
            reward=update(expected.reward, reward),
        )
        state.expectation_history.append(state.expectations)

    def _update_balance(self, aid: int, reward: float, profit: float, price: Optional[float]) -> None:
        state = self.agent_state[aid]
        if not state.active:
            return
        state.balance += reward
        state.last_profit = profit
        state.last_price = price
        if state.balance < 0:
            state.negative_steps += 1
        else:
            state.negative_steps = 0

        if state.balance < self.config.min_allowed_balance or state.negative_steps >= self.config.max_negative_steps:
            state.active = False

    def _sanitize_agent_action(self, aid: int, action: AgentAction) -> AgentAction:
        price = min(self.config.max_price, max(self.config.min_price, float(action.price)))
        participate = bool(action.participate and self.agent_state[aid].active)
        link_action = action.link_action
        if link_action is not None:
            if (
                not isinstance(link_action, tuple)
                or len(link_action) != 2
                or link_action[0] not in {"add", "remove", "maintain"}
                or link_action[1] not in self.agent_state
                or link_action[1] == aid
            ):
                link_action = None
        return AgentAction(
            price=price,
            participate=participate,
            coalition_choice=action.coalition_choice,
            link_action=link_action,
        )

    def _sanitize_arbiter_action(self, action: ArbiterAction) -> ArbiterAction:
        if action.kind == ArbiterActionType.SANCTION:
            return ArbiterAction(
                kind=action.kind,
                target_agent=action.target_agent,
                fine=max(0.0, float(action.fine)),
            )
        if action.kind == ArbiterActionType.CHANGE_LINK_COST:
            return ArbiterAction(kind=action.kind, link_cost_delta=float(action.link_cost_delta))
        return ArbiterAction(kind=ArbiterActionType.OBSERVE)
