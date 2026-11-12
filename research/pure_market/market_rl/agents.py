"""Four independent tabular RL policies; no hand-written trading strategies."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LearnerConfig:
    algorithm: str
    learning_rate: float = 0.15
    gamma: float = 0.95
    epsilon: float = 0.25


DEFAULT_LEARNERS = (
    LearnerConfig("q_learning"),
    LearnerConfig("sarsa", learning_rate=0.12, gamma=0.95),
    LearnerConfig("expected_sarsa", learning_rate=0.10),
    LearnerConfig("double_q", learning_rate=0.10),
)


def encode_state(observation, prices):
    """Finite local observation bins, shared across algorithms, not action rules."""
    public = [p for p in observation.public_prices if p is not None]
    mean = np.mean(public) if public else np.mean(prices)
    return (
        int(np.digitize(observation.balance, [0.0, 100.0, 300.0])),
        int(np.digitize(observation.last_profit, [0.0, 10.0, 40.0])),
        sum(1 << i for i in observation.neighbors),
        len(observation.coalition),
        int(np.argmin(np.abs(np.array(prices) - mean))),
        int(observation.last_fine > 0),
    )


class TabularAgent:
    def __init__(self, config, n_actions, seed=0):
        if config.algorithm not in {"q_learning", "sarsa", "expected_sarsa", "double_q"}:
            raise ValueError("Unknown learning algorithm")
        if not 0 < config.learning_rate <= 1 or not 0 <= config.gamma <= 1 or not 0 <= config.epsilon <= 1:
            raise ValueError("Invalid learning parameters")
        self.config = config
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.q = {}
        self.q2 = {}
        self.updates = 0

    def values(self, state, table=None):
        if table is not None:
            return table.get(state, np.zeros(self.n_actions))
        values = self.q.get(state, np.zeros(self.n_actions))
        if self.config.algorithm == "double_q":
            values = values + self.q2.get(state, np.zeros(self.n_actions))
        return values

    def probabilities(self, state, mask, epsilon):
        allowed = np.flatnonzero(mask)
        if not len(allowed):
            raise ValueError("At least one action must be available")
        values = self.values(state)
        greedy = allowed[values[allowed] == values[allowed].max()]
        probabilities = np.zeros(self.n_actions)
        probabilities[allowed] = epsilon / len(allowed)
        probabilities[greedy] += (1 - epsilon) / len(greedy)
        return probabilities

    def act(self, state, mask, epsilon=None):
        epsilon = self.config.epsilon if epsilon is None else epsilon
        return int(self.rng.choice(self.n_actions, p=self.probabilities(state, mask, epsilon)))

    def state_value(self, state, mask):
        scale = 0.5 if self.config.algorithm == "double_q" else 1.0
        return float(self.values(state)[mask].max() * scale * 100.0)

    def learn(self, state, action, reward, next_state, next_mask, done, next_action=None, epsilon=None):
        """One TD update. SARSA consumes the *actual* next behavior action."""
        algorithm = self.config.algorithm
        epsilon = self.config.epsilon if epsilon is None else epsilon
        allowed = np.flatnonzero(next_mask)
        table = self.q
        bootstrap = 0.0
        if algorithm == "double_q":
            table, other = (self.q, self.q2) if self.rng.random() < 0.5 else (self.q2, self.q)
            if not done:
                values = self.values(next_state, table)
                greedy = allowed[values[allowed] == values[allowed].max()]
                best = int(self.rng.choice(greedy))
                bootstrap = self.values(next_state, other)[best]
        elif not done:
            values = self.values(next_state)
            if algorithm == "q_learning":
                bootstrap = values[allowed].max()
            elif algorithm == "sarsa":
                if next_action is None or not next_mask[next_action]:
                    raise ValueError("SARSA needs the actual valid next action")
                bootstrap = values[next_action]
            else:
                bootstrap = self.probabilities(next_state, next_mask, epsilon) @ values
        row = table.setdefault(state, np.zeros(self.n_actions))
        target = reward + self.config.gamma * bootstrap
        row[action] += self.config.learning_rate * (target - row[action])
        self.updates += 1


def make_agents(n_actions, seed=0):
    return [TabularAgent(config, n_actions, seed + 100 + i) for i, config in enumerate(DEFAULT_LEARNERS)]


def _serialize(agents):
    records = []
    for agent in agents:
        records.append({
            "config": asdict(agent.config), "n_actions": agent.n_actions,
            "q": [[list(state), values.tolist()] for state, values in agent.q.items()],
            "q2": [[list(state), values.tolist()] for state, values in agent.q2.items()],
            "updates": agent.updates, "rng": agent.rng.bit_generator.state,
        })
    return records


def save_agents(path, agents, market_config, coalition_policy=None, solo_agents=None):
    data = {"format": 2, "market": asdict(market_config), "agents": _serialize(agents),
            "coalition_policy": coalition_policy.to_dict() if coalition_policy else None,
            "solo_agents": _serialize(solo_agents) if solo_agents else None}
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def _deserialize(records, migrate_fines=False):
    agents = []
    for stored in records:
        agent = TabularAgent(LearnerConfig(**stored["config"]), stored["n_actions"])
        def state(s):
            return tuple(s) + (0,) if migrate_fines and len(s) == 5 else tuple(s)

        agent.q = {state(s): np.array(v, dtype=float) for s, v in stored["q"]}
        agent.q2 = {state(s): np.array(v, dtype=float) for s, v in stored["q2"]}
        agent.updates = stored["updates"]
        agent.rng.bit_generator.state = stored["rng"]
        agents.append(agent)
    return agents


def load_agents(path):
    from .environment import MarketConfig

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["format"] not in (1, 2) or len(data["agents"]) != 4:
        raise ValueError("Unsupported checkpoint")
    market = data["market"]
    if market.get("demand_model") == "logit" and "harm_baseline" not in market:
        market["harm_baseline"] = "proposals"  # preserve labels of the first logit study
    for key in ("costs", "balances", "prices"):
        market[key] = tuple(market[key])
    config = MarketConfig(**market)
    agents = _deserialize(data["agents"], migrate_fines=data["format"] == 1)
    for agent in agents:
        if agent.n_actions != len(config.prices) * 5:
            raise ValueError("Checkpoint action space does not match market")
    return config, agents


def load_auxiliary(path):
    from .coalition_policy import CoalitionPolicy

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    policy = CoalitionPolicy.from_dict(data["coalition_policy"]) if data.get("coalition_policy") else None
    solo = _deserialize(data["solo_agents"], migrate_fines=data["format"] == 1) if data.get("solo_agents") else None
    return policy, solo
