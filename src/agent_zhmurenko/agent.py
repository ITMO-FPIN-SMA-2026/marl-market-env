"""Tabular Q-learning with confidence-gated market-regime specialization.

No neural network, market-profit oracle, or opponent-private state is used.
The shared table learns current contexts. Small direction-specific tables may
override it only after the corresponding action has actually been sampled.
"""

from __future__ import annotations

import json
import math
import random
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from social_market_env.config import MarketConfig
from social_market_env.contracts import CONTRACTS_VERSION, AgentAction, AgentObservation, MarketAgentAPI

PACKAGE = Path(__file__).resolve().parent
MARKUPS = (0.5, 0.8, 1.0, 1.1, 1.2, 1.6, 2.5, 4.0)
ACTIONS = tuple(f'markup_{x:g}' for x in MARKUPS) + ('below_rivals', 'above_rivals', 'join', 'wait')
JOIN, WAIT = 10, 11
STATE_SIZES = (7, 8, 3)
FORMAT = 3


def output_path(value: str | Path) -> Path:
    """Runtime files may live in our package or outside the repository."""
    path = Path(value).expanduser().resolve()
    repository = next((p for p in PACKAGE.parents if (p / '.git').exists()), None)
    if repository and path.is_relative_to(repository) and not path.is_relative_to(PACKAGE):
        raise ValueError(f'Output inside the repository must stay within {PACKAGE}')
    return path


@dataclass(frozen=True)
class QSettings:
    learning_rate: float = 0.12
    gamma: float = 0.9
    epsilon_start: float = 0.35
    epsilon_end: float = 0.08
    epsilon_decay: float = 0.9999
    reward_scale: float = 100.0
    reserve_reference: float = 1000.0
    use_regime_context: bool = True
    context_min_visits: int = 8
    context_prior_visits: float = 64.0
    trend_threshold: float = 0.1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name not in ('use_regime_context', 'context_min_visits') and (
                type(value) not in (int, float) or not math.isfinite(value)
            ):
                raise ValueError(f'{name} must be finite')
        if type(self.use_regime_context) is not bool or type(self.context_min_visits) is not int or self.context_min_visits < 1:
            raise ValueError('Invalid regime settings')
        if not 0 < self.learning_rate <= 1 or not 0 <= self.gamma < 1:
            raise ValueError('Invalid learning rate or discount')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or not 0 < self.epsilon_decay <= 1:
            raise ValueError('Invalid exploration schedule')
        if min(self.reward_scale, self.reserve_reference, self.trend_threshold, self.context_prior_visits) <= 0:
            raise ValueError('Scales and threshold must be positive')


class ZhmurenkoVladimirAgent(MarketAgentAPI):
    """Use bundled trained JSON by default; ``fresh=True`` explicitly opts out.

    The base table has at most 7*8*3=168 rows. The optional specialization has
    at most 3*168=504 rows. Each row contains 12 action values and visit counts.
    Links are left to the unchanged environment and its existing policies.
    """

    def __init__(self, *, market: dict | None = None, settings: dict | None = None,
                 model_path: str | Path | None = None, fresh: bool = False,
                 checkpoint_path: str | None = None):
        if type(fresh) is not bool or fresh and model_path is not None:
            raise ValueError('fresh=True and model_path are mutually exclusive')
        self.market = MarketConfig(**(market or {}))
        values = asdict(self.market)
        if not all(type(v) in (int, float) and math.isfinite(v) for k, v in values.items() if k != 'share_model'):
            raise ValueError('Invalid market numbers')
        if not 0 < self.market.p_min < self.market.p_max or self.market.tau <= 0 or self.market.demand_b <= 0:
            raise ValueError('Invalid market scales')
        if self.market.share_model not in ('logit', 'tz_linear'):
            raise ValueError('Unknown share model')
        self.settings = QSettings(**(settings or {}))
        self.q: dict[tuple[int, ...], list[float]] = {}
        self.visits: dict[tuple[int, ...], list[int]] = {}
        self.context_q: dict[tuple[int, ...], list[float]] = {}
        self.context_visits: dict[tuple[int, ...], list[int]] = {}
        self.updates = 0
        self.metadata: dict = {}
        self.rng = random.Random(0)
        self.training = False
        self.checkpoint_path = checkpoint_path
        self._pending = None
        self._step = -1
        self._median = None
        self._direction = 1
        self.stats = {}
        if not fresh:
            path = Path(model_path).expanduser() if model_path is not None else PACKAGE / 'models' / 'q_agent.json'
            if not path.is_file():
                raise FileNotFoundError(f'Trained Q-table missing: {path}. Train it first; fresh=True is the explicit untrained mode.')
            self._load(path, market_supplied=market is not None, settings_supplied=settings is not None)

    @property
    def epsilon(self):
        s = self.settings
        return max(s.epsilon_end, s.epsilon_start * s.epsilon_decay ** self.updates) if self.training else 0.0

    def reset(self, *, agent_id: int, seed: int, training: bool):
        self.training = training
        self.rng.seed(seed)
        self._pending = None
        self._step, self._median, self._direction = -1, None, 1
        self._agent_id = agent_id
        self.decision_counts = {}
        self.stats = dict(decisions=0, known_states=0, learned_decisions=0, specialized_decisions=0,
                          differs_from_cold=0, regime_changed_decisions=0, exploratory=0, offered=0, joined=0,
                          high_weight_offers=0, high_weight_joins=0, low_weight_offers=0, low_weight_joins=0)

    @staticmethod
    def _rival_median(o):
        prices = [p for i, p in o.market.last_prices.items() if i != o.agent_id and math.isfinite(p)]
        return median(prices) if prices else None

    def state(self, o: AgentObservation, *, commit: bool = False):
        """Preview is pure; only act commits past public observations."""
        reference = self._rival_median(o)
        pressure = 0 if reference is None else 1 + bisect_right(
            (-0.5, 0.5, 1.5, 3.0, 6.0), (reference - o.cost) / self.market.tau)
        offer = o.offer
        if offer is None:
            offer_state = 0
        elif o.restricted or not self.market.p_min <= offer.recommended_price <= self.market.p_max or offer.recommended_price < o.cost:
            offer_state = 7
        else:
            premium = (offer.recommended_price - o.cost) / self.market.tau
            price_band = bisect_right((1.5, 4.0), premium)
            favored = int(offer.payout_weight * len(offer.members) >= 1.0)
            offer_state = 1 + 2 * price_band + favored
        reserve = 2 if o.restricted or o.balance < 0 else int(o.balance < self.settings.reserve_reference * 0.25)
        direction = 1
        if o.step == self._step:
            direction = self._direction
        elif o.step == self._step + 1 and reference is not None and self._median is not None:
            change = (reference - self._median) / self.market.tau
            direction = 0 if change < -self.settings.trend_threshold else 2 if change > self.settings.trend_threshold else 1
        if commit:
            if o.step < self._step:
                raise ValueError('Cannot commit an earlier market observation')
            self._step, self._median, self._direction = o.step, reference, direction
        base = (pressure, offer_state, reserve)
        return base, (*base, direction)

    def available_actions(self, o):
        floor = max(self.market.p_min, o.cost)
        clip = lambda p: min(self.market.p_max, max(floor, p))
        wait = AgentAction(price=min(self.market.p_max, floor), participate=False)
        if not o.active or floor > self.market.p_max:
            return {WAIT: wait}
        actions = {i: AgentAction(price=clip(o.cost + m * self.market.tau)) for i, m in enumerate(MARKUPS)}
        reference = self._rival_median(o)
        if reference is not None:
            actions[8] = AgentAction(price=clip(reference - 0.2 * self.market.tau))
            actions[9] = AgentAction(price=clip(reference + 0.2 * self.market.tau))
        offer = o.offer
        if (offer is not None and not o.restricted and floor <= offer.recommended_price <= self.market.p_max
                and math.isfinite(offer.recommended_price) and offer.payout_weight > 0):
            actions[JOIN] = AgentAction(price=offer.recommended_price, coalition_choice=offer.plan_id)
        actions[WAIT] = wait
        return actions

    def _values(self, base, context, available=None):
        shared = self.q.get(base, [0.0] * len(ACTIONS))
        if not self.settings.use_regime_context or context not in self.context_q:
            return shared, False
        values = self.context_q[context]
        counts = self.context_visits[context]
        if available is None:
            available = list(range(8)) + ([8,9] if base[0] else []) + ([JOIN] if base[1] not in (0,7) else []) + [WAIT]
        # One confidence weight for the whole legal row preserves invariance to
        # action-independent value offsets. Per-action weights can break it.
        support = min(counts[i] for i in available)
        if support < self.settings.context_min_visits:
            return shared, False
        weight = support / (support + self.settings.context_prior_visits)
        return [common + weight * (local-common) for local,common in zip(values,shared)], True

    @staticmethod
    def _cold_action(actions):
        return 2 if 2 in actions else WAIT

    def act(self, observation):
        base, context = self.state(observation, commit=True)
        trace_key = '|'.join(map(str, context))
        self.decision_counts[trace_key] = self.decision_counts.get(trace_key, 0) + 1
        actions = self.available_actions(observation)
        values, specialized = self._values(base, context, actions)
        explore = self.training and self.rng.random() < self.epsilon
        if explore:
            index = self.rng.choice(list(actions))
        elif base not in self.q:
            index = self._cold_action(actions)
        else:
            # A fixed tie order makes frozen evaluation deterministic.
            index = max(actions, key=lambda i: (values[i], i == self._cold_action(actions), -i))
        if self.training:
            self._pending = (observation.step, base, context, index, actions[index])
        self.stats['decisions'] += 1
        self.stats['known_states'] += int(base in self.q)
        self.stats['learned_decisions'] += int(self.visits.get(base, [0] * len(ACTIONS))[index] > 0)
        self.stats['specialized_decisions'] += int(specialized and self.context_visits[context][index] >= self.settings.context_min_visits)
        def differs(first, second):
            return (first.participate != second.participate or first.coalition_choice != second.coalition_choice
                    or abs(first.price-second.price) > 1e-7)
        self.stats['differs_from_cold'] += int(differs(actions[index], actions[self._cold_action(actions)]))
        shared = self.q.get(base, [0.0] * len(ACTIONS))
        shared_index = max(actions, key=lambda i: (shared[i], i == self._cold_action(actions), -i)) if base in self.q else self._cold_action(actions)
        local_index = max(actions, key=lambda i: (values[i], i == self._cold_action(actions), -i)) if base in self.q else self._cold_action(actions)
        self.stats['regime_changed_decisions'] += int(differs(actions[local_index], actions[shared_index]))
        self.stats['exploratory'] += int(explore)
        self.stats['offered'] += int(JOIN in actions)
        self.stats['joined'] += int(index == JOIN)
        if JOIN in actions:
            ratio = observation.offer.payout_weight * len(observation.offer.members)
            for label, present in (('high', ratio > 1.25), ('low', ratio < .75)):
                self.stats[label+'_weight_offers'] += int(present)
                self.stats[label+'_weight_joins'] += int(present and index == JOIN)
        return actions[index]

    def observe_transition(self, *, observation, action, reward, next_observation, done):
        if not self.training or self._pending is None:
            return
        step, base, context, index, requested = self._pending
        self._pending = None
        if step != observation.step or action != requested:
            raise ValueError('Cannot attribute a changed action to the chosen Q entry')
        if not math.isfinite(reward):
            raise ValueError('Non-finite market reward')
        terminal = done or next_observation is None or not next_observation.active
        immediate = reward / self.settings.reward_scale
        shared_next = specialized_next = 0.0
        if not terminal:
            next_base, next_context = self.state(next_observation)
            available = self.available_actions(next_observation)
            shared = self.q.get(next_base, [0.0] * len(ACTIONS))
            contextual, _ = self._values(next_base, next_context, available)
            shared_next = max(shared[i] for i in available)
            specialized_next = max(contextual[i] for i in available)
        tables = [(self.q, self.visits, base, shared_next)]
        if self.settings.use_regime_context:
            tables.append((self.context_q, self.context_visits, context, specialized_next))
        errors = []
        for values, counts, key, future in tables:
            initial = self.q[base].copy() if values is self.context_q else [0.0] * len(ACTIONS)
            row = values.setdefault(key, initial)
            visits = counts.setdefault(key, [0] * len(ACTIONS))
            error = immediate + self.settings.gamma * future - row[index]
            row[index] += self.settings.learning_rate * error
            visits[index] += 1
            errors.append(error)
        self.updates += 1
        self.last_learning = dict(td_error=errors[-1], action=index, reward=reward, epsilon=self.epsilon)
        if terminal and self.checkpoint_path:
            self.save(self.checkpoint_path.format(agent_id=self._agent_id))

    def save(self, path):
        if self._pending is not None:
            raise ValueError('Save only after the pending transition has been delivered')
        path = output_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        def rows(values, counts):
            return [dict(state=list(key), q=values[key], visits=counts[key]) for key in sorted(values)]
        payload = dict(format_version=FORMAT, contracts_version=CONTRACTS_VERSION, actions=list(ACTIONS),
                       state_sizes=list(STATE_SIZES), market=asdict(self.market), settings=asdict(self.settings),
                       updates=self.updates, metadata=self.metadata, shared=rows(self.q, self.visits),
                       regimes=rows(self.context_q, self.context_visits))
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)

    def _load(self, path, *, market_supplied, settings_supplied):
        payload = json.loads(path.read_text(encoding='utf-8'))
        if (payload.get('format_version') != FORMAT or payload.get('contracts_version') != CONTRACTS_VERSION
                or payload.get('actions') != list(ACTIONS) or payload.get('state_sizes') != list(STATE_SIZES)):
            raise ValueError('Incompatible Q-table schema')
        market = MarketConfig(**payload['market'])
        settings = QSettings(**payload['settings'])
        if market_supplied and market != self.market or settings_supplied and settings != self.settings:
            raise ValueError('Checkpoint market/settings differ from the request')
        self.market, self.settings = market, settings
        if type(payload['updates']) is not int or payload['updates'] <= 0 or not payload['shared']:
            raise ValueError('This is not a trained Q-table; use fresh=True for the untrained control')
        if not isinstance(payload['metadata'], dict):
            raise ValueError('Invalid checkpoint metadata')
        for name, values, counts, sizes in (
            ('shared', self.q, self.visits, STATE_SIZES),
            ('regimes', self.context_q, self.context_visits, (*STATE_SIZES, 3)),
        ):
            for row in payload[name]:
                key = tuple(row['state'])
                if len(key) != len(sizes) or any(type(x) is not int or not 0 <= x < limit for x, limit in zip(key, sizes)) or key in values:
                    raise ValueError('Invalid or duplicate Q state')
                q, visits = row['q'], row['visits']
                if (len(q) != len(ACTIONS) or any(type(x) not in (int, float) or not math.isfinite(x) for x in q)
                        or len(visits) != len(ACTIONS) or any(type(n) is not int or n < 0 for n in visits)):
                    raise ValueError('Invalid Q row')
                values[key], counts[key] = list(q), list(visits)
        if sum(sum(n) for n in self.visits.values()) != payload['updates']:
            raise ValueError('Q visits do not match the update counter')
        if not settings.use_regime_context and self.context_q:
            raise ValueError('Disabled regime mechanism has nonempty specialization')
        if settings.use_regime_context and sum(sum(n) for n in self.context_visits.values()) != payload['updates']:
            raise ValueError('Regime visits do not match the update counter')
        self.updates, self.metadata = payload['updates'], payload['metadata']
