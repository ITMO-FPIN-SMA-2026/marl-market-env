"""Public market memory and symmetric pair evidence, with explicit missingness."""

from collections import deque
from dataclasses import dataclass
from statistics import fmean, pstdev

import numpy as np

from social_market_env.contracts import ArbiterActionKind
from social_market_env.core.market import consumer_surplus

GLOBAL_DIM = 14
NODE_DIM = 24
PAIR_DIM = 6


@dataclass
class Frame:
    global_features: np.ndarray
    nodes: np.ndarray
    pairs: np.ndarray
    node_ids: tuple[int, ...]
    eligible: np.ndarray
    risks: np.ndarray


class PublicMemory:
    def __init__(self, market, settings, reference):
        self.market, self.settings, self.reference = market, settings, reference
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.settings.history_window)
        self.last_step = None
        self.paid = {}
        self.last_fined = {}
        self.last_fine_step = None

    def record_applied(self, action, step):
        if action.kind == ArbiterActionKind.sanction:
            self.paid[action.target] = self.paid.get(action.target, 0.0)+action.fine
            self.last_fined[action.target] = step
            self.last_fine_step = step

    def update(self, obs):
        if type(obs.step) is not int or obs.step < 0:
            raise ValueError('Observation step must be a nonnegative integer')
        if self.last_step is not None and obs.step != self.last_step+1:
            raise ValueError('Observations must be consecutive; reset between episodes')
        self.last_step = obs.step
        active = set(obs.active_agents)
        prices = {i: float(p) for i, p in obs.prices.items() if i in active}
        if any(not np.isfinite(q) or q < 0 for q in obs.quantities.values()):
            raise ValueError('Sales must be finite and nonnegative')
        ids = tuple(sorted(prices))
        for i, p in prices.items():
            if not np.isfinite(p) or not self.market.p_min <= p <= self.market.p_max:
                raise ValueError('Public prices must be finite and inside market bounds')
        scale = self.market.p_max-self.market.p_min
        norm = lambda p: (p-self.market.p_min)/scale
        demand = self.market.demand_a
        welfare_scale = consumer_surplus(total_quantity=self.market.demand_a, market=self.market)
        previous = self.history[-1] if self.history else {}
        # A gap in participation resets that seller's contiguous observation history.
        sequences = {}
        for i in ids:
            old = []
            for frame in reversed(self.history):
                if i not in frame:
                    break
                old.append(frame[i][0])
            sequences[i] = (list(reversed(old))+[prices[i]])[-self.settings.history_window:]
        n = len(ids)
        pairs = np.zeros((n, n, PAIR_DIM), dtype=np.float32)
        risks = np.zeros(n, dtype=np.float32)
        levels = {i: self.reference.level(prices[i]) for i in ids} if self.reference else dict.fromkeys(ids, 0.0)
        for a, i in enumerate(ids):
            for b in range(a+1, n):
                j = ids[b]
                size = min(len(sequences[i]), len(sequences[j]))
                pi, pj = np.array(sequences[i][-size:]), np.array(sequences[j][-size:])
                gaps = (pi-pj)/scale
                close = float(np.exp(-abs(gaps).mean()/.05))
                sync = float(np.exp(-np.abs(np.diff(pi)-np.diff(pj)).mean()/(scale*.05))) if size > 1 else 0.0
                enough = size >= self.settings.min_history
                level = min(levels[i], levels[j])
                risk = (.5*level+.25*sync+.25*close) if enough and level > 0 else 0.0
                pair = [abs(gaps[-1]), float(gaps.std()) if size > 1 else 0.0,
                        sync, close, min(1, size/self.settings.history_window), risk]
                pairs[a, b] = pairs[b, a] = pair
                risks[a], risks[b] = max(risks[a], risk), max(risks[b], risk)
        cooldown = self.last_fine_step is None or obs.step-self.last_fine_step >= self.settings.cooldown
        episode_room = self.settings.episode_budget-sum(self.paid.values())
        vectors, eligible = [], []
        for k, i in enumerate(ids):
            seq = sequences[i]
            prior = seq[:-1]
            lag = [prior[-j] if len(prior) >= j else prices[i] for j in (1, 2, 3)]
            room = self.settings.seller_budget-self.paid.get(i, 0)
            previous_quantity = previous.get(i, (prices[i], 0))[1]
            last_fine = self.last_fined.get(i)
            vectors.append([
                norm(prices[i]), *(norm(p) for p in lag),
                (prices[i]-lag[0])/scale,
                (prices[i]-fmean(seq[-5:]))/scale,
                pstdev(seq[-5:])/scale, pstdev(seq)/scale,
                min(1, len(seq)/self.settings.history_window), float(len(seq) > 1),
                float(len(prior) >= 3), float(len(seq) >= self.settings.min_history),
                obs.quantities.get(i, 0)/demand, previous_quantity/demand,
                self.paid.get(i, 0)/self.settings.seller_budget, max(0, room)/self.settings.seller_budget,
                0 if last_fine is None else 1/(1+obs.step-last_fine),
                levels[i], float(risks[k]), float(n > 1),
                (prices[i]-fmean(prices.values()))/scale,
                float(i in previous), len(prior)/self.settings.history_window,
                max(0, episode_room)/self.settings.episode_budget,
            ])
            screened = bool(self.reference and risks[k] > (
                self.reference.risk_threshold if self.settings.screening_mode == 'calibrated' else 0.0))
            if self.settings.screening_mode == 'learned':
                # A coarse price grid can put the reference's 95th percentile
                # at p_max. Keep the risk as an input, let PPO learn abstention;
                # history, cooldown and cash budgets still mask both rollouts
                # and PPO updates in exactly the same way.
                screened = self.reference is not None
            eligible.append(bool(screened and len(seq) >= self.settings.min_history and cooldown
                and min(room, episode_room)+1e-8 >= self.settings.fine_amount))
        global_features = [
            obs.step/(obs.step+20), obs.total_quantity/demand, obs.social_welfare/welfare_scale,
            norm(obs.average_price) if prices else 0, obs.delta_average_price/scale,
            obs.delta_social_welfare/welfare_scale, pstdev(prices.values())/scale if prices else 0,
            n/max(1, len(active)), n/(n+4), len(self.history)/self.settings.history_window,
            float(cooldown), float(self.last_fine_step == obs.step-1),
            float(risks.max()) if n else 0, float(bool(self.reference)),
        ]
        def array(values, shape):
            result = np.asarray(values, dtype=np.float32).reshape(shape)
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite public market data')
            return np.clip(result, -5, 5)
        self.history.append({i: (prices[i], float(obs.quantities.get(i, 0))) for i in ids})
        return Frame(array(global_features, (GLOBAL_DIM,)), array(vectors, (n, NODE_DIM)),
                     pairs, ids, np.asarray(eligible, dtype=bool), risks)
