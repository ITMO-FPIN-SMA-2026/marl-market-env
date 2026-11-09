"""Public price/sales histories; no competitive-price reconstruction or hidden labels."""

from collections import deque
from dataclasses import dataclass
from statistics import fmean, pstdev

import numpy as np
import torch

from social_market_env.contracts import ArbiterActionKind

GLOBAL_DIM = 12
NODE_DIM = 18


@dataclass
class Frame:
    global_features: np.ndarray
    nodes: np.ndarray
    node_ids: tuple[int, ...]
    eligible: np.ndarray

    def serialise(self):
        return dict(global_features=torch.from_numpy(self.global_features.copy()),
                    nodes=torch.from_numpy(self.nodes.copy()), node_ids=list(self.node_ids),
                    eligible=torch.from_numpy(self.eligible.copy()))

    @classmethod
    def restore(cls, data):
        return cls(data['global_features'].numpy().copy(), data['nodes'].numpy().copy(),
                   tuple(data['node_ids']), data['eligible'].numpy().copy())


class PublicMemory:
    def __init__(self, market, settings):
        self.market, self.settings = market, settings
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.settings.history_window)
        self.last_step = None
        self.paid = {}
        self.last_fined = {}
        self.last_fine_step = None
        self.previous_reward = 0.0

    def update(self, obs):
        if self.last_step is not None:
            if obs.step != self.last_step + 1:
                raise ValueError('Observations must be consecutive; reset before a new episode')
            previous = obs.last_action
            if previous is not None and previous.kind == ArbiterActionKind.sanction:
                self.paid[previous.target] = self.paid.get(previous.target, 0.0) + previous.fine
                self.last_fined[previous.target] = (self.last_step, previous.fine)
                self.last_fine_step = self.last_step
        self.last_step = obs.step
        active = set(obs.active_agents)
        prices = {i: p for i, p in obs.prices.items() if i in active}
        quantities = {i: obs.quantities.get(i, 0.0) for i in prices}
        ids = tuple(sorted(prices))
        scale = self.market.p_max - self.market.p_min
        demand = self.market.demand_a
        welfare_scale = demand*demand/(2*self.market.demand_b)
        previous_prices, previous_sales = self.history[-1] if self.history else ({}, {})
        deltas = {i: (prices[i] - previous_prices.get(i, prices[i])) / scale for i in ids}
        cooldown = (self.last_fine_step is None or
                    obs.step - self.last_fine_step >= self.settings.fine_cooldown)
        vectors, eligible = [], []
        for i in ids:
            old = [p[i] for p, _ in self.history if i in p]
            lag1, lag2 = (old[-1] if old else prices[i]), (old[-2] if len(old) > 1 else prices[i])
            others = [j for j in ids if j != i]
            gaps = [abs(prices[i]-prices[j])/scale for j in others]
            joint = [[(p[i]-p[j])/scale for p, _ in self.history if i in p and j in p] for j in others]
            stable = [pstdev(d) if len(d) > 1 else 0.0 for d in joint]
            last_step, last_fine = self.last_fined.get(i, (obs.step-1000, 0.0))
            remaining = max(0.0, self.settings.fine_budget-self.paid.get(i, 0.0))
            vectors.append([
                prices[i]/scale, lag1/scale, lag2/scale, deltas[i],
                (prices[i]-fmean(old))/scale if old else 0.0,
                pstdev(old)/scale if old else 0.0, len(old)/self.settings.history_window,
                quantities[i]/demand, previous_sales.get(i, 0.0)/demand,
                last_fine/self.settings.max_fine, 1/(1+obs.step-last_step),
                remaining/self.settings.fine_budget,
                min(gaps) if gaps else 0.0, fmean(gaps) if gaps else 0.0,
                min((abs(deltas[i]-deltas[j]) for j in others), default=0.0),
                min(stable, default=0.0), fmean(stable) if stable else 0.0,
                len(others)/max(1, len(active)),
            ])
            eligible.append(cooldown and remaining+1e-8 >= self.settings.fine_amount)
        quotes = list(prices.values())
        globals_ = [
            obs.step/(obs.step+20), obs.total_quantity/demand, obs.social_welfare/welfare_scale,
            obs.average_price/scale, obs.delta_average_price/scale, obs.delta_social_welfare/welfare_scale,
            pstdev(quotes)/scale if quotes else 0.0,
            len(prices)/max(1, len(active)), len(self.history)/self.settings.history_window,
            float(cooldown), float(bool(obs.last_action and obs.last_action.kind == ArbiterActionKind.sanction)),
            len(prices)/(len(prices)+4),
        ]
        if self.settings.use_previous_reward:
            # Set by the previous transition callback, before act(t). Never r_t.
            globals_.append(self.previous_reward)
        self.history.append((dict(prices), dict(quantities)))
        def array(values, shape):
            result = np.asarray(values, dtype=np.float32).reshape(shape)
            if not np.isfinite(result).all():
                raise ValueError('Public market inputs must be finite')
            return np.clip(result, -5, 5)
        return Frame(array(globals_, (len(globals_),)), array(vectors, (len(ids), NODE_DIM)), ids,
                     np.asarray(eligible, dtype=bool))
