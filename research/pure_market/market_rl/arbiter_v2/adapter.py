"""Public-only conversion and lifecycle bridge to the four-agent market."""

from dataclasses import asdict

import numpy as np

from ..environment import ArbiterAction as MarketAction
from .contracts import ArbiterObservation, public_market


def to_observation(public, market):
    """Recover sales from published prices and the *public* clearing formula.

    No graph, costs, balances, coalition labels or environment instance needed.
    SW and its change retain this market's total-welfare definition. Training
    starts from random weights, so no old feature-scale assumptions are reused.
    """
    prices = {i: float(p) for i, p in enumerate(public.prices) if p is not None}
    ids = tuple(prices)
    p = np.array(list(prices.values()))
    if not ids:
        weights = np.zeros(0)
    elif market.share_model == "logit_outside":
        z = -p / market.logit_temperature
        weights = np.exp(z - z.max())
        weights /= weights.sum()
    else:
        weights = np.ones(1) if len(ids) == 1 else (1 - p / p.sum()) / (len(ids) - 1)
    quantities = {i: float(public.total_quantity * weights[k]) for k, i in enumerate(ids)}
    return ArbiterObservation(
        public.step, ids, prices, quantities, public.total_quantity, public.mean_price,
        public.social_welfare, public.price_change, public.welfare_change,
    )


class ArbiterAdapter:
    def __init__(self, policy, market_config, scoring=None):
        from ..arbiter_scoring import ArbiterScoring

        if asdict(public_market(market_config)) != asdict(policy.market):
            raise ValueError("Arbiter checkpoint and market have different public scales")
        self.policy = policy
        self.market = policy.market
        self.scoring = scoring or ArbiterScoring(target_welfare=market_config.welfare_bound)
        self._pending = None

    def reset(self, *, seed, training):
        self.policy.reset(seed=seed, training=training)
        self._pending = None

    def __call__(self, public):
        if self._pending is not None:
            raise RuntimeError("Deliver the previous transition before the next decision")
        observation = to_observation(public, self.market)
        original = self.policy.act(observation)
        action = MarketAction("sanction", original.fine, original.target) if original.kind == "sanction" else MarketAction()
        self._pending = (observation, original, action)
        return action

    def observe_transition(self, *, observation, action, reward, next_observation, done):
        if self._pending is None or self._pending[0].step != observation.step or action != self._pending[2]:
            raise ValueError("Transition does not match the applied market action")
        previous, original, _ = self._pending
        next_public = to_observation(next_observation, self.market) if next_observation else None
        self.policy.observe_transition(observation=previous, action=original, reward=reward,
                                       next_observation=next_public, done=done)
        self._pending = None

    def save_checkpoint(self, path):
        self.policy.metadata["reward"] = asdict(self.scoring)
        return self.policy.save_checkpoint(path)

    @classmethod
    def load(cls, path, market_config, **kwargs):
        from ..arbiter_scoring import ArbiterScoring
        from .policy import PPOGRUArbiterV2

        policy = PPOGRUArbiterV2(checkpoint_path=path, market=asdict(public_market(market_config)), **kwargs)
        return cls(policy, market_config, ArbiterScoring(**policy.metadata.get("reward", {})))
