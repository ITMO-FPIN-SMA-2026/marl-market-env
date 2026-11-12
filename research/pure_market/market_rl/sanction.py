"""Small learned public-only sanction policy; NumPy inference, no oracle inputs."""

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .arbiter_scoring import ArbiterScoring
from .environment import ArbiterAction


@dataclass(frozen=True)
class SanctionSettings:
    threshold: float = 0.7
    fine: float = 100.0
    seller_budget: float = 400.0
    episode_budget: float = 800.0
    cooldown: int = 5
    min_history: int = 3
    memory: bool = True

    def __post_init__(self):
        numeric = (self.threshold, self.fine, self.seller_budget, self.episode_budget, self.cooldown, self.min_history)
        if not np.isfinite(numeric).all():
            raise ValueError("Sanction settings must be finite")
        if not 0 <= self.threshold <= 1 or not 0 < self.fine <= min(self.seller_budget, self.episode_budget):
            raise ValueError("Invalid sanction probability or financial limits")
        if self.cooldown < 1 or self.min_history < 1:
            raise ValueError("History and cooldown must be positive")


def market_scales(config):
    result = {"p_max": float(max(config.prices)), "price_range": float(max(config.prices) - min(config.prices)),
              "welfare_scale": float(config.welfare_bound), "n_agents": 4,
              "demand_model": config.demand_model, "demand_a": config.demand_a, "demand_b": config.demand_b,
              "harm_baseline": config.effective_harm_baseline}
    if config.demand_model == "logit":
        result.update(logit_value=config.logit_value, logit_temperature=config.logit_temperature)
    return result


class PublicEncoder:
    """Current public features + 3 lags and 5-round summaries, no seller-ID input."""
    def __init__(self, scales, memory=True):
        self.scales, self.memory = scales, memory
        self.history = deque(maxlen=5)

    def reset(self):
        self.history.clear()

    def encode(self, public):
        p = np.array([0. if x is None else x for x in public.prices], dtype=float)
        alive = np.array([x is not None for x in public.prices])
        old = np.array(self.history) if self.history else p[None, :]
        hi, width = self.scales["p_max"], max(1e-9, self.scales["price_range"])
        sw = max(1e-9, self.scales["welfare_scale"])
        result = []
        for i in range(4):
            others = np.delete(p, i)
            previous = old[:, i]
            row = [p[i] / hi, *(np.sort(others) / hi), public.mean_price / hi,
                   public.social_welfare / sw, public.price_change / width, public.welfare_change / (sw * .1),
                   (p[alive].std() if alive.any() else 0.) / width,
                   float(np.mean(others == p[i])), float(np.min(np.abs(others - p[i]))) / width]
            if self.memory:
                contexts = sorted((p[j] / hi, float(old[:, j].mean() / hi), (p[j] - old[-1, j]) / width)
                                  for j in range(4) if j != i)
                row += [float(previous[-min(k, len(previous))] / hi) for k in (1, 2, 3)]
                row += [float(previous.mean() / hi), float(previous.std() / width)]
                row += [value for context in contexts for value in context]
            result.append(row)
        self.history.append(p)
        return np.asarray(result, dtype=np.float32)


class LearnedArbiter:
    """Predict harmful participation, then choose a sanction under public budgets.

    The predictor is fitted by a separate trainer to counterfactual environment
    feedback. At deployment it accepts ONLY PublicObservation. Training labels
    are not part of its recurrent state or network inputs.
    """
    def __init__(self, config, settings=None):
        self.scales = market_scales(config)
        self.settings = settings or SanctionSettings()
        self.scoring = ArbiterScoring(target_welfare=max(1e-9, self.scales["welfare_scale"]))
        self.weights = []
        self.metadata = {}
        self.reset(seed=0, training=False)

    def reset(self, *, seed, training=False):
        if training:
            raise ValueError("Fit the predictor through the training pipeline; decision inputs stay public")
        self.encoder = PublicEncoder(self.scales, self.settings.memory)
        self.paid = np.zeros(4)
        self.last_fine = -self.settings.cooldown
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self.last_decision = {}
        self.last_step = None

    def predict(self, features):
        if not self.weights:
            return np.zeros(len(features))
        value = features
        for index, (weight, bias) in enumerate(self.weights):
            value = value @ weight.T + bias
            if index + 1 < len(self.weights):
                value = np.tanh(value)
        return 1 / (1 + np.exp(-np.clip(value.reshape(-1), -30, 30)))

    def __call__(self, public):
        if self.last_step is not None and public.step != self.last_step + 1:
            raise ValueError("Observations must be consecutive; reset between episodes")
        self.last_step = public.step
        features = self.encoder.encode(public)
        if not np.isfinite(features).all():
            raise ValueError("Public observation contains nonfinite data")
        self.steps += 1
        probabilities = self.predict(features)
        eligible = np.array([p is not None for p in public.prices])
        eligible &= self.paid + self.settings.fine <= self.settings.seller_budget + 1e-9
        eligible &= self.paid.sum() + self.settings.fine <= self.settings.episode_budget + 1e-9
        ready = self.steps>=self.settings.min_history and public.step-self.last_fine>=self.settings.cooldown
        self.last_decision = {"probabilities": probabilities.tolist(), "eligible": eligible.tolist()}
        if not ready or not self.weights or not self.metadata.get("trained", True) or not eligible.any():
            return ArbiterAction()
        candidates = np.flatnonzero(eligible)
        best = candidates[probabilities[candidates] == probabilities[candidates].max()]
        target = int(self.rng.choice(best))
        if probabilities[target] < self.settings.threshold:
            return ArbiterAction()
        amount = self.settings.fine
        self.paid[target] += amount
        self.last_fine = public.step
        return ArbiterAction("sanction", amount, target)

    def save(self, path):
        payload = {"schema": "public-sanction-2", "scales": self.scales, "settings": asdict(self.settings),
                   "scoring": asdict(self.scoring), "metadata": self.metadata,
                   "weights": [[w.tolist(), b.tolist()] for w, b in self.weights]}
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path, config):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema, expected = payload.get("schema"), market_scales(config)
        if schema == "public-sanction-1":
            compatible = config.demand_model == "assignment" and payload["scales"] == {
                k: expected[k] for k in ("p_max", "price_range", "welfare_scale", "n_agents")}
        else:
            if schema == "public-sanction-2" and "harm_baseline" not in payload["scales"]:
                compatible = config.effective_harm_baseline == "proposals" and payload["scales"] == {
                    k:v for k,v in expected.items() if k != "harm_baseline"}
            else:
                compatible = schema == "public-sanction-2" and payload["scales"] == expected
        if not compatible:
            raise ValueError("Sanction checkpoint has an incompatible schema or public market scales")
        settings = dict(payload["settings"])
        experimental = [settings.pop(key, 0) for key in ("minimum_effect", "fine_rate")]
        if any(experimental) or payload.get("effect_weights"):
            raise ValueError("Experimental causal/fiscal policy requires the implementation in runs/experimental_arbiter_archive")
        policy = cls(config, SanctionSettings(**settings))
        policy.weights = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in payload["weights"]]
        width = 25 if policy.settings.memory else 11
        for weight, bias in policy.weights:
            if weight.ndim != 2 or bias.ndim != 1 or weight.shape != (len(bias), width):
                raise ValueError("Checkpoint has an invalid network shape")
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError("Checkpoint weights must be finite")
            width = len(bias)
        if policy.weights and width != 1:
            raise ValueError("Checkpoint needs one probability per seller")
        policy.metadata = payload["metadata"]
        policy.scoring = ArbiterScoring(**payload["scoring"])
        return policy
