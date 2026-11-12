"""Native ArbiterAPI bridge. No custom environment, agents or reward function."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from social_market_env.contracts import ArbiterAPI, ArbiterAction, ArbiterActionKind

from .contracts import ArbiterObservation, public_market
from .ppo import PPOGRUArbiterV2


def to_observation(public):
    # SW*, graph, coalitions, private costs and evaluator labels are not inputs.
    return ArbiterObservation(public.step, tuple(public.active_agents), dict(public.prices),
        dict(public.quantities), public.total_quantity, public.average_price,
        public.social_welfare, public.delta_average_price, public.delta_social_welfare)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class PPOGRUArbiter(ArbiterAPI):
    def __init__(self, *, market, checkpoint_path, device="auto", evaluation="greedy", state_dir=None,
                 resume_state=True):
        self.state_dir = Path(state_dir) if state_dir else None
        self.resumed = bool(self.state_dir and (self.state_dir / "index.json").exists())
        if resume_state and self.state_dir and (self.state_dir / "progress.json").exists():
            state = json.loads((self.state_dir / "progress.json").read_text())
            checkpoint_path = self.state_dir / state["checkpoint"]
            if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != state["sha256"]:
                raise ValueError("Saved arbiter checkpoint checksum differs")
            self.resumed = True
        self.policy = PPOGRUArbiterV2(market=asdict(public_market(market)),
            checkpoint_path=checkpoint_path, device=device, evaluation=evaluation)
        self.pending = None

    @property
    def last_decision(self):
        return self.policy.last_decision

    def reset(self, *, seed, training):
        self.policy.reset(seed=seed, training=training)
        self.pending = None
        if training and self.state_dir and not self.resumed:
            self.save_stage()
            self.resumed = True

    def act(self, public):
        if self.pending is not None:
            raise RuntimeError("Deliver the previous transition before act")
        converted = to_observation(public)
        internal = self.policy.act(converted)
        action = (ArbiterAction(kind=ArbiterActionKind.sanction, target=internal.target, fine=internal.fine)
                  if internal.kind == "sanction" else ArbiterAction())
        self.pending = (converted, internal, action)
        return action

    def observe_transition(self, *, observation, action, reward, next_observation, done):
        if self.pending is None or observation.step != self.pending[0].step or action != self.pending[2]:
            raise ValueError("Native applied action or transition differs from the selected action")
        previous, internal, _ = self.pending
        self.policy.observe_transition(observation=previous, action=internal, reward=reward,
            next_observation=to_observation(next_observation) if next_observation is not None else None,
            done=done)
        self.pending = None
        if done and self.policy.training and self.state_dir:
            self.save_stage()

    def save_stage(self):
        directory = self.state_dir / "weights"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"episode_{self.policy.episodes_seen:06d}_update_{self.policy.updates:04d}.pt"
        attempt = 0
        base = path
        while path.exists():
            attempt += 1
            path = base.with_name(base.stem + f"_attempt_{attempt}" + base.suffix)
        self.policy.save_checkpoint(path)
        record = {"checkpoint": str(path.relative_to(self.state_dir)),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "episodes_seen": self.policy.episodes_seen, "updates": self.policy.updates}
        index = self.state_dir / "index.json"
        history = json.loads(index.read_text()) if index.exists() else []
        atomic_json(index, history + [record])
        atomic_json(self.state_dir / "progress.json", record)
        self.last_checkpoint = path


# Previous public name retained for restored configurations.
TeamPPOGRU = PPOGRUArbiter
