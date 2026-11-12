"""Explicit warm start: copy both networks; discard old-domain optimizer/data."""

from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from .contracts import public_market
from .policy import PPOGRUArbiterV2


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def convert_checkpoint(source, destination, market, reference_path):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 3 or not payload["settings"]["use_recurrence"]:
        raise ValueError("Expected the trained custom-market PPO/GRU schema-3 checkpoint")
    policy = PPOGRUArbiterV2(market=asdict(public_market(market)), settings=payload["settings"],
        reference_path=reference_path, model_seed=payload["model_seed"], device="cpu")
    policy.actor.load_state_dict(payload["actor"], strict=True)
    policy.critic.load_state_dict(payload["critic"], strict=True)
    policy.metadata = {"actor_pretrained": True, "starting_weights": "own_trained_actor_and_critic",
        "source_checkpoint": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_market": payload["market"], "source_ppo_updates": payload["updates"],
        "source_episodes_seen": payload["episodes_seen"],
        "optimizer_reset": True, "old_rollouts_discarded": True,
        "reward_source": "native social_market_env.core.scoring.arbiter_reward",
        "target_environment_qualified": False}
    for name in ("actor", "critic"):
        if state_digest(payload[name]) != state_digest(getattr(policy, name).state_dict()):
            raise RuntimeError("Network weights changed during conversion")
    if policy.actor_optimizer.state or policy.critic_optimizer.state or policy.completed:
        raise RuntimeError("Old-domain training state must not be carried into the target")
    policy.save_checkpoint(destination)
    return {"source": str(source), "destination": str(destination),
            "actor_sha256": state_digest(policy.actor.state_dict()),
            "critic_sha256": state_digest(policy.critic.state_dict()),
            "target_market": asdict(policy.market), "optimizer_reset": True}
