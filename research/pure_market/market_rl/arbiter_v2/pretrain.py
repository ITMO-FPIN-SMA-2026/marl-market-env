"""Train the ORIGINAL actor on public sequences and environment training labels."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..agents import load_agents, load_auxiliary
from ..arbiter_scoring import harmful_groups
from ..environment import ArbiterAction as MarketAction, MarketEnv
from ..learn_arbiter import reseed, write_json
from ..train import run_episode
from .adapter import to_observation
from .contracts import ArbiterAction, ArbiterActionKind, public_market
from .features import PublicMemory
from .model import collate, greedy_choice
from .policy import cpu_tree


class PublicExplorer:
    """Training-data exploration only; never uses guilt labels to act."""
    def __init__(self, config, settings, reference, fine_probability):
        self.market, self.settings, self.reference = public_market(config), settings, reference
        self.fine_probability = fine_probability

    def reset(self, *, seed, training=False):
        self.memory = PublicMemory(self.market, self.settings, self.reference)
        self.rng = np.random.default_rng(seed)
        self.frame = None

    def __call__(self, public):
        self.frame = self.memory.update(to_observation(public, self.market))
        eligible = np.flatnonzero(self.frame.eligible)
        if not len(eligible) or self.rng.random() >= self.fine_probability:
            return MarketAction()
        target = self.frame.node_ids[int(self.rng.choice(eligible))]
        action = ArbiterAction(ArbiterActionKind.sanction, target, self.settings.fine_amount)
        self.memory.record_applied(action, public.step)
        return MarketAction("sanction", action.fine, target)


def collect_dataset(config, sources, settings, reference, *, worlds, seed, path):
    """Private labels are separate targets, never encoded into public frames."""
    populations = []
    for source in sources:
        source_config, agents = load_agents(source)
        if source_config != config:
            raise ValueError("Pretraining sources must use exactly the same market")
        coalition, solo = load_auxiliary(source)
        if coalition is None or solo is None:
            raise ValueError("Pretraining needs main, coalition and solo policies")
        populations.append((agents, coalition, solo))
    episodes, targets = [], []
    source_indices = np.random.default_rng(np.random.SeedSequence([seed, 1024])).integers(
        len(populations), size=worlds)
    for world in range(worlds):
        agents, coalition, solo = populations[int(source_indices[world])]
        independent = world % 5 == 0
        population = solo if independent else agents
        reseed(population, None if independent else coalition, seed + world)
        explorer = PublicExplorer(config, settings, reference, 0. if world % 2 == 0 else .15)
        episode, labels = [], []

        def record(public, action, info, verdict):
            frame = explorer.frame
            guilty = {i for group in harmful_groups(public, info) for i in group}
            eligible = [i for i, allowed in zip(frame.node_ids, frame.eligible) if allowed and i in guilty]
            label = np.zeros(5, dtype=np.float32)
            if eligible:
                label[[i + 1 for i in eligible]] = 1. / len(eligible)
            else:
                label[0] = 1.
            episode.append({"frame": frame})
            labels.append(label)

        run_episode(MarketEnv(config), population, seed + world, training=False,
                    coalition_policy=None if independent else coalition, solo=independent,
                    arbiter=explorer, evaluator=record)
        episodes.append(episode)
        targets.append(labels)
    batch, _, unions = collate(episodes)
    if any(ids != (0, 1, 2, 3) for ids in unions):
        raise ValueError("Expected the four market agents in every episode")
    label_array = np.zeros((*batch["valid"].shape, 5), dtype=np.float32)
    for i, labels in enumerate(targets):
        label_array[i, :len(labels)] = labels
    arrays = {key: value.numpy() for key, value in batch.items()}
    arrays["targets"] = label_array
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), {"worlds": worlds, "first_seed": seed,
        "sources": [{"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in sources],
        "public_inputs_only": True, "labels_used_only_in_training_loss": True,
        "source_policies_frozen": True, "independent_worlds": sum(i % 5 == 0 for i in range(worlds))})


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def minibatch(data, indices, device):
    indices = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[indices].to(device) for key, value in data.items()}


def assess(actor, data, device):
    loss_sum, count, sanctions, hits = 0., 0, 0, 0
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(data["valid"]), 32):
            batch = minibatch(data, np.arange(start, min(start + 32, len(data["valid"]))), device)
            targets = batch.pop("targets")
            distribution, _ = actor(batch)
            valid = batch["valid"]
            losses = -(targets * distribution.logits).sum(-1)
            loss_sum += float(losses[valid].sum().cpu())
            count += int(valid.sum().cpu())
            actions = greedy_choice(distribution)
            chosen = valid & (actions > 0)
            correct = targets.gather(-1, actions.unsqueeze(-1)).squeeze(-1) > 0
            sanctions += int(chosen.sum().cpu())
            hits += int((chosen & correct).sum().cpu())
    return {"loss": loss_sum / count, "sanctions": sanctions,
            "precision": hits / sanctions if sanctions else None,
            "validation_on_recorded_public_histories": True}


def pretrain_actor(adapter, root, train_path, validation_path, *, epochs, replica, precision_floor, save_epoch):
    """Only actor parameters change; snapshots retain the separate pretraining Adam."""
    policy = adapter.policy
    # These small supervised batches avoid GPU dispatch overhead. PPO rollouts
    # and updates still use the explicitly selected PPO device.
    training_device = torch.device("cpu")
    policy.actor.to(training_device)
    policy.metadata["pretrain_device"] = str(training_device)
    train, validation = load_dataset(train_path), load_dataset(validation_path)
    signature = hashlib.sha256(Path(train_path).read_bytes() + Path(validation_path).read_bytes()).hexdigest()
    if policy.metadata.get("pretrain_data_sha256", signature) != signature:
        raise ValueError("Resume requires identical supervised datasets")
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=.003)
    rng = np.random.default_rng(10_000 + replica)
    start = policy.metadata.get("pretrain_epoch", 0)
    if start:
        optimizer.load_state_dict(policy.metadata["pretrain_optimizer"])
        rng.bit_generator.state = policy.metadata["pretrain_shuffle_rng"]
    best = policy.metadata.get("pretrain_best")
    for epoch in range(start + 1, epochs + 1):
        policy.actor.train()
        order = rng.permutation(len(train["valid"]))
        for start_at in range(0, len(order), 32):
            batch = minibatch(train, order[start_at:start_at + 32], training_device)
            targets = batch.pop("targets")
            distribution, _ = policy.actor(batch)
            loss = -(targets * distribution.logits).sum(-1)[batch["valid"]].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite actor pretraining loss")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), policy.settings.max_gradient_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite actor pretraining gradient")
            optimizer.step()
        result = assess(policy.actor, validation, training_device)
        result.update(epoch=epoch)
        result["admissible"] = (result["precision"] is not None and result["precision"] >= precision_floor
                                and result["sanctions"] >= 10)
        if best is None or (result["admissible"], -result["loss"]) > (best["admissible"], -best["loss"]):
            best = result
        policy.metadata.update(actor_pretrained=True, pretrain_epoch=epoch, pretrain_data_sha256=signature,
            pretrain_best=best, pretrain_optimizer=cpu_tree(optimizer.state_dict()),
            pretrain_shuffle_rng=rng.bit_generator.state)
        save_epoch(epoch)
        print(f"replica={replica} actor pretrain epoch={epoch} loss={result['loss']:.4f} "
              f"precision={result['precision']} sanctions={result['sanctions']}", flush=True)
    selected = root / "weights" / f"pretrain_epoch_{best['epoch']:03d}" / "arbiter.pt"
    payload = torch.load(selected, map_location="cpu", weights_only=True)
    policy.actor.load_state_dict(payload["actor"])
    policy.actor.to(policy.device)
    policy.metadata.pop("pretrain_optimizer", None)
    policy.metadata.pop("pretrain_shuffle_rng", None)
    policy.metadata["pretrain_selected_checkpoint"] = str(selected)
    # PPO's own Adam has not been used during supervised pretraining.
    policy.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=policy.settings.learning_rate)
