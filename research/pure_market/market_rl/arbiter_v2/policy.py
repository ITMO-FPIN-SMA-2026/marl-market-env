"""Public-only recurrent PPO with a frozen empirical screening reference."""

from dataclasses import asdict
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .contracts import PublicMarketConfig as MarketConfig
from .contracts import ArbiterAction, ArbiterActionKind

from .features import Frame, PublicMemory
from .model import Actor, Critic, collate, greedy_choice
from .reference import PublicReference
from .settings import Settings

SCHEMA_VERSION = 3  # explicit demand, welfare and harm semantics; native weights only


def resolve_device(device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    resolved = torch.device(device)
    if resolved.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Supported devices: auto, cpu, cuda, mps')
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; choose an available device explicitly')
    if resolved.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; choose an available device explicitly')
    return resolved


def advantages(rewards, values, gamma, gae_lambda):
    """Complete finite-horizon episodes only; no artificial truncation bootstrap."""
    result = np.zeros(len(rewards), np.float32)
    carry = next_value = 0.0
    for t in range(len(rewards)-1, -1, -1):
        delta = rewards[t]+gamma*next_value-values[t]
        carry = delta+gamma*gae_lambda*carry
        result[t], next_value = carry, values[t]
    return result, result+np.asarray(values, np.float32)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def pack_frame(frame):
    return dict(global_features=torch.from_numpy(frame.global_features.copy()),
                nodes=torch.from_numpy(frame.nodes.copy()), pairs=torch.from_numpy(frame.pairs.copy()),
                node_ids=list(frame.node_ids), eligible=torch.from_numpy(frame.eligible.copy()),
                risks=torch.from_numpy(frame.risks.copy()))


def unpack_frame(data):
    return Frame(*(data[k].numpy().copy() for k in ('global_features', 'nodes', 'pairs')),
                 tuple(data['node_ids']), data['eligible'].numpy().copy(), data['risks'].numpy().copy())


class PPOGRUArbiterV2:
    def __init__(self, *, market=None, settings=None, reference_path=None, checkpoint_path=None,
                 checkpoint_out=None, model_seed=0, evaluation='greedy', device='cpu'):
        if evaluation not in ('greedy', 'sample'):
            raise ValueError('evaluation must be greedy or sample')
        if type(model_seed) is not int or model_seed < 0:
            raise ValueError('model_seed must be a nonnegative integer')
        self.device = resolve_device(device)
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True) if checkpoint_path else None
        if payload is not None and (not isinstance(payload, dict) or payload.get('schema_version') != SCHEMA_VERSION):
            raise ValueError('Unsupported V2 checkpoint schema; old PPO weights are incompatible')
        if payload and settings is not None and Settings(**settings) != Settings(**payload['settings']):
            raise ValueError('Settings differ from checkpoint')
        self.settings = Settings(**(payload['settings'] if payload else settings or {}))
        self.metadata = payload.get('metadata', {}) if payload else {}
        self.market = MarketConfig(**(market if market is not None else payload['market'] if payload else {}))
        if payload and asdict(self.market) != payload['market']:
            raise ValueError('Market differs from checkpoint')
        if not all(type(v) in (int, float) and math.isfinite(v) for k, v in asdict(self.market).items()
                   if k not in ('share_model', 'welfare_model', 'harm_baseline')):
            raise ValueError('Market settings must be finite')
        if not 0 < self.market.p_min < self.market.p_max or min(self.market.demand_a, self.market.demand_b) <= 0:
            raise ValueError('Invalid market scales')
        if (self.market.share_model not in ('tz_linear_weighted', 'logit_outside')
                or self.market.welfare_model != 'total_surplus'
                or self.market.harm_baseline not in ('proposals', 'nash')
                or min(self.market.welfare_scale, self.market.logit_temperature) <= 0):
            raise ValueError('Checkpoint uses an incompatible market model')
        self.reference = PublicReference.load(reference_path) if reference_path else (
            PublicReference.restore(payload['reference']) if payload and payload['reference'] else None)
        if payload and (self.reference.payload() if self.reference else None) != payload['reference']:
            # JSON lists and restored tuples are normalized by PublicReference.
            saved = PublicReference.restore(payload['reference']) if payload['reference'] else None
            if self.reference != saved:
                raise ValueError('Reference differs from checkpoint; masks cannot change mid-PPO')
        if self.reference:
            if self.reference.market != asdict(self.market):
                raise ValueError('Reference market differs')
            if (self.reference.min_history, self.reference.history_window) != (self.settings.min_history, self.settings.history_window):
                raise ValueError('Reference history contract differs')
        self.model_seed = payload['model_seed'] if payload else model_seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.model_seed)
            self.actor, self.critic = Actor(self.settings), Critic(self.settings)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.settings.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.settings.learning_rate)
        self.memory = PublicMemory(self.market, self.settings, self.reference)
        self.evaluation, self.checkpoint_out = evaluation, checkpoint_out
        self.completed, self.updates, self.episodes_seen, self.last_losses = [], 0, 0, {}
        self.reset(seed=0, training=False)
        if payload:
            self.actor.load_state_dict(payload['actor'])
            self.critic.load_state_dict(payload['critic'])
            self.actor_optimizer.load_state_dict(payload['actor_optimizer'])
            self.critic_optimizer.load_state_dict(payload['critic_optimizer'])
            self.completed = [[dict(item, frame=unpack_frame(item['frame'])) for item in ep] for ep in payload['completed']]
            self.updates, self.episodes_seen = payload['updates'], payload['episodes_seen']
            self.last_losses = payload['last_losses']
            self.generator.set_state(payload['generator'])

    def reset(self, *, seed, training):
        if getattr(self, 'episode_active', False) or getattr(self, 'episode', []) or getattr(self, 'pending', None) is not None:
            raise RuntimeError('Finish the episode before reset')
        if training and self.reference is None:
            raise ValueError('Fit a public reference before training')
        self.training = bool(training)
        self.actor.train(self.training)
        self.critic.train(self.training)
        self.memory.reset()
        self.actor_hidden, self.critic_hidden = {}, {}
        self.pending, self.episode, self.episode_active = None, [], False
        # Sampling on CPU makes state portable across CPU/CUDA/MPS checkpoints.
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.last_decision = dict(reason='reset')

    def _hidden(self, ids, saved):
        hidden = torch.zeros((1, max(1, len(ids)), self.settings.hidden_size), device=self.device)
        for k, i in enumerate(ids):
            if i in saved:
                hidden[0, k] = saved[i]
        return hidden

    def act(self, observation):
        if self.pending is not None:
            raise RuntimeError('Deliver the previous transition before act')
        frame = self.memory.update(observation)
        batch, _, unions = collate([[dict(frame=frame)]], self.device)
        ids = unions[0]
        with torch.no_grad():
            distribution, ah = self.actor(batch, self._hidden(ids, self.actor_hidden))
            value, ch = self.critic(batch, self._hidden(ids, self.critic_hidden))
            ready = self.training or self.updates > 0 or self.metadata.get('actor_pretrained', False)
            if not ready:
                choice = 0
            elif self.training or self.evaluation == 'sample':
                choice = int(torch.multinomial(distribution.probs.detach().cpu().reshape(-1), 1, generator=self.generator))
            else:
                choice = int(greedy_choice(distribution).item())
            log_prob = float(distribution.log_prob(torch.tensor([[choice]], device=self.device)).item())
        self.actor_hidden = {i: ah[0, k].detach().clone() for k, i in enumerate(ids)}
        self.critic_hidden = {i: ch[0, k].detach().clone() for k, i in enumerate(ids)}
        target = None if choice == 0 else ids[choice-1]
        action = ArbiterAction() if target is None else ArbiterAction(
            kind=ArbiterActionKind.sanction, target=target, fine=self.settings.fine_amount)
        self.pending = dict(step=observation.step, frame=frame, target=target, action=action,
                            log_prob=log_prob, value=float(value.item()))
        self.episode_active = True
        model_probs = distribution.probs.detach().cpu().reshape(-1).tolist()
        if ready and (self.training or self.evaluation == 'sample'):
            action_probs = model_probs
        else:
            action_probs = [float(k == choice) for k in range(len(model_probs))]
        self.last_decision = dict(reason='untrained_observe' if not ready else
            'no_screened_target' if not frame.eligible.any() else 'policy',
            target=target, fine=action.fine, node_ids=list(ids), eligible=frame.eligible.tolist(),
            screening_scores=frame.risks.tolist(), action_probabilities=action_probs,
            model_probabilities=model_probs,
            value=float(value.item()), trained_updates=self.updates,
            reference_sha256=self.reference.sha256 if self.reference else None)
        return action

    def observe_transition(self, *, observation, action, reward, next_observation, done):
        if self.pending is None or self.pending['step'] != observation.step:
            raise ValueError('Transition does not match pending decision')
        if action != self.pending['action']:
            raise ValueError('Applied action differs; align environment fine limits')
        if not math.isfinite(reward):
            raise ValueError('Reward must be finite')
        if not done and (next_observation is None or next_observation.step != observation.step+1):
            raise ValueError('Nonterminal transition needs the consecutive next observation')
        item, self.pending = self.pending, None
        self.memory.record_applied(action, observation.step)
        # Reward is a training target only. It never enters actor or critic inputs.
        if self.training:
            self.episode.append({k: item[k] for k in ('frame', 'target', 'log_prob', 'value')} | {'reward': float(reward)})
        if done:
            self.episode_active = False
            if self.training:
                self.completed.append(self.episode)
                self.episode = []
                self.episodes_seen += 1
                if len(self.completed) >= self.settings.batch_episodes:
                    self.learn()
                if self.checkpoint_out:
                    self.save_checkpoint(self.checkpoint_out)

    def learn(self):
        if self.episode_active or self.pending is not None or self.episode:
            raise RuntimeError('PPO updates require complete episodes')
        if not self.completed:
            return
        batch, choices, _ = collate(self.completed, self.device)
        old, adv, returns = (torch.zeros(batch['valid'].shape, device=self.device) for _ in range(3))
        for bi, ep in enumerate(self.completed):
            a, r = advantages([x['reward'] for x in ep], [x['value'] for x in ep], self.settings.gamma, self.settings.gae_lambda)
            size = len(ep)
            old[bi, :size] = torch.tensor([x['log_prob'] for x in ep], device=self.device)
            adv[bi, :size] = torch.from_numpy(a).to(self.device)
            returns[bi, :size] = torch.from_numpy(r).to(self.device)
        valid = batch['valid']
        adv = (adv-adv[valid].mean())/adv[valid].std(unbiased=False).clamp_min(1e-6)
        for epoch in range(self.settings.epochs):
            dist, _ = self.actor(batch)
            log_ratio = dist.log_prob(choices)-old
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1-self.settings.clip_ratio, 1+self.settings.clip_ratio)
            policy_loss = -torch.minimum(ratio*adv, clipped*adv)[valid].mean()
            entropy = dist.entropy()[valid].mean()
            values, _ = self.critic(batch)
            value_loss = (values-returns).square()[valid].mean()
            actor_loss = policy_loss-self.settings.entropy_coef*entropy
            critic_loss = self.settings.value_coef*value_loss
            if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                raise FloatingPointError('Nonfinite PPO loss')
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.settings.max_gradient_norm)
            critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.settings.max_gradient_norm)
            if not torch.isfinite(actor_norm) or not torch.isfinite(critic_norm):
                raise FloatingPointError('Nonfinite PPO gradient')
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            with torch.no_grad():
                updated, _ = self.actor(batch)
                post_log_ratio = updated.log_prob(choices)-old
                kl = ((post_log_ratio.exp()-1)-post_log_ratio)[valid].mean()
                clip_fraction = ((post_log_ratio.exp()-1).abs() > self.settings.clip_ratio)[valid].float().mean()
                variance = returns[valid].var(unbiased=False)
                ev = 1-(values[valid]-returns[valid]).var(unbiased=False)/variance if variance > 1e-8 else values.new_tensor(0)
            self.last_losses = dict(policy=float(policy_loss.detach()), value=float(value_loss.detach()),
                entropy=float(entropy.detach()), kl=float(kl), clip_fraction=float(clip_fraction),
                actor_gradient_norm=float(actor_norm), critic_gradient_norm=float(critic_norm),
                explained_variance=float(ev), epochs=epoch+1,
                transitions=int(valid.sum()), eligible_fraction=float(batch['eligible'].any(-1)[valid].float().mean()))
            if self.last_losses['kl'] > self.settings.target_kl:
                break
        self.completed = []
        self.updates += 1

    def save_checkpoint(self, path):
        if self.episode_active or self.pending is not None:
            raise RuntimeError('Save checkpoints at episode boundaries')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(schema_version=SCHEMA_VERSION, market=asdict(self.market), settings=asdict(self.settings),
            metadata=self.metadata,
            reference=self.reference.payload() if self.reference else None, model_seed=self.model_seed,
            actor=cpu_tree(self.actor.state_dict()), critic=cpu_tree(self.critic.state_dict()),
            actor_optimizer=cpu_tree(self.actor_optimizer.state_dict()), critic_optimizer=cpu_tree(self.critic_optimizer.state_dict()),
            completed=[[dict(item, frame=pack_frame(item['frame'])) for item in ep] for ep in self.completed],
            updates=self.updates, episodes_seen=self.episodes_seen, last_losses=self.last_losses,
            generator=self.generator.get_state())
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path
