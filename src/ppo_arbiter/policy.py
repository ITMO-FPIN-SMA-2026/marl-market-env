"""Recurrent PPO with one observe action and one fixed-fine action per seller."""

from dataclasses import asdict
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from social_market_env.config import MarketConfig
from social_market_env.contracts import ArbiterAction, ArbiterActionKind, ArbiterAPI
from .features import Frame, PublicMemory
from .model import ActorCritic, collate
from .settings import Settings

SCHEMA_VERSION = 1


def advantages(rewards, values, gamma, gae_lambda):
    """GAE over a complete episode; there is no continuation after its terminal step."""
    result = np.zeros(len(rewards), dtype=np.float32)
    carry, next_value = 0.0, 0.0
    for t in range(len(rewards)-1, -1, -1):
        delta = rewards[t]+gamma*next_value-values[t]
        carry = delta+gamma*gae_lambda*carry
        result[t], next_value = carry, values[t]
    return result, result+np.asarray(values, dtype=np.float32)


class PPOArbiter(ArbiterAPI):
    """Learn fixed fines from scalar rewards, using public history only.

    Each seller has a separate recurrent state with shared GRU weights. Neither
    actor nor critic reads SW*, costs, coalitions, networks or opponents' models.
    Previous rewards optionally enter the next observation; the critic can use
    its own recurrent encoder. Both options default to False. Weights are updated
    between episodes and stay frozen in eval. An untrained instance is a random
    initial policy, not an effective regulator; use an evaluated checkpoint.
    """

    def __init__(self, *, market=None, settings=None, model_seed=0,
                 checkpoint_path=None, checkpoint_out=None, evaluation='greedy'):
        if evaluation not in ('greedy', 'sample'):
            raise ValueError('evaluation must be greedy or sample')
        self.evaluation = evaluation
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True) if checkpoint_path else None
        if payload is not None and payload.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported PPO arbiter checkpoint schema')
        saved = payload['settings'] if payload else None
        if saved is not None and settings is not None and Settings(**saved) != Settings(**settings):
            raise ValueError('Settings differ from checkpoint')
        self.settings = Settings(**(saved or settings or {}))
        self.market = MarketConfig(**(market if market is not None else payload['market'] if payload else {}))
        if payload and asdict(self.market) != payload['market']:
            raise ValueError('Market configuration differs from checkpoint')
        for name in ('demand_a', 'demand_b'):
            value = getattr(self.market, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'market.{name} must be finite and positive')
        if not all(math.isfinite(x) for x in (self.market.p_min, self.market.p_max)) or not 0 < self.market.p_min < self.market.p_max:
            raise ValueError('Market price bounds must satisfy 0 < p_min < p_max')
        if type(model_seed) is not int or model_seed < 0:
            raise ValueError('model_seed must be a non-negative integer')
        self.model_seed = payload['model_seed'] if payload else model_seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.model_seed)
            self.model = ActorCritic(self.settings)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings.learning_rate)
        self.memory = PublicMemory(self.market, self.settings)
        self.completed, self.updates, self.last_losses = [], 0, {}
        self.checkpoint_out = str(checkpoint_out) if checkpoint_out is not None else None
        self.reset(seed=0, training=False)
        if payload:
            self._restore(payload)

    def reset(self, *, seed, training):
        if getattr(self, 'training', False) and (getattr(self, 'pending', None) is not None or getattr(self, 'episode', [])):
            raise RuntimeError('Finish the training episode before reset')
        self.training = bool(training)
        self.memory.reset()
        self.hidden_by_id = {}
        self.critic_hidden_by_id = {}
        self.episode, self.pending = [], None
        self._episode_active = False
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.last_decision = dict(kind='reset')

    def act(self, observation):
        if self.training and self.pending is not None:
            raise RuntimeError('Receive the previous transition before the next act')
        frame = self.memory.update(observation)
        batch, _, unions = collate([[dict(frame=frame)]])
        ids = unions[0]
        hidden = torch.zeros((1, max(1, len(ids)), self.settings.hidden_size))
        for k, node in enumerate(ids):
            if node in self.hidden_by_id:
                hidden[0, k] = self.hidden_by_id[node]
        if self.settings.separate_critic:
            critic_hidden = torch.zeros_like(hidden)
            for k, node in enumerate(ids):
                if node in self.critic_hidden_by_id:
                    critic_hidden[0, k] = self.critic_hidden_by_id[node]
            hidden = (hidden, critic_hidden)
        with torch.no_grad():
            outputs, state = self.model(batch, hidden)
            dist = self.model.distribution(outputs)
            choice = (int(torch.multinomial(dist.probs.reshape(-1), 1, generator=self.generator).item())
                      if self.training or self.evaluation == 'sample' else int(dist.probs.reshape(-1).argmax().item()))
            log_prob = float(dist.log_prob(torch.tensor([[choice]])).item())
            value = float(outputs['value'].item())
        if self.settings.separate_critic:
            state, critic_state = state
            self.critic_hidden_by_id = {node: critic_state[0, k].detach().clone() for k, node in enumerate(ids)}
        self.hidden_by_id = {node: state[0, k].detach().clone() for k, node in enumerate(ids)}
        target = None if choice == 0 else ids[choice-1]
        action = (ArbiterAction() if target is None else
                  ArbiterAction(kind=ArbiterActionKind.sanction, target=target, fine=self.settings.fine_amount))
        self.pending = dict(step=observation.step, frame=frame, target=target, value=value,
                            log_prob=log_prob, action=action, training=self.training)
        self._episode_active = True
        self.last_decision = dict(kind=str(action.kind), target=target, fine=action.fine,
                                  estimated_return=value, observe_probability=float(dist.probs[0, 0, 0]))
        return action

    def observe_transition(self, *, observation, action, reward, next_observation, done):
        if self.pending is None or self.pending['step'] != observation.step:
            raise ValueError('Transition does not match pending action')
        item, self.pending = self.pending, None
        if not math.isfinite(reward):
            raise ValueError('Arbiter reward must be finite')
        # This callback runs before act(t+1), in train and eval alike. It never
        # exposes the reward of an action before that action has been selected.
        self.memory.previous_reward = float(reward)
        if item['training']:
            if action != item['action']:
                self.episode = []
                raise ValueError('Applied action differs from requested action; align environment fine limits')
            self.episode.append({k: item[k] for k in ('frame', 'target', 'value', 'log_prob')} | {'reward': float(reward)})
            if done:
                self.completed.append(self.episode)
                self.episode = []
                if len(self.completed) >= self.settings.batch_episodes:
                    self.learn()
        if done:
            self._episode_active = False
            if item['training'] and self.checkpoint_out:
                self.save_checkpoint(self.checkpoint_out)

    def learn(self):
        """Update from complete on-policy episodes; flush a final partial batch explicitly."""
        if self.pending is not None or self.episode:
            raise RuntimeError('Finish the training episode before updating PPO')
        if not self.completed:
            return
        batch, choices, _ = collate(self.completed)
        shape = batch['valid'].shape
        old_log_prob, adv, returns = (torch.zeros(shape) for _ in range(3))
        for i, ep in enumerate(self.completed):
            a, r = advantages([x['reward'] for x in ep], [x['value'] for x in ep],
                              self.settings.gamma, self.settings.gae_lambda)
            n = len(ep)
            old_log_prob[i, :n] = torch.tensor([x['log_prob'] for x in ep])
            adv[i, :n], returns[i, :n] = torch.from_numpy(a), torch.from_numpy(r)
        valid = batch['valid']
        selected = adv[valid]
        adv = (adv-selected.mean())/selected.std(unbiased=False).clamp_min(1e-6)
        for _ in range(self.settings.epochs):
            outputs, _ = self.model(batch)
            dist = self.model.distribution(outputs)
            new_log_prob = dist.log_prob(choices)
            log_ratio = new_log_prob-old_log_prob
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1-self.settings.clip_ratio, 1+self.settings.clip_ratio)
            policy_loss = -torch.minimum(ratio*adv, clipped*adv)[valid].mean()
            value_loss = (outputs['value']-returns).square()[valid].mean()
            entropy = dist.entropy()[valid].mean()
            loss = policy_loss+self.settings.value_coef*value_loss-self.settings.entropy_coef*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            self.optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.max_gradient_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite PPO gradient')
            self.optimizer.step()
            kl = ((ratio-1)-log_ratio)[valid].mean()
            self.last_losses = dict(policy=float(policy_loss.detach()), value=float(value_loss.detach()),
                                    entropy=float(entropy.detach()), kl=float(kl.detach()))
            if self.last_losses['kl'] > self.settings.target_kl:
                break
        self.completed = []
        self.updates += 1

    def save_checkpoint(self, path):
        if self._episode_active or self.pending is not None:
            raise RuntimeError('Save at an episode boundary')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        episodes = [[dict(frame=x['frame'].serialise(), target=x['target'], value=x['value'],
                          log_prob=x['log_prob'], reward=x['reward']) for x in ep] for ep in self.completed]
        payload = dict(schema_version=SCHEMA_VERSION, market=asdict(self.market), settings=asdict(self.settings),
                       model_seed=self.model_seed, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                       completed=episodes, updates=self.updates, generator=self.generator.get_state(),
                       last_losses=self.last_losses)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _restore(self, payload):
        self.model.load_state_dict(payload['model'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.updates = int(payload['updates'])
        self.completed = [[dict(x, frame=Frame.restore(x['frame'])) for x in ep] for ep in payload['completed']]
        self.generator.set_state(payload['generator'])
        self.last_losses = dict(payload['last_losses'])
