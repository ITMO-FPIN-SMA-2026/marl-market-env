"""A shared GRU per seller, with IDs used only to align memory and actions."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .features import GLOBAL_DIM, NODE_DIM


def collate(episodes):
    """Keep each episode's seller axis stable when participants disappear/reappear."""
    unions = [tuple(sorted({i for x in ep for i in x['frame'].node_ids})) for ep in episodes]
    b, t, n = len(episodes), max(map(len, episodes)), max(1, max(map(len, unions)))
    global_dim = episodes[0][0]['frame'].global_features.size
    arrays = dict(global_features=np.zeros((b, t, global_dim), np.float32),
                  nodes=np.zeros((b, t, n, NODE_DIM), np.float32),
                  present=np.zeros((b, t, n), bool), eligible=np.zeros((b, t, n), bool),
                  valid=np.zeros((b, t), bool))
    choices = np.zeros((b, t), np.int64)
    for bi, ep in enumerate(episodes):
        index = {node: k for k, node in enumerate(unions[bi])}
        for ti, item in enumerate(ep):
            f = item['frame']
            if f.global_features.size != global_dim:
                raise ValueError('Cannot mix policies with different input schemas in one batch')
            arrays['global_features'][bi, ti] = f.global_features
            arrays['valid'][bi, ti] = True
            for k, node in enumerate(f.node_ids):
                position = index[node]
                arrays['nodes'][bi, ti, position] = f.nodes[k]
                arrays['present'][bi, ti, position] = True
                arrays['eligible'][bi, ti, position] = f.eligible[k]
            target = item.get('target')
            choices[bi, ti] = 0 if target is None else 1+index[target]
    return {k: torch.from_numpy(v) for k, v in arrays.items()}, torch.from_numpy(choices), unions


class ActorCritic(nn.Module):
    def __init__(self, settings):
        super().__init__()
        h = settings.hidden_size
        self.hidden_size = h
        self.encoder = nn.Sequential(nn.Linear(NODE_DIM+GLOBAL_DIM, h), nn.Tanh())
        self.gru = nn.GRUCell(h, h)
        self.context = nn.Sequential(nn.Linear(2*h+GLOBAL_DIM, h), nn.Tanh())
        self.observe = nn.Linear(h, 1)
        self.sanction = nn.Sequential(nn.Linear(2*h, h), nn.Tanh(), nn.Linear(h, 1))
        self.value = nn.Linear(h, 1)
        self.separate_critic = settings.separate_critic
        if settings.use_previous_reward:
            # Preserve the exact original actor/critic initialization in a paired
            # experiment. New reward columns start at zero and can learn normally.
            self.encoder[0] = self._add_zero_column(self.encoder[0])
            self.context[0] = self._add_zero_column(self.context[0], position=GLOBAL_DIM)
        if self.separate_critic:
            self.critic_encoder = deepcopy(self.encoder)
            self.critic_gru = deepcopy(self.gru)
            self.critic_context = deepcopy(self.context)

    @staticmethod
    def _add_zero_column(layer, position=None):
        position = layer.in_features if position is None else position
        with torch.random.fork_rng(devices=[]):
            extended = nn.Linear(layer.in_features+1, layer.out_features)
        with torch.no_grad():
            extended.weight[:, :position].copy_(layer.weight[:, :position])
            extended.weight[:, position].zero_()
            extended.weight[:, position+1:].copy_(layer.weight[:, position:])
            extended.bias.copy_(layer.bias)
        return extended

    @staticmethod
    def _pool(state, present):
        mean = state.sum(1)/present.sum(1).clamp_min(1)
        maximum = state.masked_fill(~present, -1e9).max(1).values
        maximum = torch.where(present.any(1), maximum, torch.zeros_like(maximum))
        return mean, maximum

    def forward(self, batch, hidden=None):
        nodes, global_ = batch['nodes'], batch['global_features']
        b, t, n = nodes.shape[:3]
        zero = nodes.new_zeros((b, n, self.hidden_size))
        if self.separate_critic:
            state, critic_state = (zero, zero.clone()) if hidden is None else hidden
        else:
            state = zero if hidden is None else hidden
        logits, values = [], []
        for step in range(t):
            present = batch['present'][:, step].unsqueeze(-1)
            global_step = global_[:, step]
            inputs = torch.cat((nodes[:, step], global_step[:, None].expand(-1, n, -1)), -1)
            encoded = self.encoder(inputs)
            state = self.gru(encoded.reshape(b*n, -1), state.reshape(b*n, -1)).reshape(b, n, -1)
            state = state*present  # An absent seller cannot retain an unobserved continuation.
            mean, maximum = self._pool(state, present)
            context = self.context(torch.cat((global_step, mean, maximum), -1))
            targets = self.sanction(torch.cat((state, context[:, None].expand(-1, n, -1)), -1)).squeeze(-1)
            targets = targets.masked_fill(~batch['eligible'][:, step], -1e9)
            logits.append(torch.cat((self.observe(context), targets), -1))
            if self.separate_critic:
                critic_encoded = self.critic_encoder(inputs)
                critic_state = self.critic_gru(critic_encoded.reshape(b*n, -1), critic_state.reshape(b*n, -1)).reshape(b, n, -1)
                critic_state = critic_state*present
                cm, cx = self._pool(critic_state, present)
                value_context = self.critic_context(torch.cat((global_step, cm, cx), -1))
            else:
                value_context = context
            values.append(self.value(value_context).squeeze(-1))
        final_state = (state, critic_state) if self.separate_critic else state
        return dict(logits=torch.stack(logits, 1), value=torch.stack(values, 1)), final_state

    @staticmethod
    def distribution(outputs):
        return Categorical(logits=outputs['logits'])
