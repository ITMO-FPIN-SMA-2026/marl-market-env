"""Separate recurrent actor/critic; joint intervention/target probabilities."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from .features import GLOBAL_DIM, NODE_DIM, PAIR_DIM


def collate(episodes, device='cpu'):
    if not episodes or any(not ep for ep in episodes):
        raise ValueError('Expected nonempty complete episodes')
    unions = [tuple(sorted({i for item in ep for i in item['frame'].node_ids})) for ep in episodes]
    b, t = len(episodes), max(map(len, episodes))
    n = max(1, max(map(len, unions)))
    arrays = dict(global_features=np.zeros((b, t, GLOBAL_DIM), np.float32),
                  nodes=np.zeros((b, t, n, NODE_DIM), np.float32),
                  pairs=np.zeros((b, t, n, n, PAIR_DIM), np.float32),
                  present=np.zeros((b, t, n), bool), eligible=np.zeros((b, t, n), bool),
                  valid=np.zeros((b, t), bool))
    choices = np.zeros((b, t), np.int64)
    for bi, ep in enumerate(episodes):
        positions = {i: k for k, i in enumerate(unions[bi])}
        for ti, item in enumerate(ep):
            frame = item['frame']
            arrays['global_features'][bi, ti] = frame.global_features
            arrays['valid'][bi, ti] = True
            ix = [positions[i] for i in frame.node_ids]
            for k, a in enumerate(ix):
                arrays['nodes'][bi, ti, a] = frame.nodes[k]
                arrays['present'][bi, ti, a] = True
                arrays['eligible'][bi, ti, a] = frame.eligible[k]
                for j, c in enumerate(ix):
                    arrays['pairs'][bi, ti, a, c] = frame.pairs[k, j]
            target = item.get('target')
            choices[bi, ti] = 0 if target is None else 1+positions[target]
    return ({k: torch.from_numpy(v).to(device) for k, v in arrays.items()},
            torch.from_numpy(choices).to(device), unions)


class RecurrentEncoder(nn.Module):
    def __init__(self, hidden_size, use_recurrence=True):
        super().__init__()
        h = hidden_size
        self.hidden_size = h
        self.pair = nn.Sequential(nn.Linear(PAIR_DIM, h), nn.Tanh())
        self.node = nn.Sequential(nn.Linear(NODE_DIM+GLOBAL_DIM+h, h), nn.Tanh())
        self.gru = nn.GRUCell(h, h) if use_recurrence else None
        self.context = nn.Sequential(nn.Linear(2*h+GLOBAL_DIM, h), nn.Tanh())

    def forward(self, batch, hidden=None):
        nodes, global_, present = batch['nodes'], batch['global_features'], batch['present']
        b, t, n = nodes.shape[:3]
        pair_mask = present.unsqueeze(-1) & present.unsqueeze(-2)
        pair_mask = pair_mask & ~torch.eye(n, dtype=torch.bool, device=nodes.device)
        pair_features = self.pair(batch['pairs'])*pair_mask.unsqueeze(-1)
        messages = pair_features.sum(-2)/pair_mask.sum(-1, keepdim=True).clamp_min(1)
        expanded = global_.unsqueeze(-2).expand(-1, -1, n, -1)
        encoded = self.node(torch.cat((nodes, expanded, messages), -1))
        state = nodes.new_zeros((b, n, self.hidden_size)) if hidden is None else hidden
        states, contexts = [], []
        for step in range(t):
            state = (self.gru(encoded[:, step].reshape(b*n, -1), state.reshape(b*n, -1)).reshape(b, n, -1)
                     if self.gru is not None else encoded[:, step])
            mask = present[:, step].unsqueeze(-1)
            state = state*mask
            mean = state.sum(1)/mask.sum(1).clamp_min(1)
            maximum = state.masked_fill(~mask, -1e9).max(1).values
            maximum = torch.where(mask.any(1), maximum, torch.zeros_like(maximum))
            contexts.append(self.context(torch.cat((global_[:, step], mean, maximum), -1)))
            states.append(state)
        return torch.stack(states, 1), torch.stack(contexts, 1), state


class Actor(nn.Module):
    def __init__(self, settings):
        super().__init__()
        h = settings.hidden_size
        self.encoder = RecurrentEncoder(h, settings.use_recurrence)
        self.gate = nn.Linear(h, 1)
        self.target = nn.Sequential(nn.Linear(2*h, h), nn.Tanh(), nn.Linear(h, 1))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(settings.initial_fine_probability/(1-settings.initial_fine_probability)))

    def forward(self, batch, hidden=None):
        state, context, hidden = self.encoder(batch, hidden)
        targets = self.target(torch.cat((state, context.unsqueeze(-2).expand_as(state)), -1)).squeeze(-1)
        eligible = batch['eligible']
        targets = targets.masked_fill(~eligible, -1e9)
        gate = self.gate(context)
        available = eligible.any(-1, keepdim=True)
        observe = torch.where(available, F.logsigmoid(-gate), torch.zeros_like(gate))
        fine = (F.logsigmoid(gate)+F.log_softmax(targets, -1)).masked_fill(~eligible, -1e9)
        return Categorical(logits=torch.cat((observe, fine), -1)), hidden


class Critic(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.encoder = RecurrentEncoder(settings.hidden_size, settings.use_recurrence)
        self.value = nn.Linear(settings.hidden_size, 1)

    def forward(self, batch, hidden=None):
        _, context, hidden = self.encoder(batch, hidden)
        return self.value(context).squeeze(-1), hidden


def greedy_choice(distribution):
    """Decide intervention first, then target; this is distinct from joint MAP."""
    probs = distribution.probs
    target = probs[..., 1:].argmax(-1)+1
    return torch.where(probs[..., 1:].sum(-1) > probs[..., 0], target, torch.zeros_like(target))
