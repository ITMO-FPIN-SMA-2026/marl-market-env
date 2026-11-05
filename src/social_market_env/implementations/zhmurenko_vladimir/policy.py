"""Conditional categorical actor and a scalar critic on a shared MLP."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .actions import Candidates, LINK_FEATURES, PRICE_FEATURES
from .features import FEATURE_NAMES
from .settings import Settings


@dataclass(frozen=True)
class Choice:
    participation: int
    coalition: int | None
    price: int | None
    link_kind: int
    link_target: int | None


def categorical(logits: torch.Tensor, mask: tuple[bool, ...] | None = None) -> Categorical:
    if mask is not None:
        logits = logits.masked_fill(~torch.tensor(mask, dtype=torch.bool), -torch.inf)
    return Categorical(logits=logits)


class LegacyActorCritic(nn.Module):
    """Exact version-1 network retained solely for honest checkpoint comparisons."""
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        hidden, candidate_hidden = settings.hidden_size, settings.candidate_hidden_size
        self.encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.participation = nn.Linear(hidden, 2)
        self.coalition = nn.Linear(hidden, 2)
        self.link_kind = nn.Linear(hidden, 4)
        self.critic = nn.Linear(hidden, 1)
        self.price = nn.Sequential(nn.Linear(hidden + len(PRICE_FEATURES), candidate_hidden), nn.Tanh(), nn.Linear(candidate_hidden, 1))
        self.link_target = nn.Sequential(nn.Linear(hidden + len(LINK_FEATURES) + 4, candidate_hidden), nn.Tanh(), nn.Linear(candidate_hidden, 1))

    def state(self, features: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(torch.from_numpy(features))
        return hidden, self.critic(hidden).squeeze(-1)

    def price_logits(self, hidden: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(rows)
        return self.price(torch.cat((hidden.expand(len(rows), -1), tensor), dim=-1)).squeeze(-1)

    def target_logits(self, hidden: torch.Tensor, rows: np.ndarray, kind: int) -> torch.Tensor:
        tensor = torch.from_numpy(rows)
        operation = torch.nn.functional.one_hot(torch.tensor(kind), num_classes=4).float().expand(len(rows), -1)
        return self.link_target(torch.cat((hidden.expand(len(rows), -1), tensor, operation), dim=-1)).squeeze(-1)

    def choose(self, features: np.ndarray, candidates: Candidates, *, stochastic: bool, generator: torch.Generator) -> Choice:
        def pick(distribution: Categorical) -> int:
            if stochastic:
                return int(torch.multinomial(distribution.probs, 1, generator=generator).item())
            return int(distribution.probs.argmax().item())

        with torch.no_grad():
            hidden, _ = self.state(features)
            participation = pick(categorical(self.participation(hidden), (True, candidates.can_sell)))
            coalition = pick(categorical(self.coalition(hidden))) if participation and candidates.can_join else None
            price = pick(categorical(self.price_logits(hidden, candidates.price_features))) if participation and coalition != 1 else None
            kind = pick(categorical(self.link_kind(hidden), (True, *(bool(group) for group in candidates.links[1:]))))
            target = pick(categorical(self.target_logits(hidden, candidates.link_features[kind], kind))) if kind else None
        return Choice(participation, coalition, price, kind, target)

    def evaluate_choice(self, features: np.ndarray, candidates: Candidates, choice: Choice):
        """Only active decisions contribute; unused price/coalition heads get no gradient."""
        hidden, value = self.state(features)
        terms = [(categorical(self.participation(hidden), (True, candidates.can_sell)), choice.participation)]
        if choice.coalition is not None:
            terms.append((categorical(self.coalition(hidden)), choice.coalition))
        if choice.price is not None:
            terms.append((categorical(self.price_logits(hidden, candidates.price_features)), choice.price))
        terms.append((categorical(self.link_kind(hidden), (True, *(bool(group) for group in candidates.links[1:]))), choice.link_kind))
        if choice.link_target is not None:
            terms.append((categorical(self.target_logits(hidden, candidates.link_features[choice.link_kind], choice.link_kind)), choice.link_target))
        log_probability = torch.stack([distribution.log_prob(torch.tensor(index)) for distribution, index in terms]).sum()
        entropy = torch.stack([distribution.entropy() for distribution, _ in terms]).sum()
        return log_probability, entropy, value


class ActorCritic(LegacyActorCritic):
    """Version 2: smooth learned price density and disjoint actor/critic features."""

    def __init__(self, settings: Settings) -> None:
        nn.Module.__init__(self)
        hidden, candidate_hidden = settings.hidden_size, settings.candidate_hidden_size
        self.settings = settings
        self.encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.critic_encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.participation = nn.Linear(hidden, 2)
        self.coalition = nn.Linear(hidden, 2)
        self.link_kind = nn.Linear(hidden, 4)
        self.critic = nn.Linear(hidden, 1)
        # Both center and width are learned. No cost-plus rule or economic oracle.
        self.price = nn.Linear(hidden, 2)
        self.link_target = nn.Sequential(nn.Linear(hidden + len(LINK_FEATURES) + 4, candidate_hidden), nn.Tanh(), nn.Linear(candidate_hidden, 1))

    def state(self, features: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.from_numpy(features)
        value = self.critic(self.critic_encoder(tensor)).squeeze(-1)
        if self.settings.architecture_version>=3:
            remaining = (self.settings.horizon*(1-tensor[FEATURE_NAMES.index('step')])).clamp(0,self.settings.horizon)
            if self.settings.gamma==1:
                weight=remaining
            elif self.settings.gamma==0:
                weight=(remaining>0).float()
            else:
                weight=(1-torch.pow(torch.tensor(self.settings.gamma),remaining))/(1-self.settings.gamma)
            # A learned return-rate times the known remaining discount mass.
            # No market/reward prediction formula or hidden state is involved.
            value=value*weight
        return self.encoder(tensor), value

    def price_parameters(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.price(hidden)
        center = torch.sigmoid(raw[0])
        scale = self.settings.price_scale_min + (self.settings.price_scale_max-self.settings.price_scale_min)*torch.sigmoid(raw[1])
        return center, scale

    def price_logits(self, hidden: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
        center, scale = self.price_parameters(hidden)
        price = torch.from_numpy(rows[:, 0])
        return -0.5*((price-center)/scale).square()

    def actor_parameters(self):
        return [parameter for name,parameter in self.named_parameters() if not name.startswith('critic')]

    def critic_parameters(self):
        return [parameter for name,parameter in self.named_parameters() if name.startswith('critic')]

    def joint_entropy(self, features: np.ndarray, candidates: Candidates) -> torch.Tensor:
        """Exact entropy of the conditional action tree, including branch weights."""
        hidden = self.encoder(torch.from_numpy(features))
        participation = categorical(self.participation(hidden), (True,candidates.can_sell))
        coalition = categorical(self.coalition(hidden)) if candidates.can_join else None
        join_probability = coalition.probs[1] if coalition is not None else torch.zeros(())
        price = categorical(self.price_logits(hidden,candidates.price_features))
        sale_entropy = (coalition.entropy() if coalition is not None else torch.zeros(())) + (1-join_probability)*price.entropy()
        kind = categorical(self.link_kind(hidden),(True,*(bool(group) for group in candidates.links[1:])))
        links_entropy = kind.entropy()
        for index,group in enumerate(candidates.links):
            if group:
                links_entropy = links_entropy + kind.probs[index]*categorical(self.target_logits(hidden,candidates.link_features[index],index)).entropy()
        return participation.entropy() + participation.probs[1]*sale_entropy + links_entropy

    def evaluate_choice(self, features: np.ndarray, candidates: Candidates, choice: Choice):
        log_probability, _, value = super().evaluate_choice(features,candidates,choice)
        return log_probability, self.joint_entropy(features,candidates), value
