"""Public MarketAgentAPI adapter; one pending decision, no retained autograd graphs."""

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from social_market_env.contracts import AgentAction, AgentObservation, MarketAgentAPI

from .actions import Candidates, build_candidates
from .checkpoint import CHECKPOINT_VERSION, read_checkpoint, write_checkpoint
from .features import FEATURE_NAMES, FEATURE_VERSION, FeatureEncoder
from .learning import update
from .policy import ActorCritic, Choice, LegacyActorCritic
from .settings import MarketSettings, Settings


@dataclass(frozen=True)
class Pending:
    observation: AgentObservation
    action: AgentAction
    features: np.ndarray
    candidates: Candidates
    choice: Choice


class ZhmurenkoVladimirAgent(MarketAgentAPI):
    def __init__(self, *, market: dict[str, Any] | None = None, settings: dict[str, Any] | None = None,
                 model_seed: int = 0, checkpoint_path: str | None = None, evaluation_mode: str = 'greedy') -> None:
        if evaluation_mode not in ('greedy', 'seeded'):
            raise ValueError('evaluation_mode must be greedy or seeded')
        self.evaluation_mode = evaluation_mode
        payload = read_checkpoint(checkpoint_path) if checkpoint_path else None
        saved_settings = dict(payload['settings']) if payload else dict(settings or {})
        if payload:
            saved_settings.setdefault('architecture_version',1)
        self.settings = Settings(**saved_settings)
        self.market = MarketSettings(**payload['market']) if payload else MarketSettings.from_mapping(market)
        if payload and settings is not None:
            requested_settings=dict(settings)
            requested_settings.setdefault('architecture_version',self.settings.architecture_version)
            if Settings(**requested_settings) != self.settings:
                raise ValueError('Settings differ from the checkpoint; use its saved settings')
        if payload and market is not None and MarketSettings.from_mapping(market) != self.market:
            raise ValueError('Market bounds/normalization differ from the checkpoint')
        self.model_seed = int(payload['model_seed'] if payload else model_seed)
        self.encoder = FeatureEncoder(self.settings, self.market)
        # Initialization never consumes another agent's/global torch random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.model_seed)
            model_class = LegacyActorCritic if self.settings.architecture_version==1 else ActorCritic
            self.model = model_class(self.settings)
        actor_parameters = self.model.actor_parameters() if self.settings.architecture_version>=2 else self.model.parameters()
        self.optimizer = torch.optim.Adam(actor_parameters, lr=self.settings.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(),lr=self.settings.critic_learning_rate) if self.settings.architecture_version>=2 else None
        self.generator = torch.Generator(device='cpu').manual_seed(self.model_seed)
        self.agent_id: int | None = None
        self.training = False
        self._pending: Pending | None = None
        self.updates = 0
        self.last_metrics: dict[str, float] = {}
        self._learning_metrics = deque(maxlen=2)
        self.episode_info = {}
        if payload:
            self._restore(payload)

    def reset(self, *, agent_id: int, seed: int, training: bool) -> None:
        if training and self.settings.architecture_version==1:
            raise ValueError('Version-1 checkpoints are evaluation-only; train version 2 from a new initialization')
        self.agent_id = agent_id
        self.training = bool(training)
        derived = int(np.random.SeedSequence([int(seed), int(agent_id), self.model_seed]).generate_state(1, dtype=np.uint64)[0])
        self.generator.manual_seed(derived % (2**63 - 1))
        self.model.train(self.training)
        self._pending = None
        self.last_metrics = {}
        self._learning_metrics.clear()
        self.episode_info = {'agent_id':agent_id,'reset_seed':int(seed),'training':bool(training)}

    def act(self, observation: AgentObservation) -> AgentAction:
        if self.agent_id is None or observation.agent_id != self.agent_id:
            raise ValueError('reset() must establish the observation agent_id')
        if self.training and self._pending is not None:
            raise RuntimeError('Previous decision is waiting for observe_transition()')
        self.episode_info.setdefault('cost',float(observation.cost))
        features = self.encoder.encode(observation).values
        candidates = build_candidates(observation, self.settings, self.market)
        choice = self.model.choose(features, candidates, stochastic=self.training or self.evaluation_mode == 'seeded', generator=self.generator)
        join = choice.coalition == 1
        price = observation.offer.recommended_price if join else (
            candidates.prices[choice.price] if choice.price is not None else candidates.fallback_price)
        link = candidates.links[choice.link_kind][choice.link_target] if choice.link_target is not None else None
        action = AgentAction(price=float(price), participate=bool(choice.participation),
                             coalition_choice=int(observation.offer.plan_id) if join else None, link_action=link)
        if self.training:
            self._pending = Pending(observation, action, features, candidates, choice)
        return action

    def observe_transition(self, *, observation: AgentObservation, action: AgentAction, reward: float,
                           next_observation: AgentObservation | None, done: bool) -> None:
        if not self.training:
            return
        pending, self._pending = self._pending, None
        if pending is None or observation != pending.observation:
            raise ValueError('Transition does not match the pending observation')
        if action != pending.action:
            raise ValueError('Executed action differs from the policy action; refusing an incorrect gradient')
        if not math.isfinite(reward):
            raise ValueError('Reward must be finite')
        if not done and (next_observation is None or next_observation.agent_id != self.agent_id):
            raise ValueError('A nonterminal transition needs the same agent next observation')
        next_features = None if done else self.encoder.encode(next_observation).values
        self.last_metrics = update(self.model, self.optimizer, critic_optimizer=self.critic_optimizer, settings=self.settings, features=pending.features,
                                   candidates=pending.candidates, choice=pending.choice, reward=float(reward),
                                   next_features=next_features, done=done)
        self.updates += 1
        self.last_metrics.update(step=observation.step,done=bool(done),raw_reward=float(reward),
                                 price=action.price,cost=observation.cost,margin=action.price-observation.cost,
                                 participate=action.participate,joined=action.coalition_choice is not None)
        self._learning_metrics.append(dict(self.last_metrics))

    def drain_learning_metrics(self) -> list[dict[str, float]]:
        """Diagnostics only; at most two callbacks occur during one environment step."""
        result = list(self._learning_metrics)
        self._learning_metrics.clear()
        return result

    def save_checkpoint(self, path: str | Path, *, overwrite: bool = False) -> Path:
        if self._pending is not None:
            raise RuntimeError('Finish the pending transition before saving')
        return write_checkpoint(path, {
            'checkpoint_version': CHECKPOINT_VERSION, 'feature_version': FEATURE_VERSION,
            'feature_names': list(FEATURE_NAMES), 'settings': asdict(self.settings), 'market': asdict(self.market),
            'model_seed': self.model_seed, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'updates': self.updates,
            'critic_optimizer': self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None,
            'rng_state': self.generator.get_state(), 'torch_version': str(torch.__version__),
        }, overwrite=overwrite)

    def load_checkpoint(self, path: str | Path) -> None:
        if self._pending is not None:
            raise RuntimeError('Cannot load during an unfinished transition')
        payload = read_checkpoint(path)
        saved_settings=dict(payload['settings'])
        saved_settings.setdefault('architecture_version',1)
        if Settings(**saved_settings) != self.settings or MarketSettings(**payload['market']) != self.market:
            raise ValueError('Checkpoint architecture/normalization/settings mismatch')
        self._restore(payload)

    def _restore(self, payload: dict[str, Any]) -> None:
        self.model.load_state_dict(payload['model'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer'])
        if self.critic_optimizer is not None:
            self.critic_optimizer.load_state_dict(payload['critic_optimizer'])
        self.generator.set_state(payload['rng_state'])
        self.updates = int(payload['updates'])
        self.model_seed = int(payload['model_seed'])
