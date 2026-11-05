"""Train one policy across environment seeds using explicit checkpoint handoff."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from social_market_env.config import config_from_dict, config_to_dict, load_config
from social_market_env.runner.experiment import build_environment

from .agent import ZhmurenkoVladimirAgent
from .checkpoint import read_checkpoint
from .metrics import episode_metrics, experiment_metadata, play_episode, summarize
from .settings import PACKAGE_ROOT, writable_path

AGENT_IMPL = 'social_market_env.implementations.zhmurenko_vladimir.agent:ZhmurenkoVladimirAgent'


def train(*, config_path: Path, output_dir: Path, episodes_per_seed: int | None = None,
          checkpoint: Path | None = None) -> dict:
    raw = config_to_dict(load_config(config_path))
    raw.update(mode='train', strict=True)
    if episodes_per_seed is not None:
        raw['episodes_per_seed'] = episodes_per_seed
    if raw['episodes_per_seed'] < 1 or not raw['seeds']:
        raise ValueError('Training needs nonempty seeds and positive episodes_per_seed')
    spec = raw['agents']['overrides'].get(0)
    if spec is None or spec['impl'] != AGENT_IMPL:
        raise ValueError('Training config must assign this implementation to agent 0')
    spec['kwargs'].setdefault('settings',{})['horizon']=raw['horizon']
    if checkpoint and read_checkpoint(checkpoint)['settings']['horizon']!=raw['horizon']:
        raise ValueError('Checkpoint horizon differs from the experiment horizon')
    output = writable_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'experiment.json').write_text(json.dumps(raw, indent=2), encoding='utf-8')
    (output / 'metadata.json').write_text(json.dumps(experiment_metadata(),indent=2),encoding='utf-8')
    previous_checkpoint = str(checkpoint.resolve()) if checkpoint else None
    episodes, losses, initial, learner = [], [], None, None
    with (output / 'episodes.jsonl').open('w', encoding='utf-8') as stream:
        for seed in raw['seeds']:
            spec['kwargs']['checkpoint_path'] = previous_checkpoint
            env = build_environment(config=config_from_dict(raw), seed=seed)
            # Access only our own explicitly configured component, never a competitor's internals.
            learner = env.agents[0]
            if not isinstance(learner, ZhmurenkoVladimirAgent):
                raise TypeError('Unexpected learner implementation')
            if initial is None:
                initial = {name: value.detach().clone() for name, value in learner.model.state_dict().items()}
            for episode in range(raw['episodes_per_seed']):
                before = learner.updates
                records, learning, reset_seed = play_episode(env, seed=seed, episode=episode, training=True,
                    directory=output / f'seed_{seed}' / f'episode_{episode}', learner=learner)
                metrics = dict(episode_metrics(records, agent_id=0,cost=learner.episode_info.get('cost')), seed=seed, episode=episode, reset_seed=reset_seed,
                               updates=learner.updates-before)
                metrics['learning']={key:sum(row[key] for row in learning)/len(learning) for key in
                    ('critic_loss','critic_objective','td_error','entropy','gradient_norm','actor_gradient_norm','critic_gradient_norm')} if learning else {}
                losses.extend(learning)
                episodes.append(metrics)
                stream.write(json.dumps(metrics) + '\n')
                stream.flush()
                if (episode + 1) % 10 == 0 or episode + 1 == raw['episodes_per_seed']:
                    print(f'seed={seed} episode={episode+1} reward={metrics["total_reward"]:.3f} updates={learner.updates}', flush=True)
            path = output / f'agent_0_seed_{seed}.pt'
            learner.save_checkpoint(path)
            previous_checkpoint = str(path)
    changes = {name: float((value - initial[name]).double().norm()) for name, value in learner.model.state_dict().items()}
    report = {'summary': summarize(episodes), 'checkpoint': previous_checkpoint, 'updates': learner.updates,
              'parameter_delta_l2': sum(delta**2 for delta in changes.values())**0.5,
              'parameter_changes': changes, 'losses_finite': all(math.isfinite(value) for item in losses for value in item.values()),
              'last_learning_metrics': learner.last_metrics, 'learning_observations': len(losses),
              'settings': asdict(learner.settings), 'model_parameters': sum(p.numel() for p in learner.model.parameters()),
              'torch_version': str(torch.__version__), 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    (output / 'learning.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in losses), encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path, default=PACKAGE_ROOT / 'experiments/train.yaml')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--episodes-per-seed', type=int)
    parser.add_argument('--checkpoint', type=Path, help='resume an explicit checkpoint')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    train(config_path=args.config, output_dir=args.output_dir, episodes_per_seed=args.episodes_per_seed, checkpoint=args.checkpoint)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
