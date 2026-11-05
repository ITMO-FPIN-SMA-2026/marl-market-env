"""Frozen evaluation and paired baseline experiments on held-out seeds."""

import argparse
import copy
import json
import hashlib
from pathlib import Path

import torch

from social_market_env.config import config_from_dict, config_to_dict, load_config
from social_market_env.runner.experiment import build_environment

from .checkpoint import read_checkpoint
from .metrics import episode_metrics, experiment_metadata, play_episode, summarize
from .settings import PACKAGE_ROOT, writable_path
from .train import AGENT_IMPL


def evaluate(*, config_path: Path, checkpoint: Path, output_dir: Path,
             episodes_per_seed: int | None = None, baselines=('CartelAgent', 'BestResponseAgent', 'RandomAgent'),
             reference_checkpoint: Path | None = None) -> dict:
    raw = config_to_dict(load_config(config_path))
    raw.update(mode='eval', strict=True)
    if episodes_per_seed is not None:
        raw['episodes_per_seed'] = episodes_per_seed
    if raw['episodes_per_seed'] < 1 or not raw['seeds']:
        raise ValueError('Evaluation needs nonempty seeds and positive episodes_per_seed')
    payload = read_checkpoint(checkpoint)
    if payload['settings']['horizon']!=raw['horizon']:
        raise ValueError('Checkpoint horizon differs from the evaluation horizon')
    if reference_checkpoint and read_checkpoint(reference_checkpoint)['settings']['horizon']!=raw['horizon']:
        raise ValueError('Reference checkpoint horizon differs from the evaluation horizon')
    output = writable_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'experiment.json').write_text(json.dumps(raw, indent=2), encoding='utf-8')
    (output/'metadata.json').write_text(json.dumps(experiment_metadata(),indent=2),encoding='utf-8')
    hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in (checkpoint,reference_checkpoint) if path is not None}
    report = {'checkpoint': str(checkpoint.resolve()), 'strategies': {}, 'frozen_weights_verified': True,'checkpoint_hashes':hashes}
    neural_strategies = (('old_trained','old_trained_sampled') if reference_checkpoint else ()) + ('trained', 'trained_sampled', 'untrained', 'untrained_sampled')
    for strategy in (*neural_strategies, *baselines):
        config_data = copy.deepcopy(raw)
        selected_payload=read_checkpoint(reference_checkpoint) if strategy.startswith('old_') else payload
        saved_settings=dict(selected_payload['settings'])
        saved_settings.setdefault('architecture_version',1)
        kwargs = {'settings': saved_settings, 'model_seed': selected_payload['model_seed']}
        if strategy in neural_strategies:
            kwargs['evaluation_mode'] = 'seeded' if strategy.endswith('_sampled') else 'greedy'
            if strategy.startswith(('trained','old_trained')):
                kwargs['checkpoint_path'] = str((reference_checkpoint if strategy.startswith('old_') else checkpoint).resolve())
            replacement = {'impl': AGENT_IMPL, 'kwargs': kwargs}
        else:
            replacement = {'impl': 'social_market_env.baselines.agents:' + strategy,
                           'kwargs': {'low':20, 'high':60} if strategy == 'RandomAgent' else {}}
        config_data['agents']['overrides'][0] = replacement
        episodes = []
        for seed in raw['seeds']:
            env = build_environment(config=config_from_dict(config_data), seed=seed)
            own = env.agents[0] if strategy in neural_strategies else None
            before = {name:value.clone() for name,value in own.model.state_dict().items()} if own else None
            initial_updates = own.updates if own else None
            for episode in range(raw['episodes_per_seed']):
                records, _, reset_seed = play_episode(env, seed=seed, episode=episode, training=False,
                    directory=output / strategy / f'seed_{seed}' / f'episode_{episode}',learner=own)
                episodes.append(dict(episode_metrics(records, agent_id=0,cost=own.episode_info.get('cost') if own else None), seed=seed, episode=episode, reset_seed=reset_seed))
            if own and (own.updates != initial_updates or any(not torch.equal(value, before[name]) for name,value in own.model.state_dict().items())):
                raise AssertionError('Evaluation changed trained parameters')
            print(f'{strategy}: seed={seed} complete', flush=True)
        summary = summarize(episodes)
        report['strategies'][strategy] = summary
        (output/strategy/'episodes.json').write_text(json.dumps(episodes, indent=2), encoding='utf-8')
    trained = report['strategies']['trained']['mean_reward_by_seed']
    report['paired_seed_reward_differences'] = {
        strategy:{seed:trained[seed] - value for seed,value in summary['mean_reward_by_seed'].items()}
        for strategy,summary in report['strategies'].items() if strategy != 'trained'}
    sampled = report['strategies']['trained_sampled']['mean_reward_by_seed']
    report['sampled_paired_seed_reward_differences'] = {
        strategy:{seed:sampled[seed] - value for seed,value in summary['mean_reward_by_seed'].items()}
        for strategy,summary in report['strategies'].items() if strategy != 'trained_sampled'}
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path, default=PACKAGE_ROOT / 'experiments/eval.yaml')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--reference-checkpoint', type=Path,help='preserved version-1 model for paired comparison')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--episodes-per-seed', type=int)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    evaluate(config_path=args.config, checkpoint=args.checkpoint, output_dir=args.output_dir, episodes_per_seed=args.episodes_per_seed,
             reference_checkpoint=args.reference_checkpoint)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
