"""Portable native seller states; no executable pickle or branch modifications."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile


def save_population(agents, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, agent in agents.items():
        path = directory/f'agent_{i}.json'
        if hasattr(agent, 'save'):
            # Native agents may require their save destination outside the Git
            # checkout. Serialize there, then atomically install the JSON state.
            with tempfile.TemporaryDirectory(prefix='team-seller-state-') as temporary:
                serialized = Path(temporary) / path.name
                agent.save(serialized)
                pending = path.with_suffix('.tmp')
                shutil.copyfile(serialized, pending)
                pending.replace(path)
        else:
            names = ('opening_markup', 'join_slack', 'max_cartel_premium', 'min_payout_weight', 'learn_rate')
            constructor = {name: float(getattr(agent, name)) for name in names}
            constructor['damping'] = float(agent.base_damping)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(constructor=constructor, learned=dict(
                join_bias=float(agent.join_bias), ema_reward=float(agent._ema_reward))), indent=2)+'\n')
            temporary.replace(path)


def load_population(prototypes, directory, market):
    directory = Path(directory)
    public_market = asdict(market)
    result = {
        0: type(prototypes[0]).from_checkpoint(directory/'agent_0.json', market=public_market),
        1: type(prototypes[1]).load(directory/'agent_1.json'),
        3: type(prototypes[3])(model_path=directory/'agent_3.json', market=public_market),
    }
    payload = json.loads((directory/'agent_2.json').read_text())
    result[2] = type(prototypes[2])(market=public_market, **payload['constructor'])
    result[2].join_bias = payload['learned']['join_bias']
    result[2]._ema_reward = payload['learned']['ema_reward']
    if result[1].market != public_market:
        raise ValueError('Saved seller market differs from this run')
    return dict(sorted(result.items()))
