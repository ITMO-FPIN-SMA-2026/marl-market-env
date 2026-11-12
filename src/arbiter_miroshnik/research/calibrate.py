"""Fit immutable public screening references without running an environment."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from social_market_env.contracts import ArbiterObservation

from ..checkpoint import market_config, market_payload

from ..features import PublicMemory
from ..reference import PublicReference
from ..settings import Settings

PUBLIC_FIELDS = {'step', 'active_agents', 'prices', 'quantities', 'total_quantity', 'average_price',
                 'social_welfare', 'delta_average_price', 'delta_social_welfare'}


def observation_from_public(row):
    if set(row) != PUBLIC_FIELDS:
        raise ValueError(f'Expected only the public schema, missing/extra fields: {set(row) ^ PUBLIC_FIELDS}')
    return ArbiterObservation(**dict(row, prices={int(k): v for k, v in row['prices'].items()},
                                    quantities={int(k): v for k, v in row['quantities'].items()}),
                              target_social_welfare=0.0, last_action=None, fines_history=())


def validate_dataset(data, role):
    if data.get('schema_version') != 1 or data.get('split_role') != role:
        raise ValueError(f'Expected schema 1, split_role={role}')
    episodes = data['episodes']
    ids = [ep['episode_id'] for ep in episodes]
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Require nonempty unique episode IDs')
    for ep in episodes:
        if not ep['observations']:
            raise ValueError('Empty calibration episode')
        for row in ep['observations']:
            observation_from_public(row)
    return tuple(ids)


def fit_reference(reference_data, calibration_data, *, settings=None, risk_quantile=.99, source_hashes=()):
    settings = settings or Settings()
    if not 0 < risk_quantile <= 1:
        raise ValueError('risk_quantile must be in (0,1]')
    reference_ids = validate_dataset(reference_data, 'reference_train_nonharmful')
    calibration_ids = validate_dataset(calibration_data, 'calibration_nonharmful')
    if reference_data['market'] != calibration_data['market']:
        raise ValueError('Datasets have different markets')
    market = market_config(reference_data['market'])
    for ep in reference_data['episodes']:
        memory = PublicMemory(market, settings, None)
        for row in ep['observations']:
            memory.update(observation_from_public(row))
    prices = [p for ep in reference_data['episodes'] for row in ep['observations'] for p in row['prices'].values()]
    if not prices or not np.isfinite(prices).all():
        raise ValueError('Reference dataset has no finite prices')
    reference = PublicReference(market_payload(market), float(np.median(prices)), float(np.quantile(prices, .95)),
        .7, settings.min_history, settings.history_window, reference_ids, calibration_ids, tuple(source_hashes))
    maxima = []
    for ep in calibration_data['episodes']:
        memory = PublicMemory(market, settings, reference)
        maximum = 0.0
        matured = False
        for row in ep['observations']:
            frame = memory.update(observation_from_public(row))
            maximum = max(maximum, float(frame.risks.max()) if len(frame.risks) else 0)
            matured |= bool(len(frame.nodes) and (frame.nodes[:, 11] > 0).any())
        if not matured:
            raise ValueError('Each calibration episode must contain a mature seller history')
        maxima.append(maximum)
    # Calibrate against an episode maximum rather than treating time steps as iid.
    observed = float(np.quantile(maxima, risk_quantile, method='higher'))
    threshold = max(.7, observed+1e-6)
    reference = replace(reference, risk_threshold=min(threshold, 1.000001))
    return reference, dict(reference_episodes=len(reference_ids), calibration_episodes=len(calibration_ids),
        risk_quantile=risk_quantile, episode_max_scores=maxima, threshold=reference.risk_threshold,
        observe_only_screen=reference.risk_threshold >= 1,
        warning='Empirical calibration only: no finite-sample FP guarantee; no hidden labels read.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--history-window', type=int, default=20)
    parser.add_argument('--min-history', type=int, default=5)
    parser.add_argument('--risk-quantile', type=float, default=.99)
    args = parser.parse_args()
    raw = [path.read_bytes() for path in (args.reference, args.calibration)]
    ref, report = fit_reference(*(json.loads(blob) for blob in raw),
        settings=Settings(history_window=args.history_window, min_history=args.min_history),
        risk_quantile=args.risk_quantile, source_hashes=[hashlib.sha256(blob).hexdigest() for blob in raw])
    ref.save(args.out)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
