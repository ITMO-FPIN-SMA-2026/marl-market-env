"""Frozen empirical reference from disjoint public, nonharmful training histories.

This is a screening score, NOT a probability of guilt or a Nash oracle.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .checkpoint import market_config, market_payload


@dataclass(frozen=True)
class PublicReference:
    market: dict
    centre_price: float
    upper_price: float
    risk_threshold: float
    min_history: int
    history_window: int
    reference_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    source_hashes: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self):
        config = market_config(self.market)
        if self.market != market_payload(config):
            raise ValueError('Reference must store the complete market configuration')
        if self.version != 1 or not 0 < config.p_min < config.p_max:
            raise ValueError('Invalid reference schema or price bounds')
        for value in (self.centre_price, self.upper_price, self.risk_threshold):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Reference values must be finite')
        if not config.p_min <= self.centre_price <= self.upper_price <= config.p_max:
            raise ValueError('Reference prices outside market bounds')
        if not .5 <= self.risk_threshold <= 1.000001:
            raise ValueError('Risk threshold outside supported range')
        if type(self.min_history) is not int or type(self.history_window) is not int or not 3 <= self.min_history <= self.history_window:
            raise ValueError('Invalid reference history contract')
        if not self.reference_ids or not self.calibration_ids:
            raise ValueError('A fitted reference requires reference and calibration episode IDs')
        if len(set(self.reference_ids)) != len(self.reference_ids) or len(set(self.calibration_ids)) != len(self.calibration_ids):
            raise ValueError('Duplicate episode IDs')
        if set(self.reference_ids) & set(self.calibration_ids):
            raise ValueError('Reference and calibration episodes must be disjoint')

    def level(self, price):
        scale = max(self.upper_price-self.centre_price, .05*(self.market['p_max']-self.market['p_min']))
        return float(np.clip((price-self.upper_price)/scale, 0, 1))

    def payload(self):
        return asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    @classmethod
    def restore(cls, payload):
        data = dict(payload)
        for name in ('reference_ids', 'calibration_ids', 'source_hashes'):
            data[name] = tuple(data[name])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.restore(json.loads(Path(path).read_text()))

    def save(self, path):
        # Calibration artifacts are immutable: use a new path for a new fit.
        with Path(path).open('x') as stream:
            json.dump(self.payload(), stream, indent=2, allow_nan=False)
            stream.write('\n')
