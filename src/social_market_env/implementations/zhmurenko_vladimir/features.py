"""Versioned features from AgentObservation only, with explicit missingness."""

import math
from dataclasses import dataclass

import numpy as np

from social_market_env.contracts import AgentObservation

from .settings import MarketSettings, Settings

FEATURE_VERSION = 1

def _optional_names(names):
    return tuple(part for name in names for part in (name, name + '_present'))


FEATURE_NAMES = (
    ('cost', 'balance', 'active', 'restricted', 'step', 'horizon')
    + _optional_names(('last_price', 'last_quantity', 'last_reward', 'market_average_price', 'market_total_quantity'))
    + tuple(part for group in ('public_prices', 'neighbor_weights', 'link_costs')
            for part in ((group + '_count',) + _optional_names(tuple(group + '_' + stat for stat in ('min', 'max', 'mean', 'std')))))
    + ('available_targets', 'offer_present')
    + _optional_names(('offer_price', 'offer_weight', 'offer_members'))
    + _optional_names(tuple('expected_' + name for name in ('average_price', 'quantity', 'gross_profit', 'coalition_payout', 'reward')))
    + ('evaluation_present',)
    + _optional_names(tuple('evaluated_' + name for name in ('price', 'quantity', 'gross_profit', 'coalition_payout', 'fine', 'link_fee', 'reward', 'profit_error', 'reward_error')))
    + _optional_names(('previous_join',)) + ('previous_plan_count',) + _optional_names(('previous_join_fraction',))
)


def finite(value: object) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


@dataclass(frozen=True)
class Encoded:
    names: tuple[str, ...]
    values: np.ndarray


class FeatureEncoder:
    def __init__(self, settings: Settings, market: MarketSettings) -> None:
        self.settings, self.market = settings, market

    def encode(self, observation: AgentObservation) -> Encoded:
        o, names, values = observation, [], []
        p, q, r = self.market.p_max, self.market.quantity_scale, self.settings.reward_scale

        def scalar(name: str, value: object, scale: float = 1.0) -> None:
            names.append(name)
            values.append(float(np.clip(float(value) / scale, -10, 10)) if finite(value) else 0.0)

        def optional(name: str, value: object, scale: float = 1.0) -> None:
            scalar(name, value, scale)
            scalar(name + '_present', int(finite(value)))

        def aggregate(name: str, source, scale: float) -> None:
            valid = np.asarray([float(value) for value in source if finite(value)], dtype=np.float64)
            scalar(name + '_count', math.log1p(valid.size))
            # Clip before aggregating to prevent overflow from malformed public input.
            normalized = np.clip(valid / scale, -10, 10)
            for suffix, function in (('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)):
                optional(name + '_' + suffix, float(function(normalized)) if valid.size else None)

        scalar('cost', o.cost, p)
        scalar('balance', o.balance, self.settings.balance_scale)
        scalar('active', int(o.active))
        scalar('restricted', int(o.restricted))
        scalar('step', o.step, self.settings.horizon)
        scalar('horizon', self.settings.horizon, 50.0)
        optional('last_price', o.last_price, p)
        optional('last_quantity', o.last_quantity if o.step else None, q)
        optional('last_reward', o.last_profit if o.step else None, r)
        optional('market_average_price', o.market.average_price, p)
        optional('market_total_quantity', o.market.total_quantity, q)
        aggregate('public_prices', o.market.last_prices.values(), p)
        aggregate('neighbor_weights', o.neighbors.values(), 1.0)
        aggregate('link_costs', o.link_costs.values(), r)
        scalar('available_targets', math.log1p(len(set(o.link_costs) - {o.agent_id})))
        scalar('offer_present', int(o.offer is not None))
        optional('offer_price', o.offer.recommended_price if o.offer else None, p)
        optional('offer_weight', o.offer.payout_weight if o.offer else None)
        optional('offer_members', len(o.offer.members) if o.offer else None, 12.0)
        for name, scale in (('average_price', p), ('quantity', q), ('gross_profit', r), ('coalition_payout', r), ('reward', r)):
            optional('expected_' + name, getattr(o.expectations, name), scale)
        evaluation = o.last_evaluation
        scalar('evaluation_present', int(evaluation is not None))
        for name, scale in (('price', p), ('quantity', q), ('gross_profit', r), ('coalition_payout', r),
                            ('fine', r), ('link_fee', r), ('reward', r), ('profit_error', r), ('reward_error', r)):
            optional('evaluated_' + name, getattr(evaluation, name) if evaluation else None, scale)
        optional('previous_join', int(evaluation.joined_plan_id is not None) if evaluation else None)
        scalar('previous_plan_count', math.log1p(len(o.last_coalition_plan_ids)))
        optional('previous_join_fraction', sum(o.last_coalition_join_vector) / len(o.last_coalition_join_vector)
                 if o.last_coalition_join_vector else None)
        if tuple(names) != FEATURE_NAMES:
            raise RuntimeError('Feature schema differs from the checkpoint schema')
        return Encoded(FEATURE_NAMES, np.asarray(values, dtype=np.float32))
