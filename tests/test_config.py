from pathlib import Path

import pytest

from social_market_env.config import ComponentSpec, ConfigError, config_from_dict, load_config


def test_load_config_reads_nested_sections_and_keeps_defaults(tmp_path: Path) -> None:
    path = tmp_path / 'experiment.yaml'
    path.write_text(
        'name: demo\n'
        'market: {tau: 10}\n'
        'agents:\n'
        '  count: 4\n'
        '  overrides: {2: {impl: "pkg.mod:Cls", kwargs: {x: 1}}}\n',
        encoding='utf-8',
    )

    config = load_config(path)

    assert config.name == 'demo'
    assert config.market.tau == 10
    assert config.market.demand_a == 120.0
    assert config.agents.count == 4
    assert config.agents.overrides == {2: ComponentSpec(impl='pkg.mod:Cls', kwargs={'x': 1})}


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match='Unknown keys in config.market'):
        config_from_dict({'market': {'taus': 3}})


def test_invalid_share_model_is_rejected() -> None:
    with pytest.raises(ConfigError, match='share_model'):
        config_from_dict({'market': {'share_model': 'magic'}})


def test_non_positive_demand_slope_is_rejected() -> None:
    with pytest.raises(ConfigError, match='market.demand_b must be positive'):
        config_from_dict({'market': {'demand_b': 0}})


def test_inverted_cost_range_is_rejected() -> None:
    with pytest.raises(ConfigError, match='agents.cost_low must not exceed agents.cost_high'):
        config_from_dict({'agents': {'cost_low': 30, 'cost_high': 20}})
