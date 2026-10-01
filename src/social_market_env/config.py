"""Experiment configuration and YAML loading."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

SHARE_MODELS: frozenset[str] = frozenset({'logit', 'tz_linear'})
TOPOLOGIES: frozenset[str] = frozenset({'complete', 'star', 'path', 'inner_coalitions', 'random'})
NETWORK_MODES: frozenset[str] = frozenset({'static', 'evolving'})
RUN_MODES: frozenset[str] = frozenset({'eval', 'train'})


class ConfigError(ValueError):
    """Raised when an experiment config is invalid."""


@dataclass
class ComponentSpec:
    """Where to import a pluggable component from."""

    impl: str
    """Import path in the form 'package.module:ClassName'."""
    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the constructor."""


@dataclass
class MarketConfig:
    """Demand and price bounds of the market."""

    demand_a: float = 120.0
    """Intercept A of the demand Q = max(0, A - B * mean price)."""
    demand_b: float = 1.0
    """Slope B of the demand."""
    tau: float = 5.0
    """Logit temperature: lower means buyers react harder to price differences."""
    p_min: float = 1.0
    """Lowest allowed price."""
    p_max: float = 100.0
    """Highest allowed price."""
    share_model: str = 'logit'
    """How demand is split between sellers: 'logit' or the assignment's 'tz_linear'."""


@dataclass
class FinanceConfig:
    """Financial reserve rules."""

    initial_balance: float = 1000.0
    """Reserve of every agent at the start."""
    max_negative_steps: int = 3
    """Agent is eliminated after more than this many negative steps in a row."""


@dataclass
class NetworkConfig:
    """Initial topology and link evolution parameters."""

    topology: str = 'random'
    """One of complete, star, path, inner_coalitions, random."""
    mode: str = 'evolving'
    """'static' keeps the initial graph, 'evolving' updates it every step."""
    initial_density: float = 0.3
    """Target share of present links for the random topology."""
    inner_groups: int = 3
    """Number of cliques for the inner_coalitions topology."""
    a_keep: float = 2.0
    """Base logit of keeping an existing link."""
    a_form: float = -4.0
    """Base logit of forming a missing link."""
    w_affinity: float = 1.0
    """Weight of the initial pair affinity z_ij."""
    w_history: float = 2.0
    """Weight of cooperation history H_ij."""
    w_utility: float = 0.5
    """Weight of received utility U_ij."""
    w_coop: float = 1.0
    """Weight of expected cooperation gain C_ij; 0 skips its computation."""
    w_link_cost: float = 1.0
    """Weight of the arbiter-controlled link cost c_ij."""
    agent_link_bonus: float = 1.5
    """Logit bonus per endpoint asking to add or maintain a link."""
    memory_rate: float = 0.2
    """Update rate of the H_ij and U_ij moving averages."""
    link_cost_decay: float = 0.0
    """Share of every link cost that fades each step."""
    utility_scale: float = 100.0
    """Money scale used to squash profits into [-1, 1] signals."""


@dataclass
class CoalitionsConfig:
    """Coalition policy and compliance rules."""

    policy: ComponentSpec = field(
        default_factory=lambda: ComponentSpec(impl='social_market_env.baselines.coalitions:NoCoalitions')
    )
    """Coalition policy implementation."""
    compliance_tolerance: float = 0.5
    """Largest price deviation from the recommendation that still counts as compliance."""


@dataclass
class ArbiterConfig:
    """Arbiter implementation and limits."""

    policy: ComponentSpec = field(
        default_factory=lambda: ComponentSpec(
            impl='social_market_env.baselines.arbiters:AlwaysObserve'
        )
    )
    """Arbiter implementation."""
    max_fine: float = 100.0
    """Upper bound of a single fine."""
    min_fine: float = 1.0
    """Smallest fine that counts as a sanction; smaller fines are turned into observe."""
    max_link_cost_delta: float = 5.0
    """Largest absolute change of one link cost per step."""


@dataclass
class ScoringConfig:
    """Coefficients of the arbiter reward and the harm threshold."""

    alpha: float = 1.0
    """Weight of the relative welfare gap |SW - SW*| / SW*."""
    beta: float = 0.1
    """Cost of any intervention."""
    lambda_fp: float = 0.5
    """Penalty for fining an agent outside harmful coalitions."""
    lambda_fn: float = 0.5
    """Penalty for leaving a harmful coalition unaddressed."""
    harm_margin: float = 0.05
    """Coalition is harmful when welfare is this share below its competitive counterfactual."""


@dataclass
class AgentsConfig:
    """Market agents: count, implementation and costs."""

    count: int = 12
    """Number of agents."""
    default: ComponentSpec = field(
        default_factory=lambda: ComponentSpec(
            impl='social_market_env.baselines.agents:BestResponseAgent'
        )
    )
    """Implementation used for every agent without an override."""
    cost_low: float = 15.0
    """Lower bound of uniformly drawn unit costs."""
    cost_high: float = 22.0
    """Upper bound of uniformly drawn unit costs."""
    overrides: dict[int, ComponentSpec] = field(default_factory=dict)
    """Per-agent implementations by agent id."""


@dataclass
class ExperimentConfig:
    """Everything needed to run an experiment."""

    name: str = 'experiment'
    """Name used in the run directory."""
    horizon: int = 200
    """Steps per episode."""
    seeds: list[int] = field(default_factory=lambda: [0])
    """Seeds; each seed draws its own costs and network."""
    episodes_per_seed: int = 1
    """Episodes played with the same components per seed."""
    mode: str = 'eval'
    """'eval' freezes components (training=False), 'train' lets them learn."""
    strict: bool = False
    """Crash on the first error in team code instead of falling back."""
    act_seconds_limit: float = 1.0
    """Calls slower than this are counted as slow."""
    market: MarketConfig = field(default_factory=MarketConfig)
    """Market section."""
    finance: FinanceConfig = field(default_factory=FinanceConfig)
    """Finance section."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    """Network section."""
    coalitions: CoalitionsConfig = field(default_factory=CoalitionsConfig)
    """Coalitions section."""
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    """Arbiter section."""
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    """Scoring section."""
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    """Agents section."""


def load_config(path: Path) -> ExperimentConfig:
    """Read and validate an experiment config from YAML."""
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Build and validate an experiment config from plain data."""
    config = _build(cls=ExperimentConfig, data=data, path='config')
    validate_config(config)
    return config


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert a config to plain data for logs."""
    return asdict(config)


def validate_config(config: ExperimentConfig) -> None:
    """Raise ConfigError when values are outside their allowed sets or ranges."""
    checks = [*_choice_checks(config), *_range_checks(config)]
    problems = [message for passed, message in checks if not passed]
    if problems:
        raise ConfigError('; '.join(problems))


def _choice_checks(config: ExperimentConfig) -> list[tuple[bool, str]]:
    return [
        (config.mode in RUN_MODES, f'mode must be one of {sorted(RUN_MODES)}'),
        (
            config.market.share_model in SHARE_MODELS,
            f'market.share_model must be one of {sorted(SHARE_MODELS)}',
        ),
        (
            config.network.topology in TOPOLOGIES,
            f'network.topology must be one of {sorted(TOPOLOGIES)}',
        ),
        (
            config.network.mode in NETWORK_MODES,
            f'network.mode must be one of {sorted(NETWORK_MODES)}',
        ),
    ]


def _range_checks(config: ExperimentConfig) -> list[tuple[bool, str]]:
    market, arbiter, agents = config.market, config.arbiter, config.agents
    return [
        (market.tau > 0, 'market.tau must be positive'),
        (market.demand_b > 0, 'market.demand_b must be positive'),
        (market.p_min > 0, 'market.p_min must be positive'),
        (market.p_min < market.p_max, 'market.p_min must be below market.p_max'),
        (arbiter.max_fine >= 0, 'arbiter.max_fine must be non-negative'),
        (arbiter.min_fine >= 0, 'arbiter.min_fine must be non-negative'),
        (arbiter.max_link_cost_delta > 0, 'arbiter.max_link_cost_delta must be positive'),
        (agents.count >= 2, 'agents.count must be at least 2'),
        (agents.cost_low <= agents.cost_high, 'agents.cost_low must not exceed agents.cost_high'),
        (config.finance.max_negative_steps >= 0, 'finance.max_negative_steps must be non-negative'),
        (config.horizon >= 1, 'horizon must be at least 1'),
    ]


def _build(*, cls: type, data: Any, path: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f'{path} must be a mapping')
    unknown = set(data) - {item.name for item in fields(cls)}
    if unknown:
        raise ConfigError(f'Unknown keys in {path}: {sorted(unknown)}')
    hints = get_type_hints(cls)
    values = {
        name: _convert(value=value, hint=hints[name], path=f'{path}.{name}')
        for name, value in data.items()
    }
    return cls(**values)


def _convert(*, value: Any, hint: Any, path: str) -> Any:
    if is_dataclass(hint):
        return _build(cls=hint, data=value, path=path)
    if get_origin(hint) is dict and get_args(hint)[1] is ComponentSpec:
        return {
            int(key): _build(cls=ComponentSpec, data=item, path=f'{path}.{key}')
            for key, item in (value or {}).items()
        }
    return value
