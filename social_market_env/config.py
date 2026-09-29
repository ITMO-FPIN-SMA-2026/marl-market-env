from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    # Market
    demand_a: float = 100.0
    demand_b: float = 1.0
    min_price: float = 1.0
    max_price: float = 100.0

    # Episode
    horizon: int = 100
    seed: int = 42

    # Network evolution (MVP heuristic; replaceable later)
    initial_edge_probability: float = 0.25
    base_link_probability: float = 0.10
    similarity_weight: float = 0.25
    history_weight: float = 0.25
    utility_weight: float = 0.20
    coalition_weight: float = 0.20
    arbiter_link_cost_weight: float = 0.20

    # Financial reserve / elimination
    initial_balance: float = 1000.0
    min_allowed_balance: float = -250.0
    max_negative_steps: int = 3

    # Coalition reward distribution: MVP supports equal split only.
    coalition_bonus_scale: float = 0.05

    # Arbiter reward (matches the structure from the assignment)
    target_social_welfare: float = 0.0
    welfare_deviation_penalty: float = 1.0
    intervention_penalty: float = 0.1
    false_positive_penalty: float = 0.5
    false_negative_penalty: float = 0.5

    # A sanction is considered "correct" in this MVP if the target belongs to
    # a coalition that raises its average price above the market average.
    harmful_coalition_margin: float = 0.05
