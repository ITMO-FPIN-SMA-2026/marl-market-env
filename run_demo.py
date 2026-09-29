from pprint import pprint

from social_market_env import (
    ConnectedComponentsCoalitionPolicy,
    EnvironmentConfig,
    RandomMarketAgent,
    SocialMarketEnvironment,
    ThresholdArbiter,
)


def main():
    agent_ids = list(range(5))
    agents = {i: RandomMarketAgent(min_price=20, max_price=80) for i in agent_ids}
    costs = {0: 15.0, 1: 18.0, 2: 20.0, 3: 17.0, 4: 22.0}

    config = EnvironmentConfig(
        demand_a=120.0,
        demand_b=1.0,
        horizon=20,
        initial_edge_probability=0.30,
        seed=42,
    )

    env = SocialMarketEnvironment(
        agents=agents,
        agent_costs=costs,
        coalition_policy=ConnectedComponentsCoalitionPolicy(),
        arbiter=ThresholdArbiter(jump_threshold=4.0, fine=15.0),
        config=config,
    )

    env.reset()
    history = env.run()

    print(f"Completed {len(history)} steps")
    print("Last step:")
    pprint(history[-1])


if __name__ == "__main__":
    main()
