from statistics import mean

from georgii_agent import QLearningMarketAgent
from social_market_env import (
    ConnectedComponentsCoalitionPolicy,
    EnvironmentConfig,
    RandomMarketAgent,
    SocialMarketEnvironment,
    ThresholdArbiter,
)

COSTS = {0: 15.0, 1: 18.0, 2: 20.0, 3: 17.0, 4: 22.0}
GEORGII_ID = 0


def make_env(georgii, seed):
    agents = {i: RandomMarketAgent(min_price=20, max_price=80) for i in COSTS}
    agents[GEORGII_ID] = georgii
    config = EnvironmentConfig(demand_a=120.0, demand_b=1.0, horizon=50,
                               initial_edge_probability=0.30, seed=seed)
    return SocialMarketEnvironment(
        agents=agents,
        agent_costs=COSTS,
        coalition_policy=ConnectedComponentsCoalitionPolicy(),
        arbiter=ThresholdArbiter(jump_threshold=4.0, fine=15.0),
        config=config,
    )


def run_episode(agent, seed):
    env = make_env(agent, seed)
    env.reset()
    history = env.run()
    my_total = sum(h["agent_rewards"][GEORGII_ID] for h in history)
    others = mean(sum(h["agent_rewards"][i] for h in history) for i in COSTS if i != GEORGII_ID)
    my_avg_price = mean(h["prices"][GEORGII_ID] for h in history if GEORGII_ID in h["prices"])
    fines = sum(1 for h in history if h["arbiter_action"]["target_agent"] == GEORGII_ID)
    return my_total, others, my_avg_price, fines


def main():
    agent = QLearningMarketAgent()

    print("=== Обучение (500 эпизодов по 50 шагов) ===")
    chunk = []
    for ep in range(1, 501):
        chunk.append(run_episode(agent, seed=ep)[0])
        if ep % 100 == 0:
            print(f"эпизоды {ep-99:>3}-{ep:<3}  средняя награда за эпизод: {mean(chunk):8.1f}"
                  f"   epsilon={agent.epsilon:.2f}")
            chunk = []

    print("\n=== Проверка на 100 новых эпизодах (без исследования) ===")
    agent.training = False
    test_seeds = range(10_000, 10_100)
    q = [run_episode(agent, s) for s in test_seeds]
    r = [run_episode(RandomMarketAgent(20, 80), s) for s in test_seeds]
    print(f"Агент Георгия:     награда {mean(x[0] for x in q):8.1f}  | средняя цена {mean(x[2] for x in q):5.1f}"
          f" | штрафов за эпизод {mean(x[3] for x in q):.2f}")
    print(f"Случайный агент:   награда {mean(x[0] for x in r):8.1f}  | средняя цена {mean(x[2] for x in r):5.1f}"
          f" | штрафов за эпизод {mean(x[3] for x in r):.2f}")
    print(f"(остальные 4 случайных агента в среднем получают {mean(x[1] for x in q):.1f})")

    print("\nВыученная политика (состояние -> наценка к себестоимости):")
    print("  состояние = (цена vs рынок: 0 дешевле/1 рядом/2 дороже, прибыль 0/1, деньги 0/1, соседи 0/1)")
    for s, m in sorted(agent.greedy_policy().items()):
        print(f"  {s} -> x{m}  (цена {COSTS[GEORGII_ID]*m:.1f})")


if __name__ == "__main__":
    main()
