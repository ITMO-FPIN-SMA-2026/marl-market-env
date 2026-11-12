"""Exact finite-grid incentive audit; no learning or hidden-label approximations."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from market_rl import MarketConfig
from market_rl.coalitions import clear_market


def audit(config):
    prices, costs = np.array(config.prices), np.array(config.costs)
    size = len(prices)
    profits = np.zeros((size,)*4 + (4,))
    welfare = np.zeros((size,)*4)
    for choices in product(range(size), repeat=4):
        _, reward, _, cs = clear_market(prices[list(choices)], costs, np.ones(4, bool),
                                        config.demand_a, config.demand_b, model=config.demand_model,
                                        logit_value=config.logit_value, logit_temperature=config.logit_temperature)
        profits[choices] = reward
        welfare[choices] = cs + reward.sum()
    equilibria = []
    for choices in product(range(size), repeat=4):
        stable = True
        for player in range(4):
            for price in range(size):
                alternative = list(choices)
                alternative[player] = price
                if profits[tuple(alternative)][player] > profits[choices][player] + 1e-9:
                    stable = False
                    break
            if not stable:
                break
        if stable:
            equilibria.append({"prices": prices[list(choices)].tolist(),
                               "profits": profits[choices].tolist(), "welfare": float(welfare[choices])})
    coordinated = [{"price": float(price), "joint_profit": float(profits[(index,)*4].sum()),
                    "welfare": float(welfare[(index,)*4])} for index, price in enumerate(prices)]
    return {"profiles_enumerated": size**4, "pure_nash": equilibria, "common_prices": coordinated,
            "coalition_optimum": max(coordinated, key=lambda r: r["joint_profit"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/market_incentive_audit.json"))
    parser.add_argument("--demand-model", choices=["assignment", "logit"], default="assignment")
    args = parser.parse_args()
    report = audit(MarketConfig(demand_model=args.demand_model))
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
