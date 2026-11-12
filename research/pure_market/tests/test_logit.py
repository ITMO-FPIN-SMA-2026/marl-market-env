from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.audit_market import audit
from market_rl import MarketConfig, MarketEnv
from market_rl.coalitions import clear_market
from market_rl.sanction import LearnedArbiter
from market_rl.arbiter_scoring import harmful_groups


class LogitTests(unittest.TestCase):
    def test_outside_option_accounting_and_consumer_surplus_derivative(self):
        prices, costs = np.array([4., 4., 6., 6.]), np.array([1., 2., 3., 4.])
        active = np.ones(4, dtype=bool)
        def clear(p, mask=active):
            return clear_market(p, costs, mask, 100., 5., model="logit")
        q, profit, mean, cs = clear(prices)
        self.assertGreater(q.sum(), 0)
        self.assertLess(q.sum(), 100)
        self.assertAlmostEqual(prices @ q, mean*q.sum())
        np.testing.assert_allclose(profit, (prices-costs)*q)
        for i in range(4):
            plus, minus = prices.copy(), prices.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            derivative = (clear(plus)[3]-clear(minus)[3])/2e-5
            self.assertAlmostEqual(derivative, -q[i], places=6)
        self.assertLess(clear(prices+4)[0].sum(), q.sum())
        self.assertLess(clear(prices+4)[3], cs)
        self.assertEqual(clear(prices, np.zeros(4,bool))[0].sum(),0)
        self.assertEqual(clear(prices, np.zeros(4,bool))[3],0)
        self.assertLessEqual(cs+profit.sum(),MarketConfig(demand_model="logit").welfare_bound)

    def test_competitive_equilibrium_differs_from_cartel(self):
        result = audit(MarketConfig(demand_model="logit"))
        self.assertEqual(len(result["pure_nash"]),1)
        nash = result["pure_nash"][0]
        cartel = result["coalition_optimum"]
        self.assertEqual(nash["prices"],[4.,4.,6.,6.])
        self.assertEqual(cartel["price"],8.)
        self.assertGreater(cartel["joint_profit"],sum(nash["profits"]))
        self.assertLess(cartel["welfare"],nash["welfare"])

    def test_default_mode_unchanged_and_checkpoint_cannot_cross_modes(self):
        default = MarketEnv(MarketConfig(dynamic_network=False))
        explicit = MarketEnv(MarketConfig(dynamic_network=False,demand_model="assignment"))
        for env in (default,explicit):
            env.reset(12)
        self.assertEqual(default.market_step([0,5,10,15]),explicit.market_step([0,5,10,15]))
        logit = MarketConfig(demand_model="logit")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"arbiter.json"
            LearnedArbiter(MarketConfig()).save(path)
            with self.assertRaises(ValueError):
                LearnedArbiter.load(path,logit)
            LearnedArbiter(logit).save(path)
            with self.assertRaises(ValueError):
                LearnedArbiter.load(path,replace(logit,logit_temperature=2.))
            with self.assertRaises(ValueError):
                LearnedArbiter.load(path,replace(logit,harm_baseline="proposals"))

    def test_logit_harm_uses_competition_not_unexecuted_proposals(self):
        config = MarketConfig(demand_model="logit",dynamic_network=False)
        outcomes=[]
        for proposal in (1,26):
            env=MarketEnv(config)
            env.reset(1)
            public=env.market_step([proposal]*4,{frozenset(range(4)):3})
            _,rewards,_,info=env.settle()
            outcomes.append((public,info["counterfactuals"]))
            self.assertEqual(harmful_groups(public,info),[(0,1,2,3)])
            self.assertAlmostEqual(rewards.sum(),440.)
        self.assertEqual(outcomes[0],outcomes[1])
        env=MarketEnv(config)
        env.reset(1)
        public=env.market_step([15]*4)  # same high price, independently
        _,_,_,info=env.settle()
        self.assertEqual(public,outcomes[0][0])
        self.assertEqual(harmful_groups(public,info),[])


if __name__ == "__main__":
    unittest.main()
