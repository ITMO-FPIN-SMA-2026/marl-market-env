"""Recipient-learning curriculum before training a welfare-oriented regulator."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .agents import load_agents,load_auxiliary,save_agents
from .environment import ArbiterAction,MarketEnv
from .learn_arbiter import record_weights,write_json
from .train import run_episode


class SanctionExploration:
    """Training-only randomized fiscal exposure, independent of prices/labels."""
    def __init__(self,maximum):
        self.maximum=maximum

    def reset(self,*,seed,training=False):
        self.rng=np.random.default_rng(seed+3000000)
        self.last=-3
        self.paid=0.

    def __call__(self,public):
        if public.step-self.last<3 or self.paid+ self.maximum>1000 or self.rng.random()>.5:
            return ArbiterAction()
        active=[i for i,p in enumerate(public.prices) if p is not None]
        if not active:
            return ArbiterAction()
        target=int(self.rng.choice(active))
        amount=float(self.rng.choice([.4*self.maximum,self.maximum]))
        self.paid+=amount
        self.last=public.step
        return ArbiterAction("sanction",amount,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=13000000)
    args=parser.parse_args()
    if args.episodes<1:
        parser.error("episodes must be positive")
    config,agents=load_agents(args.market)
    coalition,solo=load_auxiliary(args.market)
    if coalition is None or solo is None:
        parser.error("The input snapshot must include coalition and solo policies")
    args.out.mkdir(parents=True,exist_ok=False)
    stage=args.out/"weights"/"recipient_curriculum"
    stage.mkdir(parents=True)
    initial=stage/"episode_00000.json"
    save_agents(initial,agents,config,coalition,solo)
    record_weights(stage,(initial,),stage=stage.name,episode=0,seed=args.seed)
    write_json(args.out/"curriculum.json",{"market":asdict(config),"seed":args.seed,"episodes":args.episodes,
        "fine_maximum":25.,"ramp_episodes":200,"teacher":"price-independent randomized sanctions",
        "cooldown":3,"episode_budget":1000.,"agent_epsilon_scale":.5})
    rows=[]
    for episode in range(args.episodes):
        maximum=25.*min(1.,(episode+1)/200.)
        for is_solo,population in ((True,solo),(False,agents)):
            metrics=run_episode(MarketEnv(config),population,args.seed+episode,training=True,epsilon_scale=.5,
                                coalition_policy=None if is_solo else coalition,solo=is_solo,
                                arbiter=SanctionExploration(maximum))
        rows.append({"episode":episode+1,**metrics})
        if (episode+1)%100==0 or episode+1==args.episodes:
            path=stage/f"episode_{episode+1:05d}.json"
            save_agents(path,agents,config,coalition,solo)
            record_weights(stage,(path,),stage=stage.name,episode=episode+1,seed=args.seed,metrics=metrics)
            print(f"Curriculum episode={episode+1} SW={metrics['social_welfare']:.2f} "
                  f"survivors={metrics['survivors']} fine-states="
                  f"{[sum(state[-1]>0 for state in a.q) for a in agents]}",flush=True)
            write_json(args.out/"training.json",rows)
    save_agents(args.out/"agents.json",agents,config,coalition,solo)


if __name__=="__main__":
    main()
