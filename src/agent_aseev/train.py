import sys
from statistics import mean

from social_market_env.config import load_config
from social_market_env.runner.experiment import build_environment, episode_seed

from agent_aseev import DEFAULT_CHECKPOINT, GeorgiiAgent


def main(config_path: str, episodes: int = 200, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
    config = load_config(config_path)
    agent = GeorgiiAgent(checkpoint=None)
    chunk = []
    for i, seed in enumerate(range(1000, 1000 + episodes), 1):
        env = build_environment(config=config, seed=seed)
        env.agents[0] = agent
        env.reset(seed=episode_seed(seed=seed, episode=0), training=True)
        total = 0.0
        while not env.done:
            total += env.step().record["agent_rewards"].get(0, 0.0)
        chunk.append(total)
        if i % 10 == 0:
            agent.save(checkpoint)
            print(f"эпизод {i:4d}  средняя награда {mean(chunk):8.0f}  epsilon {agent.epsilon:.2f}", flush=True)
            chunk = []
    agent.save(checkpoint)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
