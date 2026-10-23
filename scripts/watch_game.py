"""Watch one GRF match in a window, in real time. Our one controlled player acts randomly; the built-in AI plays everyone else.

Usage:  python scripts\\watch_game.py --seed 5 --steps 600
"""
import argparse
import numpy as np
import gfootball.env as football_env

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="11_vs_11_stochastic")
ap.add_argument("--steps", type=int, default=600)   # 600 steps = 1/5 of a full 3000-step match
ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()

env = football_env.create_environment(
    env_name=a.scenario, representation="raw", render=True,
    number_of_left_players_agent_controls=1,
    other_config_options={"game_engine_random_seed": a.seed, "real_time": True})
obs = env.reset()
rng = np.random.default_rng(a.seed)
for step in range(a.steps):
    obs, reward, done, info = env.step(int(rng.integers(19)))
    if done:
        break
print("finished after", step + 1, "steps, score", obs[0]["score"])
env.close()