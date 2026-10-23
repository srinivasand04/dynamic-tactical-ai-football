import time, numpy as np, gfootball.env as football_env
env = football_env.create_environment(
    env_name="11_vs_11_stochastic", representation="raw", render=False,
    number_of_left_players_agent_controls=1, write_full_episode_dumps=False)
obs = env.reset()
o = obs[0]
print("keys:", sorted(o.keys())[:12], "...")
print("left_team shape:", np.array(o["left_team"]).shape, "ball:", o["ball"], "ball_owned_team:", o["ball_owned_team"], "game_mode:", o["game_mode"])
t=time.time(); steps=0; done=False
while not done and steps < 300:
    obs, r, done, info = env.step(env.action_space.sample()); steps += 1
print(f"{steps} steps in {time.time()-t:.1f}s  score={obs[0]['score']}")
env.close()
# multi-agent: control all 11 left players, simple115v2 representation
env = football_env.create_environment(env_name="11_vs_11_stochastic", representation="simple115v2",
    number_of_left_players_agent_controls=11, render=False)
obs = env.reset(); print("multi-agent obs shape:", obs.shape, "action space:", env.action_space)
obs, r, done, info = env.step(env.action_space.sample()); print("step ok, reward shape:", np.shape(r))
env.close()
env = football_env.create_environment(env_name="academy_3_vs_1_with_keeper", representation="simple115v2",
    number_of_left_players_agent_controls=3, render=False)
obs=env.reset(); print("academy 3v1 obs:", obs.shape); env.close()
