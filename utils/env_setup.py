import gymnasium as gym
import multigrid.envs

def make_env(env_id, num_agents, steps_per_episode, record_video):
    if record_video:
        try:
            env = gym.make(env_id, num_agents=num_agents, render_mode='rgb_array', max_steps=steps_per_episode)
        except TypeError:
            env = gym.make(env_id, num_agents=num_agents, max_steps=steps_per_episode)
    else:
        env = gym.make(env_id, num_agents=num_agents, max_steps=steps_per_episode)
    return env

def get_obs_state_action_dims(env):
    obs0, infos = env.reset()
    first_obs = list(obs0.values())[0]
    if isinstance(first_obs, dict) and 'image' in first_obs:
        obs_shape = first_obs['image'].shape
    else:
        obs_shape = env.observation_space[0]['image'].shape
        
    state_shape = infos['state'].shape
    action_dim = env.action_space[0].n
    return obs_shape, state_shape, action_dim
