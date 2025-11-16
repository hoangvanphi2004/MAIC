import numpy as np
from utils.make_env import make_env
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from PDSAC import PDSACDiscreteAgent, ReplayBufferPDSAC 

def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
    
env = make_parallel_env("fullobs_collect_treasure", 1, 1)
#env = make_parallel_env("multi_speaker_listener", 1, 1)

num_agents = len(env.observation_space)
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
agents = PDSACDiscreteAgent(num_agents, state_dim, action_dim, "cuda")
replay_buffer = ReplayBufferPDSAC()


episodes = 200
steps_per_episode = 200
batch_size = 64
start_steps = 1000

rewards, test_rewards = [], []
total_steps = 0

for ep in range(episodes):
    obs = env.reset()[0]
    ep_reward = 0
    for step in range(steps_per_episode):
        
        total_steps += 1
        actions = []
        if total_steps < start_steps:
            for agent_obs in obs:
                actions.append(env.action_space[0].sample())
        else:
            for i, policy in enumerate(agents.policies):
                ob = np.array(obs, dtype=np.float32)[i];
                actions.append(policy.get_action(ob, device="cuda").item())
        print("Actions:", actions)
        next_obs, rewards, dones, infos = env.step([actions])
        next_obs = next_obs[0];
        replay_buffer.push(np.array(obs, dtype=np.float64), actions, np.array(rewards).sum(), np.array(next_obs, dtype=np.float64), dones.all())
        obs = next_obs
        ep_reward +=  np.array(rewards).sum()
        #print("start print:", obs, actions, np.array(rewards).sum(), next_obs.shape, dones.all())
        if len(replay_buffer) > batch_size:
            agents.update(replay_buffer, batch_size)

        #env.envs[0].render(mode="human")
        if dones.all():
            break
    print(f"ep: {ep}, ep_rw: {ep_reward}")