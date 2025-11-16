import numpy as np
import gymnasium as gym
import time

import torch
import multigrid.envs
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
#from PDSAC import PDSACDiscreteAgent, ReplayBufferPDSAC 
from PDSAC_joint_valid import PDSACReinforceAgent, ReplayBufferPDSAC
    
env = gym.make('MultiGrid-EmptyMultiTarget-8x8-v0', agents=3)

num_agents = len(env.observation_space)
state_dim = np.prod(env.observation_space[0]['image'].shape)
action_dim = env.action_space[0].n
print("Num agents:", num_agents, "State dim:", state_dim, "Action dim:", action_dim)
# agents = PDSACDiscreteAgent(num_agents, state_dim, action_dim, "cuda")
# replay_buffer = ReplayBufferPDSAC()
agents = PDSACReinforceAgent(num_agents, state_dim, action_dim)
replay_buffer = ReplayBufferPDSAC(capacity=1_000_000)

episodes = 50000
steps_per_episode = 200
batch_size = 1024
start_steps = 1000
steps_per_update = 100
updates_num = 4

rewards, test_rewards = [], []
total_steps = 0

for ep in range(episodes):
    obs, infos = env.reset()
    #print("Initial obs:", obs)
    obs = np.array([np.array(ob['image'], dtype=np.float32).reshape(-1) for _, ob in obs.items()])
    ep_reward = np.zeros(num_agents)
    for step in range(steps_per_episode):
        total_steps += 1
        actions = []
        if total_steps < start_steps:
            for agent_obs in obs:
                actions.append(env.action_space[0].sample())
        else:
            # st = time.time()
            # for i, policy in enumerate(agents.policies):
            #     ob = obs[i];
            #     action, log_prob = policy.get_action(ob, device="cuda")
            #     actions.append(action.item())
            # obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agents.device)
            actions = agents.select_action(obs, evaluate=False)
            # print("Selected actions:", actions)
            # et = time.time()
            # print("time taken:", et - st)
        input_actions = {id: actions[id] for id in range(len(actions))}
        #if(total_steps % 300 == 0): 
        #print("Actions:", actions)
        next_obs, rewards, terminated, truncated, infos = env.step(input_actions)
        dones = np.array([i_terminated or i_truncated for i_terminated, i_truncated in zip(list(terminated.values()), list(truncated.values()))])
        rewards = [rewards[id] for id in range(len(rewards))]
        done = dones.all() or (step == steps_per_episode - 1)
        reward = sum(rewards)
        # print(dones, reward, rewards)
        #print("Terminated:", terminated, "Truncated:", truncated, "dones:", dones)
        #obs = np.array([np.array(ob['image'], dtype=np.float32).reshape(-1) for _, ob in obs.items()])
        next_obs = np.array([np.array(ob['image'], dtype=np.float32).reshape(-1) for _, ob in next_obs.items()])
        # if(reward > 0): print("obs shape:", obs.shape, "next_obs shape:", next_obs.shape, "actions shape:", actions, "rewards shape:", reward, "dones shape:", done)
        replay_buffer.push(obs, actions, reward, next_obs, done)
        obs = next_obs
        ep_reward += np.array(rewards)
        #print("start print:", obs, actions, np.array(rewards).sum(), next_obs.shape, dones.all())
        if len(replay_buffer) > batch_size and total_steps % steps_per_update == 0:
            for i in range(updates_num):
                st = time.time()
                metrics = agents.update(replay_buffer, batch_size)
                if i == 0:  # Only print first update to avoid clutter
                    print(f"[Ep {ep:4d} | Step {step:3d}] "
                          f"Actor: {metrics['actor_loss']:7.4f} | Critic: {metrics['critic_loss']:7.4f} | "
                          f"Q: {metrics['avg_q']:7.3f} (Q1: {metrics['avg_q1']:6.3f}, Q2: {metrics['avg_q2']:6.3f}) | "
                          f"α: {metrics['alpha']:6.4f} (loss: {metrics['alpha_loss']:7.4f}) | "
                          f"Entropy: {metrics['entropy']:6.4f} | Reward: {metrics['reward']:6.3f}")
                #print("Update time taken:", et - st)
        if dones.all():
            break
    if(ep % 200 == 199): env = gym.make('MultiGrid-EmptyMultiTarget-8x8-v0', agents=3, render_mode="human")
    else: env = gym.make('MultiGrid-EmptyMultiTarget-8x8-v0', agents=3)
    print(f"ep: {ep}, ep_rw: {ep_reward}")