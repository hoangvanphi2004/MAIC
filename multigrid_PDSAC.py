import numpy as np
import gymnasium as gym
import time
import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import torch
import multigrid.envs
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
#from PDSAC import PDSACDiscreteAgent, ReplayBufferPDSAC 
from PDSAC_valid import PDSAC, ReplayBuffer


def smooth_curve(values, weight=0.9):
    """Exponential moving average smoothing."""
    if not values:
        return []
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def save_training_plots(metrics, plots_dir, episode, final=False):
    """Save training plots using matplotlib."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'PDSAC Training Metrics (Episode {episode})', fontsize=16)
    
    # Plot 1: Actor Loss
    ax1 = axes[0, 0]
    if metrics['actor_losses']:
        ax1.plot(metrics['steps'], metrics['actor_losses'], 
                alpha=0.5, label='Actor Loss', color='blue')
        ax1.plot(metrics['steps'], smooth_curve(metrics['actor_losses']), 
                label='Actor Loss (smoothed)', linewidth=2, color='darkblue')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Actor Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Critic Loss
    ax2 = axes[0, 1]
    if metrics['critic_losses']:
        ax2.plot(metrics['steps'], metrics['critic_losses'], 
                alpha=0.5, label='Critic Loss', color='red')
        ax2.plot(metrics['steps'], smooth_curve(metrics['critic_losses']), 
                label='Critic Loss (smoothed)', linewidth=2, color='darkred')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Critic Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Q Values
    ax3 = axes[0, 2]
    if metrics['q_values']:
        ax3.plot(metrics['steps'], metrics['q_values'], 
                alpha=0.5, label='Q Value', color='green')
        ax3.plot(metrics['steps'], smooth_curve(metrics['q_values']), 
                label='Q Value (smoothed)', linewidth=2, color='darkgreen')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Q Value')
    ax3.set_title('Average Q Values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Alpha (Temperature)
    ax4 = axes[1, 0]
    if metrics['alpha_values']:
        ax4.plot(metrics['steps'], metrics['alpha_values'], 
                alpha=0.5, label='Alpha', color='purple')
        ax4.plot(metrics['steps'], smooth_curve(metrics['alpha_values']), 
                label='Alpha (smoothed)', linewidth=2, color='darkviolet')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Alpha')
    ax4.set_title('Temperature Parameter (α)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Entropy
    ax5 = axes[1, 1]
    if metrics['entropy_values']:
        ax5.plot(metrics['steps'], metrics['entropy_values'], 
                alpha=0.5, label='Entropy', color='orange')
        ax5.plot(metrics['steps'], smooth_curve(metrics['entropy_values']), 
                label='Entropy (smoothed)', linewidth=2, color='darkorange')
    ax5.set_xlabel('Training Steps')
    ax5.set_ylabel('Entropy')
    ax5.set_title('Policy Entropy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Episode Rewards
    ax6 = axes[1, 2]
    if metrics['episode_rewards']:
        episodes = np.array(metrics['episodes'])
        rewards = np.array(metrics['episode_rewards'])
        
        ax6.plot(episodes, rewards, alpha=0.3, color='green', label='Episode Reward')
        ax6.plot(episodes, smooth_curve(rewards.tolist()), 
                label='Reward (smoothed)', linewidth=2, color='darkgreen')
        
        # Add trend line
        if len(episodes) > 10:
            z = np.polyfit(episodes, rewards, 1)
            p = np.poly1d(z)
            ax6.plot(episodes, p(episodes), "--", 
                    label=f'Trend (slope={z[0]:.4f})', 
                    color='red', linewidth=2)
    
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Total Reward')
    ax6.set_title('Episode Rewards')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'training_final.png' if final else f'training_ep{episode}.png'
    save_path = plots_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_training_summary(metrics):
    """Print training statistics summary."""
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    
    if metrics['episode_rewards']:
        rewards = np.array(metrics['episode_rewards'])
        print(f"\nEpisode Rewards:")
        print(f"  Initial: {rewards[0]:.3f}")
        print(f"  Final: {rewards[-1]:.3f}")
        print(f"  Max: {rewards.max():.3f}")
        print(f"  Mean: {rewards.mean():.3f} ± {rewards.std():.3f}")
    
    if metrics['actor_losses']:
        losses = np.array(metrics['actor_losses'])
        print(f"\nActor Loss:")
        print(f"  Initial: {losses[0]:.4f}")
        print(f"  Final: {losses[-1]:.4f}")
        print(f"  Mean: {losses.mean():.4f} ± {losses.std():.4f}")
    
    if metrics['critic_losses']:
        losses = np.array(metrics['critic_losses'])
        print(f"\nCritic Loss:")
        print(f"  Initial: {losses[0]:.4f}")
        print(f"  Final: {losses[-1]:.4f}")
        print(f"  Mean: {losses.mean():.4f} ± {losses.std():.4f}")
    
    if metrics['q_values']:
        q_vals = np.array(metrics['q_values'])
        print(f"\nQ Values:")
        print(f"  Initial: {q_vals[0]:.3f}")
        print(f"  Final: {q_vals[-1]:.3f}")
        print(f"  Mean: {q_vals.mean():.3f} ± {q_vals.std():.3f}")
    
    if metrics['entropy_values']:
        entropy = np.array(metrics['entropy_values'])
        print(f"\nEntropy:")
        print(f"  Initial: {entropy[0]:.4f}")
        print(f"  Final: {entropy[-1]:.4f}")
        print(f"  Mean: {entropy.mean():.4f} ± {entropy.std():.4f}")
    
    print("="*60)


# Create directories for saving plots and models
run_dir = Path('./runs/pdsac_multitarget')
plots_dir = run_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

# Initialize tracking metrics
training_metrics = {
    'actor_losses': [],
    'critic_losses': [],
    'q_values': [],
    'alpha_values': [],
    'entropy_values': [],
    'episode_rewards': [],
    'episodes': [],
    'steps': []
}
    
env = gym.make('MultiGrid-MultiTargetEmpty-8x8-v0', num_agents=3)

num_agents = len(env.observation_space)
state_dim = np.prod(env.observation_space[0]['image'].shape)
action_dim = env.action_space[0].n
print("Num agents:", num_agents, "State dim:", state_dim, "Action dim:", action_dim)
# agents = PDSACDiscreteAgent(num_agents, state_dim, action_dim, "cuda")
# replay_buffer = ReplayBufferPDSAC()
agents = PDSAC(num_agents, state_dim, action_dim)
replay_buffer = ReplayBuffer(capacity=1_000_000)

episodes = 50000
steps_per_episode = 40
batch_size = 2048
start_steps = 1000
steps_per_update = 100
updates_num = 4
plot_interval = 1000  # Save plots every N episodes

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
            # Track metrics for this update cycle
            cycle_metrics = {
                'actor_loss': [],
                'critic_loss': [],
                'avg_q': [],
                'alpha': [],
                'entropy': []
            }
            
            for i in range(updates_num):
                st = time.time()
                metrics = agents.update(replay_buffer, batch_size)
                
                # Collect metrics
                cycle_metrics['actor_loss'].append(metrics['actor_loss'])
                cycle_metrics['critic_loss'].append(metrics['critic_loss'])
                cycle_metrics['avg_q'].append(metrics['avg_q'])
                cycle_metrics['alpha'].append(metrics['alpha'])
                cycle_metrics['entropy'].append(metrics['entropy'])
                
                if i == 0:  # Only print first update to avoid clutter
                    print(f"[Ep {ep:4d} | Step {step:3d}] "
                          f"Actor: {metrics['actor_loss']:7.4f} | Critic: {metrics['critic_loss']:7.4f} | "
                          f"Q: {metrics['avg_q']:7.3f} (Q1: {metrics['avg_q1']:6.3f}, Q2: {metrics['avg_q2']:6.3f}) | "
                          f"α: {metrics['alpha']:6.4f} (loss: {metrics['alpha_loss']:7.4f}) | "
                          f"Entropy: {metrics['entropy']:6.4f} | Reward: {metrics['reward']:6.3f}")
                #print("Update time taken:", et - st)
            
            # Store average metrics for plotting
            training_metrics['actor_losses'].append(np.mean(cycle_metrics['actor_loss']))
            training_metrics['critic_losses'].append(np.mean(cycle_metrics['critic_loss']))
            training_metrics['q_values'].append(np.mean(cycle_metrics['avg_q']))
            training_metrics['alpha_values'].append(np.mean(cycle_metrics['alpha']))
            training_metrics['entropy_values'].append(np.mean(cycle_metrics['entropy']))
            training_metrics['steps'].append(total_steps)
        if dones.all():
            break
    
    # Store episode metrics
    training_metrics['episodes'].append(ep)
    training_metrics['episode_rewards'].append(ep_reward.sum())
    
    # Save plots periodically
    if ep > 0 and ep % plot_interval == 0:
        save_training_plots(training_metrics, plots_dir, ep)
        print(f"Plots saved to {plots_dir}")
    
    # if(ep % 200 == 0): env = gym.make('MultiGrid-MultiTargetEmpty-8x8-TurnForward-v0', num_agents=1, render_mode="human")
    # else: env = gym.make('MultiGrid-MultiTargetEmpty-8x8-TurnForward-v0', num_agents=1)
    print(f"ep: {ep}, ep_rw: {ep_reward}")

# Save final plots
print("\nSaving final training plots...")
save_training_plots(training_metrics, plots_dir, episodes, final=True)

rewards_path = run_dir / 'episode_rewards.json'
with open(rewards_path, 'w') as f:
    json.dump({
        'episodes': training_metrics['episodes'],
        'rewards': training_metrics['episode_rewards']
    }, f, indent=2)
print(f"Rewards saved to {rewards_path}")

print(f"Final plots saved to {plots_dir}")
print_training_summary(training_metrics)
