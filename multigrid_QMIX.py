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
from QMIX_simple import QMIXAgent


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
    fig.suptitle(f'QMIX Training Metrics (Episode {episode})', fontsize=16)
    
    # Plot 1: Q Loss
    ax1 = axes[0, 0]
    if metrics['q_losses'] and metrics['training_steps']:
        ax1.plot(metrics['training_steps'], metrics['q_losses'], 
                alpha=0.5, label='Q Loss', color='blue')
        ax1.plot(metrics['training_steps'], smooth_curve(metrics['q_losses']), 
                label='Q Loss (smoothed)', linewidth=2, color='darkblue')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Q Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Q Values
    ax2 = axes[0, 1]
    if metrics['q_values'] and metrics['training_steps']:
        ax2.plot(metrics['training_steps'], metrics['q_values'], 
                alpha=0.5, label='Q Value', color='red')
        ax2.plot(metrics['training_steps'], smooth_curve(metrics['q_values']), 
                label='Q Value (smoothed)', linewidth=2, color='darkred')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Q Value')
    ax2.set_title('Average Q Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon Decay
    ax3 = axes[0, 2]
    if metrics['epsilon_values'] and metrics['steps']:
        ax3.plot(metrics['steps'], metrics['epsilon_values'], 
                alpha=0.5, label='Epsilon', color='green')
        ax3.plot(metrics['steps'], smooth_curve(metrics['epsilon_values']), 
                label='Epsilon (smoothed)', linewidth=2, color='darkgreen')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (ε)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Episode Rewards
    ax4 = axes[1, 0]
    if metrics['episode_rewards']:
        episodes = np.array(metrics['episodes'])
        rewards = np.array(metrics['episode_rewards'])
        
        ax4.plot(episodes, rewards, alpha=0.3, color='purple', label='Episode Reward')
        ax4.plot(episodes, smooth_curve(rewards.tolist()), 
                label='Reward (smoothed)', linewidth=2, color='darkviolet')
        
        # Add trend line
        if len(episodes) > 10:
            z = np.polyfit(episodes, rewards, 1)
            p = np.poly1d(z)
            ax4.plot(episodes, p(episodes), "--", 
                    label=f'Trend (slope={z[0]:.4f})', 
                    color='red', linewidth=2)
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Episode Rewards')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Training Steps per Episode
    ax5 = axes[1, 1]
    if metrics['episode_lengths']:
        episodes = np.array(metrics['episodes'])
        lengths = np.array(metrics['episode_lengths'])
        
        ax5.plot(episodes, lengths, alpha=0.3, color='orange', label='Episode Length')
        ax5.plot(episodes, smooth_curve(lengths.tolist()), 
                label='Length (smoothed)', linewidth=2, color='darkorange')
    
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Steps')
    ax5.set_title('Episode Length')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Win Rate
    ax6 = axes[1, 2]
    if metrics['win_rate']:
        episodes = np.array(metrics['episodes'][-len(metrics['win_rate']):])
        win_rates = np.array(metrics['win_rate'])
        
        ax6.plot(episodes, win_rates, alpha=0.5, color='cyan', label='Win Rate')
        ax6.plot(episodes, smooth_curve(win_rates.tolist()), 
                label='Win Rate (smoothed)', linewidth=2, color='darkcyan')
    
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Win Rate')
    ax6.set_title('Success Rate')
    ax6.set_ylim([0, 1.05])
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'training_final.png' if final else f'training_ep{episode}.png'
    save_path = plots_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def print_training_summary(metrics):
    """Print training statistics summary."""
    print("\n" + "="*60)
    print("QMIX TRAINING STATISTICS")
    print("="*60)
    
    if metrics['episode_rewards']:
        rewards = np.array(metrics['episode_rewards'])
        print(f"\nEpisode Rewards:")
        print(f"  Initial: {rewards[0]:.3f}")
        print(f"  Final: {rewards[-1]:.3f}")
        print(f"  Max: {rewards.max():.3f}")
        print(f"  Mean: {rewards.mean():.3f} ± {rewards.std():.3f}")
    
    if metrics['q_losses']:
        losses = np.array(metrics['q_losses'])
        print(f"\nQ Loss:")
        print(f"  Initial: {losses[0]:.4f}")
        print(f"  Final: {losses[-1]:.4f}")
        print(f"  Mean: {losses.mean():.4f} ± {losses.std():.4f}")
    
    if metrics['q_values']:
        q_vals = np.array(metrics['q_values'])
        print(f"\nQ Values:")
        print(f"  Initial: {q_vals[0]:.3f}")
        print(f"  Final: {q_vals[-1]:.3f}")
        print(f"  Mean: {q_vals.mean():.3f} ± {q_vals.std():.3f}")
    
    if metrics['win_rate']:
        win_rate = np.array(metrics['win_rate'])
        print(f"\nWin Rate:")
        print(f"  Initial: {win_rate[0]:.4f}")
        print(f"  Final: {win_rate[-1]:.4f}")
        print(f"  Mean: {win_rate.mean():.4f}")
    
    print("="*60)


def run_qmix_training(
    episodes=50000,
    steps_per_episode=40,
    train_start=1000,
    update_interval=100,
    plot_interval=1000,
    save_interval=5000,
    env_id='MultiGrid-MultiTargetEmpty-8x8-v0',
    num_agents=3,
):
    """Main training function for QMIX."""
    
    # Create directories for saving plots and models
    run_dir = Path('./runs/qmix_multitarget')
    plots_dir = run_dir / 'plots'
    models_dir = run_dir / 'models'
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tracking metrics
    training_metrics = {
        'q_losses': [],
        'q_values': [],
        'epsilon_values': [],
        'episode_rewards': [],
        'episode_lengths': [],
        'win_rate': [],
        'episodes': [],
        'steps': [],  # For per-episode tracking
        'training_steps': []  # For per-training-step tracking (loss, q_values)
    }

    # Environment setup
    env = gym.make(env_id, num_agents=num_agents)
    num_agents = len(env.observation_space)
    single_obs_dim = np.prod(env.observation_space[0]['image'].shape)
    state_dim = single_obs_dim * num_agents  # Full state is all agents' obs concatenated
    action_dim = env.action_space[0].n
    print("Num agents:", num_agents, "Single obs dim:", single_obs_dim, "State dim:", state_dim, "Action dim:", action_dim)

    # QMIX Agent
    # observation_dim = state_dim + num_agents (agent ID concatenated)
    # single_obs_dim is per-agent observation (with agent ID), state is full global state
    agent = QMIXAgent(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        observation_dim=single_obs_dim + num_agents,
        num_episodes=episodes
    )

    total_steps = 0
    win_counter = 0
    win_window = 100

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        obs = np.array([obs_dict[i]['image'].astype(np.float32).reshape(-1) for i in range(num_agents)])
        
        # Prepare initial observations with agent ID
        obs_with_id = []
        for i, o in enumerate(obs):
            agent_id = np.zeros(num_agents)
            agent_id[i] = 1
            obs_with_id.append(np.concatenate([o, agent_id]))
        obs_with_id = np.array(obs_with_id)
        
        # Initialize hidden states for DRQN
        # Not needed for simplified agent
        
        ep_reward = 0.0
        ep_length = 0
        
        for step in range(steps_per_episode):
            total_steps += 1
            
            # Get available actions (for MultiGrid, all actions are available)
            available_actions = np.ones((num_agents, action_dim))
            
            # Select actions
            actions = agent.select_action(available_actions, obs_with_id)
            
            # Step environment
            input_actions = {i: int(actions[i]) for i in range(num_agents)}
            next_obs_dict, rewards_dict, terminated, truncated, info = env.step(input_actions)
            dones = np.array([terminated[i] or truncated[i] for i in range(num_agents)])
            
            # Process observations
            next_obs = np.array([next_obs_dict[i]['image'].astype(np.float32).reshape(-1) for i in range(num_agents)])
            next_obs_with_id = []
            for i, o in enumerate(next_obs):
                agent_id = np.zeros(num_agents)
                agent_id[i] = 1
                next_obs_with_id.append(np.concatenate([o, agent_id]))
            next_obs_with_id = np.array(next_obs_with_id)
            
            # Get rewards
            reward = float(sum(rewards_dict.values()))
            done = dones.all() or (step == steps_per_episode - 1)
            
            # Construct full state from obs_dict
            state = np.concatenate([obs_dict[i]['image'].astype(np.float32).reshape(-1) for i in range(num_agents)])
            next_state = np.concatenate([next_obs_dict[i]['image'].astype(np.float32).reshape(-1) for i in range(num_agents)])
            
            # Push transition to buffer
            agent.buffer.push(state, actions, reward, next_state, float(done))
            
            obs_dict = next_obs_dict
            obs = next_obs
            obs_with_id = next_obs_with_id
            ep_reward += reward
            ep_length += 1
            
            if total_steps > train_start and total_steps % update_interval == 0:
                for _ in range(4):
                    train_info = agent.train()
                    if train_info:
                        training_metrics['q_losses'].append(train_info['loss'])
                        training_metrics['q_values'].append(train_info['q_value'])
                        training_metrics['training_steps'].append(total_steps)
                
                if total_steps % 500 == 0:
                    agent.update_target()
            
            if done:
                break
        
        # No need to push episode anymore
        
        # Check for win (all agents reached targets)
        is_win = (ep_reward > 0)
        if is_win:
            win_counter += 1
        
        # Calculate rolling win rate
        if ep > 0 and ep % win_window == 0:
            win_rate = win_counter / win_window
            training_metrics['win_rate'].append(win_rate)
            win_counter = 0
        
        # Store metrics
        training_metrics['episodes'].append(ep)
        training_metrics['episode_rewards'].append(ep_reward)
        training_metrics['episode_lengths'].append(ep_length)
        training_metrics['epsilon_values'].append(agent.epsilon)
        training_metrics['steps'].append(total_steps)
        
        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon - agent.epsilon_decay)
        
        # Print progress
        if (ep + 1) % 50 == 0:
            recent_rewards = training_metrics['episode_rewards'][-50:]
            avg_reward = np.mean(recent_rewards)
            max_reward = np.max(recent_rewards)
            print(f"Ep {ep+1:5d}/{episodes} | Reward (avg): {avg_reward:7.3f} | Max: {max_reward:7.3f} | "
                  f"ε: {agent.epsilon:.4f} | Steps: {total_steps} | Buffer: {len(agent.buffer)}")
        
        # Save plots periodically
        if ep > 0 and ep % plot_interval == 0:
            save_training_plots(training_metrics, plots_dir, ep)
        
        # Save model periodically
        if (ep + 1) % save_interval == 0:
            model_path = models_dir / f'qmix_model_ep{ep+1}.pth'
            torch.save(agent.q_net.state_dict(), str(model_path))
            print(f"Saved model to {model_path}")

    # Save final model and plots
    print("\nSaving final model and plots...")
    final_model_path = models_dir / 'qmix_model_final.pth'
    torch.save(agent.q_net.state_dict(), str(final_model_path))
    save_training_plots(training_metrics, plots_dir, episodes, final=True)
    
    rewards_path = run_dir / 'episode_rewards.json'
    with open(rewards_path, 'w') as f:
        json.dump({
            'episodes': training_metrics['episodes'],
            'rewards': training_metrics['episode_rewards'],
            'episode_lengths': training_metrics['episode_lengths']
        }, f, indent=2)
    print(f"Rewards saved to {rewards_path}")
    
    print(f"Final model saved to {final_model_path}")
    print_training_summary(training_metrics)


if __name__ == '__main__':
    run_qmix_training(
        episodes=50000,
        steps_per_episode=40,
        train_start=1000,
        update_interval=100,
        plot_interval=5000,
        save_interval=5000,
        env_id='MultiGrid-MultiTargetEmpty-8x8-v0',
        num_agents=3,
    )
