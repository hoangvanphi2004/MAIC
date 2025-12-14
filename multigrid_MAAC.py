import argparse
import gymnasium as gym
import torch
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from gymnasium.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC

import multigrid.envs


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
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'MAAC Training Metrics (Episode {episode})', fontsize=16)
    
    # Plot 1: Critic Loss
    ax1 = axes[0, 0]
    if metrics['critic_losses']:
        ax1.plot(metrics['steps'], metrics['critic_losses'], 
                alpha=0.5, label='Critic Loss', color='blue')
        ax1.plot(metrics['steps'], smooth_curve(metrics['critic_losses']), 
                label='Critic Loss (smoothed)', linewidth=2, color='darkblue')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Critic Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Policy Loss
    ax2 = axes[0, 1]
    if metrics['policy_losses']:
        ax2.plot(metrics['steps'], metrics['policy_losses'], 
                alpha=0.5, label='Policy Loss', color='red')
        ax2.plot(metrics['steps'], smooth_curve(metrics['policy_losses']), 
                label='Policy Loss (smoothed)', linewidth=2, color='darkred')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Policy Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Episode Rewards
    ax3 = axes[1, 0]
    if metrics['episode_rewards']:
        ax3.plot(metrics['episodes'], metrics['episode_rewards'], 
                alpha=0.5, label='Episode Reward', color='green')
        ax3.plot(metrics['episodes'], smooth_curve(metrics['episode_rewards']), 
                label='Episode Reward (smoothed)', linewidth=2, color='darkgreen')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Reward')
    ax3.set_title('Episode Rewards')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mean Agent Rewards
    ax4 = axes[1, 1]
    if metrics['mean_agent_rewards']:
        ax4.plot(metrics['episodes'], metrics['mean_agent_rewards'], 
                alpha=0.5, label='Mean Agent Reward', color='orange')
        ax4.plot(metrics['episodes'], smooth_curve(metrics['mean_agent_rewards']), 
                label='Mean Agent Reward (smoothed)', linewidth=2, color='darkorange')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Mean Reward per Agent')
    ax4.set_title('Mean Agent Rewards')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'training_final.png' if final else f'training_ep{episode}.png'
    save_path = plots_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))
    
    # Create plots directory
    plots_dir = run_dir / 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize tracking metrics
    training_metrics = {
        'critic_losses': [],
        'policy_losses': [],
        'episode_rewards': [],
        'mean_agent_rewards': [],
        'episodes': [],
        'steps': []
    }

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    # Num agent lager than 3
    env = gym.make('MultiGrid-PassSparse-8x8-v0', num_agents=3)
    # print("env observation space:", env.observation_space)
    # print("env action space:", env.action_space)
    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [np.prod(obsp['image'].shape) for _, obsp in env.observation_space.items()],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for _, acsp in env.action_space.items()])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs, infos = env.reset()
        obs = np.array([[np.array(ob['image'], dtype=np.float32).reshape(-1) for _, ob in obs.items()]])
        model.prep_rollouts(device='cpu')

        recent_total_rewards = np.zeros((config.n_rollout_threads, model.nagents))
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(np.array(obs[:, i], dtype=np.float32))),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            # print(torch_obs[0].shape) # (n_rollout_threads, num_agents, obs_dim)
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # print(agent_actions) # list (num_agents) of (n_rollout_threads, action_dim)
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            actions = [np.argmax(one_hot) for one_hot in actions[0]]
            # print("Actions:", actions) # list (n_rollout_threads) of action ints
            actions = {id: actions[id] for id in range(len(actions))}
            # print("Actions:", actions) # list (n_rollout_threads) of action ints
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            dones = [i_terminated or i_truncated for i_terminated, i_truncated in zip(list(terminated.values()), list(truncated.values()))]
            # print("Terminated:", terminated, "Truncated:", truncated, "dones:", dones)
            # print(obs, agent_actions, rewards, next_obs, dones)
            rewards = np.array([[reward for _, reward in rewards.items()]]) # shape (n_rollout_threads, num_agents)
            #print(rewards)
            dones = np.array([dones]) # shape (n_rollout_threads, num_agents)
            next_obs = np.array([[np.array(ob['image'], dtype=np.float32).reshape(-1) for _, ob in next_obs.items()]])
            #print("obs shape:", obs.shape, "next_obs shape:", next_obs.shape, "actions shape:", agent_actions, "rewards shape:", rewards.shape, "dones shape:", dones.shape)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            recent_total_rewards += rewards
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                print("ep_i:", ep_i, "t:", t, "Updating on GPU", recent_total_rewards)
                if config.use_gpu:    
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                
                # Track losses for this update cycle
                cycle_critic_losses = []
                cycle_policy_losses = []
                
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    critic_loss = model.update_critic(sample, logger=logger)
                    policy_loss = model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                    
                    if critic_loss is not None:
                        cycle_critic_losses.append(critic_loss)
                    if policy_loss is not None:
                        cycle_policy_losses.append(policy_loss)
                
                # Store average losses for plotting
                if cycle_critic_losses:
                    training_metrics['critic_losses'].append(np.mean(cycle_critic_losses))
                if cycle_policy_losses:
                    training_metrics['policy_losses'].append(np.mean(cycle_policy_losses))
                training_metrics['steps'].append(t)
                
                model.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        # print("Episode rewards:", ep_rews)
        print(f"Episode {ep_i} rewards: {recent_total_rewards}")
        
        # Store episode metrics
        training_metrics['episodes'].append(ep_i)
        training_metrics['episode_rewards'].append(recent_total_rewards.sum())
        training_metrics['mean_agent_rewards'].append(recent_total_rewards.mean())
        
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)
        
        # Save plots periodically
        if ep_i > 0 and ep_i % config.plot_interval == 0:
            save_training_plots(training_metrics, plots_dir, ep_i)
            print(f"Plots saved to {plots_dir}")

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    
    # Save final plots
    save_training_plots(training_metrics, plots_dir, config.n_episodes, final=True)
    
    rewards_path = run_dir / 'episode_rewards.json'
    with open(rewards_path, 'w') as f:
        json.dump({
            'episodes': training_metrics['episodes'],
            'rewards': training_metrics['episode_rewards']
        }, f, indent=2)
    print(f"Rewards saved to {rewards_path}")
    
    print(f"\nFinal plots saved to {plots_dir}")
    
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="-", help="Name of environment")
    parser.add_argument("--model_name", default="default",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=40, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=10000, type=int)
    parser.add_argument("--plot_interval", default=10000, type=int,
                        help="Interval for saving training plots")
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    run(config)
