import time
import warnings
from pathlib import Path
import gymnasium as gym
import multigrid.envs
import numpy as np
import torch
from MASAC_valid_reinforce import EpisodeMemory, ReplayBuffer, SAC_REINFORCE
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import imageio
except Exception:
    imageio = None
def run_training(
    env_id='MultiGrid-PassSparse-8x8-v0',
    num_agents=3,
    episodes=5000,
    steps_per_episode=40,
    replay_size=int(1e6),
    batch_size=64,
    start_steps=1000,
    steps_per_update=4,
    updates_num=4,
    save_every=500,
    model_dir='runs/masac_reinforce',
    render=False,
    record_video=False,
    video_every=500,
):
    run_path = Path(model_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    def _moving_average(vals, window=20):
        if len(vals) < 1:
            return np.array([])
        window = max(1, min(window, len(vals)))
        cumsum = np.cumsum(np.insert(vals, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / float(window)
    if record_video:
        try:
            env = gym.make(env_id, num_agents=num_agents, render_mode='rgb_array')
        except TypeError:
            env = gym.make(env_id, num_agents=num_agents)
    else:
        env = gym.make(env_id, num_agents=num_agents)
    obs0, _ = env.reset()
    first_obs = list(obs0.values())[0]
    if isinstance(first_obs, dict) and 'image' in first_obs:
        state_shape = first_obs['image'].shape
    else:
        state_shape = env.observation_space[0]['image'].shape
    action_dim = env.action_space[0].n
    print('Env:', env_id, 'num_agents:', num_agents, 'state_shape:', state_shape, 'action_dim:', action_dim)
    agent = SAC_REINFORCE(
        state_shape,
        action_dim,
        num_agents=num_agents,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        tau=0.02,
        alpha=0.01,
        auto_entropy_tuning=False,
    )
    replay_buffer = ReplayBuffer(capacity=replay_size)
    episode_memory = EpisodeMemory(num_agents=num_agents)
    total_steps = 0
    metrics = {
        'episode_rewards': [],
        'critic_losses': [],
        'actor_losses': [],
        'entropies': [],
        'q_values': [],
        'alpha_values': [],
    }
    for ep in range(episodes):
        obs_dict, _ = env.reset()
        obs = np.array([np.array(o['image'], dtype=np.float32) for _, o in obs_dict.items()])
        episode_memory.clear()
        ep_reward = 0.0
        frames = []
        record_this_ep = False
        if record_video:
            if (ep + 1) % video_every == 0 or ep == episodes - 1:
                record_this_ep = True
        for step in range(steps_per_episode):
            total_steps += 1
            if total_steps < start_steps:
                actions = [env.action_space[i].sample() for i in range(num_agents)]
                log_probs = [None] * num_agents
            else:
                actions, log_probs = agent.select_action(obs, evaluate=False)
            input_actions = {i: int(actions[i]) for i in range(len(actions))}
            next_obs_dict, rewards_dict, terminated, truncated, info = env.step(input_actions)
            dones = np.array([terminated[i] or truncated[i] for i in terminated.keys()])
            rewards_list = [rewards_dict[i] for i in range(num_agents)]
            shared_reward = float(sum(rewards_list))
            next_obs = np.array([np.array(o['image'], dtype=np.float32) for _, o in next_obs_dict.items()])
            done = dones.all() or (step == steps_per_episode - 1)
            if record_this_ep:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception:
                    pass
            replay_buffer.push(obs, actions, shared_reward, next_obs, done)
            episode_memory.push(obs, actions, shared_reward, log_probs)
            obs = next_obs
            ep_reward += shared_reward
            if done:
                break
        if len(replay_buffer) > batch_size and (ep + 1) % steps_per_update == 0:
            cycle_critic_loss = []
            cycle_q = []
            cycle_alpha = []
            for _ in range(updates_num):
                sac_stats = agent.update_sac(replay_buffer, batch_size)
                if sac_stats:
                    cycle_critic_loss.append(sac_stats.get('critic_loss', 0.0))
                    cycle_q.append(sac_stats.get('q_value', 0.0))
                    if 'alpha_value' in sac_stats:
                        cycle_alpha.append(sac_stats['alpha_value'])
            if cycle_critic_loss:
                metrics['critic_losses'].append(np.mean(cycle_critic_loss))
            if cycle_q:
                metrics['q_values'].append(np.mean(cycle_q))
            if cycle_alpha:
                metrics['alpha_values'].append(np.mean(cycle_alpha))
        try:
            reinforce_stats = agent.update_reinforce(episode_memory)
            if reinforce_stats:
                metrics['actor_losses'].append(reinforce_stats.get('policy_loss', 0.0))
                metrics['entropies'].append(reinforce_stats.get('entropy', 0.0))
        except Exception as e:
            print('REINFORCE update failed:', e)
        metrics['episode_rewards'].append(ep_reward)
        if (ep + 1) % 10 == 0:
            avg_recent = np.mean(metrics['episode_rewards'][-10:])
            print(f'Ep {ep+1}/{episodes} | Reward (avg10): {avg_recent:.3f} | Steps: {total_steps}')
        if (ep + 1) % save_every == 0:
            model_path = run_path / f'model_ep{ep+1}.pth'
            agent.save(str(model_path))
            print('Saved model to', model_path)
        if record_this_ep and frames:
            if imageio is None:
                warnings.warn('imageio not installed; cannot save video')
            else:
                try:
                    frames_u8 = [np.asarray(f).astype(np.uint8) for f in frames]
                except Exception:
                    frames_u8 = frames
                vid_path = run_path / f'run_ep{ep+1}.mp4'
                try:
                    imageio.mimsave(str(vid_path), frames_u8, fps=8)
                    print('Saved video to', vid_path)
                except Exception:
                    try:
                        with imageio.get_writer(str(vid_path), fps=8) as writer:
                            for f in frames_u8:
                                writer.append_data(f)
                        print('Saved video to', vid_path)
                    except Exception as ex:
                        warnings.warn(f'Failed to write video: {ex}')
                try:
                    if plt is not None:
                        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
                        (ax_reward, ax_policy), (ax_critic, ax_q), (ax_entropy, ax_alpha) = axes
                        rewards = metrics.get('episode_rewards', [])
                        if rewards:
                            x = np.arange(len(rewards))
                            ax_reward.plot(x, rewards, color='tab:blue', alpha=0.35, label='Episode Reward')
                            ma = _moving_average(np.array(rewards), window=20)
                            if ma.size:
                                ax_reward.plot(np.arange(len(ma)) + 19, ma, color='tab:orange', label='20-ep MA')
                            ax_reward.set_title('Episode Reward')
                            ax_reward.legend()
                            ax_reward.grid(True, alpha=0.3)
                        policy_losses = metrics.get('actor_losses', [])
                        if policy_losses:
                            ax_policy.plot(policy_losses, marker='o', ms=3)
                            ax_policy.set_title('Policy Loss (avg per log interval)')
                            ax_policy.grid(True, alpha=0.3)
                        critic_losses = metrics.get('critic_losses', [])
                        if critic_losses:
                            ax_critic.plot(critic_losses, marker='o', ms=3)
                            ax_critic.set_title('Critic Loss (avg per log interval)')
                            ax_critic.grid(True, alpha=0.3)
                        q_values = metrics.get('q_values', [])
                        if q_values:
                            ax_q.plot(q_values, marker='o', ms=3)
                            ax_q.set_title('Q-value (avg per log interval)')
                            ax_q.grid(True, alpha=0.3)
                        entropy_vals = metrics.get('entropies', [])
                        if entropy_vals:
                            ax_entropy.plot(entropy_vals, marker='o', ms=3)
                            ax_entropy.set_title('Entropy (avg per episode)')
                            ax_entropy.grid(True, alpha=0.3)
                        else:
                            ax_entropy.set_axis_off()
                        alpha_vals = metrics.get('alpha_values', [])
                        if alpha_vals:
                            ax_alpha.plot(alpha_vals, marker='o', ms=3)
                            ax_alpha.set_title('Alpha')
                            ax_alpha.grid(True, alpha=0.3)
                        else:
                            ax_alpha.set_axis_off()
                        plt.tight_layout()
                        out = run_path / 'metrics_grid.png'
                        plt.savefig(str(out), dpi=150)
                        plt.close(fig)
                        print('Saved metric plots to', out)
                except Exception as e:
                    warnings.warn(f'Failed to save metric plots: {e}')
    agent.save(str(run_path / 'model_final.pth'))
    print('Training finished. Saved final model.')
    try:
        if plt is None:
            warnings.warn('matplotlib not available; skipping plots')
        else:
            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            (ax_reward, ax_policy), (ax_critic, ax_q), (ax_entropy, ax_alpha) = axes
            rewards = metrics.get('episode_rewards', [])
            if rewards:
                x = np.arange(len(rewards))
                ax_reward.plot(x, rewards, color='tab:blue', alpha=0.35, label='Episode Reward')
                ma = _moving_average(np.array(rewards), window=20)
                if ma.size:
                    ax_reward.plot(np.arange(len(ma)) + 19, ma, color='tab:orange', label='20-ep MA')
                ax_reward.set_title('Episode Reward')
                ax_reward.legend()
                ax_reward.grid(True, alpha=0.3)
            policy_losses = metrics.get('actor_losses', [])
            if policy_losses:
                ax_policy.plot(policy_losses, marker='o', ms=3)
                ax_policy.set_title('Policy Loss (avg per log interval)')
                ax_policy.grid(True, alpha=0.3)
            critic_losses = metrics.get('critic_losses', [])
            if critic_losses:
                ax_critic.plot(critic_losses, marker='o', ms=3)
                ax_critic.set_title('Critic Loss (avg per log interval)')
                ax_critic.grid(True, alpha=0.3)
            q_values = metrics.get('q_values', [])
            if q_values:
                ax_q.plot(q_values, marker='o', ms=3)
                ax_q.set_title('Q-value (avg per log interval)')
                ax_q.grid(True, alpha=0.3)
            entropy_vals = metrics.get('entropies', [])
            if entropy_vals:
                ax_entropy.plot(entropy_vals, marker='o', ms=3)
                ax_entropy.set_title('Entropy (avg per episode)')
                ax_entropy.grid(True, alpha=0.3)
            else:
                ax_entropy.set_axis_off()
            alpha_vals = metrics.get('alpha_values', [])
            if alpha_vals:
                ax_alpha.plot(alpha_vals, marker='o', ms=3)
                ax_alpha.set_title('Alpha')
                ax_alpha.grid(True, alpha=0.3)
            else:
                ax_alpha.set_axis_off()
            plt.tight_layout()
            out = run_path / 'metrics_grid.png'
            plt.savefig(str(out), dpi=150)
            plt.close(fig)
            print('Saved metric plots to', out)
    except Exception as e:
        warnings.warn(f'Failed to save plots: {e}')
    return metrics
if __name__ == '__main__':
    run_training(
        episodes=500,
        steps_per_episode=100,
        batch_size=1024,
        updates_num=4,
        save_every=50,
        video_every=50,
        record_video=True,
    )