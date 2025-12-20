import time
import warnings
import json
from pathlib import Path

import gymnasium as gym
import multigrid.envs
import numpy as np
import torch

from QMIX import QMIX, ReplayBuffer

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import imageio
except Exception:
    imageio = None


def run_training(
    env_id='MultiGrid-MultiTargetEmpty-8x8-v0',
    num_agents=3,
    episodes=5000,
    steps_per_episode=40,
    replay_size=int(5e5),
    batch_size=64,
    start_steps=1000,
    steps_per_update=4,
    updates_num=4,
    save_every=500,
    model_dir='runs/qmix_multitarget',
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
            env = gym.make(env_id, num_agents=num_agents, render_mode='rgb_array', max_steps = steps_per_episode)
        except TypeError:
            env = gym.make(env_id, num_agents=num_agents, max_steps = steps_per_episode)
    else:
        env = gym.make(env_id, num_agents=num_agents, max_steps = steps_per_episode)

    obs0, _ = env.reset()
    first_obs = list(obs0.values())[0]
    if isinstance(first_obs, dict) and 'image' in first_obs:
        state_shape = first_obs['image'].shape
    else:
        state_shape = env.observation_space[0]['image'].shape
    action_dim = env.action_space[0].n

    print('Env:', env_id, 'num_agents:', num_agents, 'state_shape:', state_shape, 'action_dim:', action_dim)

    agent = QMIX(
        obs_shape=state_shape,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        tau=0.02,
        share_agent=True,
    )
    replay_buffer = ReplayBuffer(capacity=replay_size)

    total_steps = 0
    metrics = {
        'episode_rewards': [],
        'losses': [],
        'q_tot_values': [],
        'epsilons': [],
    }

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        obs = np.array([np.array(o['image'], dtype=np.float32) for _, o in obs_dict.items()])
        # no per-episode memory needed for QMIX
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
            else:
                actions = agent.select_action(obs)

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

            obs = next_obs
            ep_reward += shared_reward

            if done:
                break

        if len(replay_buffer) > batch_size and (ep + 1) % steps_per_update == 0:
            cycle_loss = []
            cycle_qtot = []
            cycle_eps = []
            for _ in range(updates_num):
                stats = agent.update(replay_buffer, batch_size)
                if stats:
                    cycle_loss.append(stats.get('loss', 0.0))
                    cycle_qtot.append(stats.get('q_tot', 0.0))
                    cycle_eps.append(stats.get('epsilon', 0.0))
            if cycle_loss:
                metrics['losses'].append(float(np.mean(cycle_loss)))
            if cycle_qtot:
                metrics['q_tot_values'].append(float(np.mean(cycle_qtot)))
            if cycle_eps:
                metrics['epsilons'].append(float(cycle_eps[-1]))

        # no policy-gradient update for QMIX

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
                
                # Update metricgrid after saving video
                try:
                    if plt is not None:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        (ax_reward, ax_loss), (ax_qtot, ax_eps) = axes

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

                        losses = metrics.get('losses', [])
                        if losses:
                            ax_loss.plot(losses, marker='o', ms=3)
                            ax_loss.set_title('TD Loss (avg per log interval)')
                            ax_loss.grid(True, alpha=0.3)

                        qtot_vals = metrics.get('q_tot_values', [])
                        if qtot_vals:
                            ax_qtot.plot(qtot_vals, marker='o', ms=3)
                            ax_qtot.set_title('Q_tot (avg per log interval)')
                            ax_qtot.grid(True, alpha=0.3)

                        eps_vals = metrics.get('epsilons', [])
                        if eps_vals:
                            ax_eps.plot(eps_vals, marker='o', ms=3)
                            ax_eps.set_title('Epsilon')
                            ax_eps.grid(True, alpha=0.3)

                        # hide unused axes if no data
                        if not losses:
                            ax_loss.set_axis_off()
                        if not qtot_vals:
                            ax_qtot.set_axis_off()
                        if not eps_vals:
                            ax_eps.set_axis_off()

                        plt.tight_layout()
                        out = run_path / 'metrics_grid.png'
                        plt.savefig(str(out), dpi=150)
                        plt.close(fig)
                        print('Saved metric plots to', out)
                except Exception as e:
                    warnings.warn(f'Failed to save metric plots: {e}')

    agent.save(str(run_path / 'model_final.pth'))
    rewards_path = run_path / 'episode_rewards.json'
    with open(rewards_path, 'w') as f:
        json.dump({
            'episodes': list(range(len(metrics['episode_rewards']))),
            'rewards': metrics['episode_rewards']
        }, f, indent=2)
    print(f"Rewards saved to {rewards_path}")
    print('Training finished. Saved final model.')
    try:
        if plt is None:
            warnings.warn('matplotlib not available; skipping plots')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            (ax_reward, ax_loss), (ax_qtot, ax_eps) = axes

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

            losses = metrics.get('losses', [])
            if losses:
                ax_loss.plot(losses, marker='o', ms=3)
                ax_loss.set_title('TD Loss (avg per log interval)')
                ax_loss.grid(True, alpha=0.3)

            qtot_vals = metrics.get('q_tot_values', [])
            if qtot_vals:
                ax_qtot.plot(qtot_vals, marker='o', ms=3)
                ax_qtot.set_title('Q_tot (avg per log interval)')
                ax_qtot.grid(True, alpha=0.3)

            eps_vals = metrics.get('epsilons', [])
            if eps_vals:
                ax_eps.plot(eps_vals, marker='o', ms=3)
                ax_eps.set_title('Epsilon')
                ax_eps.grid(True, alpha=0.3)

            if not losses:
                ax_loss.set_axis_off()
            if not qtot_vals:
                ax_qtot.set_axis_off()
            if not eps_vals:
                ax_eps.set_axis_off()

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
        episodes=10000,
        steps_per_episode=40,
        batch_size=1024,
        updates_num=4,
        save_every=500,
        video_every=500,
        record_video=True,
    )