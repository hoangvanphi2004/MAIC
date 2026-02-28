from pathlib import Path
import numpy as np

from components import ReplayBuffer
from utils.plotter import plot_metrics
from utils.video_logger import save_video
from utils.metric_tracker import moving_average, save_rewards_json
from utils.agent_factory import get_agent_class
from utils.env_setup import make_env, get_obs_state_action_dims

def run_training(args):
    run_path = Path(args.model_dir) / args.algo.lower()
    run_path.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_id, args.num_agents, args.steps_per_episode, args.record_video)
    obs_shape, state_shape, action_dim = get_obs_state_action_dims(env)
    print(f'Env: {args.env_id} | num_agents: {args.num_agents} | obs_shape: {obs_shape} | state_shape: {state_shape} | action_dim: {action_dim} | Algo: {args.algo.upper()}')

    AgentClass = get_agent_class(args.algo)
    
    agent = AgentClass(
        obs_shape,
        state_shape,
        action_dim,
        num_agents=args.num_agents,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=0.02,
        auto_entropy_tuning=args.auto_entropy_tuning
    )

    replay_buffer = ReplayBuffer(capacity=args.replay_size)

    total_steps = 0
    metrics = {
        'episode_rewards': [],
        'critic_losses': [],
        'actor_losses': [],
        'entropies': [],
        'q_values': [],
        'alpha_values': [],
        'alpha1_values': [],
        'alpha2_values': [],
    }

    for ep in range(args.episodes):
        obs_dict, infos = env.reset()
        obs = np.array([np.array(o['image'], dtype=np.float32) for _, o in obs_dict.items()])
        state = np.array(infos['state'], dtype=np.float32)
        ep_reward = 0.0
        frames = []
        record_this_ep = False
        
        if args.record_video:
            if (ep + 1) % args.video_every == 0 or ep == args.episodes - 1:
                record_this_ep = True

        for step in range(args.steps_per_episode):
            total_steps += 1
            if total_steps < args.start_steps:
                actions = [env.action_space[i].sample() for i in range(args.num_agents)]
                log_probs = [None] * args.num_agents
            else:
                actions, log_probs = agent.select_action(obs, evaluate=False)

            input_actions = {i: int(actions[i]) for i in range(len(actions))}
            next_obs_dict, rewards_dict, terminated, truncated, infos = env.step(input_actions)
            dones = np.array([terminated[i] or truncated[i] for i in terminated.keys()])
            rewards_list = [rewards_dict[i] for i in range(args.num_agents)]
            shared_reward = float(sum(rewards_list))
            next_obs = np.array([np.array(o['image'], dtype=np.float32) for _, o in next_obs_dict.items()])
            next_state = np.array(infos['state'], dtype=np.float32)
            done = dones.all() or (step == args.steps_per_episode - 1)
            
            if record_this_ep:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception:
                    pass

            if len(replay_buffer) > args.batch_size and total_steps >= args.start_steps and total_steps % args.steps_per_update == 0:
                for _ in range(args.updates_num):
                    result_dict = agent.update(replay_buffer, args.batch_size)
                    if result_dict:
                        metrics['actor_losses'].append(result_dict.get('actor_loss', 0.0))
                        metrics['critic_losses'].append(result_dict.get('critic_loss', 0.0))
                        metrics['q_values'].append(result_dict.get('q_value', 0.0))
                        metrics['alpha1_values'].append(result_dict.get('alpha1_loss', 0.0))
                        metrics['alpha2_values'].append(result_dict.get('alpha2_loss', 0.0))
                        metrics['entropies'].append(result_dict.get('entropy', 0.0))
            
            # print(f"obs shape: {obs.shape}")
            # print(f"state shape: {state.shape}")
            # print(f"actions shape: {actions}")
            # print(f"shared_reward: {shared_reward}")
            # print(f"next_obs shape: {next_obs.shape}")
            # print(f"next_state shape: {next_state.shape}")
            # print(f"done: {done}")

            replay_buffer.push(obs, state, actions, shared_reward, next_obs, next_state, done)
            obs = next_obs
            state = next_state
            ep_reward += shared_reward
            
            if done:
                break

        metrics['episode_rewards'].append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg_recent = np.mean(metrics['episode_rewards'][-10:])
            print(f'Ep {ep+1}/{args.episodes} | Reward (avg10): {avg_recent:.3f} | Steps: {total_steps}')

        if (ep + 1) % args.save_every == 0:
            model_path = run_path / f'model_ep{ep+1}.pth'
            if hasattr(agent, 'save'):
                agent.save(str(model_path))
                print('Saved model to', model_path)

        if record_this_ep and frames:
            vid_path = run_path / f'run_ep{ep+1}.mp4'
            save_video(frames, vid_path)

        if (ep + 1) % args.plot_every == 0:
            plot_metrics(metrics, run_path, moving_average)

    if hasattr(agent, 'save'):
        agent.save(str(run_path / 'model_final.pth'))
    
    save_rewards_json(metrics['episode_rewards'], run_path / 'episode_rewards.json')

    print('Training finished. Saved final model.')
    plot_metrics(metrics, run_path, moving_average)
    return metrics
