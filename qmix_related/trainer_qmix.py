from pathlib import Path
import numpy as np

from components import ReplayBuffer, EpisodeMemory
from utils.plotter import plot_metrics
from utils.video_logger import save_video
from utils.metric_tracker import update_metrics, moving_average, save_rewards_json
from utils.agent_factory import get_agent_class
from utils.env_setup import make_env, get_state_action_dims

def run_training_qmix(args):
    run_path = Path(args.model_dir) / args.algo.lower()
    run_path.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_id, args.num_agents, args.steps_per_episode, args.record_video)
    state_shape, action_dim = get_state_action_dims(env)
    
    print(f'Env: {args.env_id} | num_agents: {args.num_agents} | state_shape: {state_shape} | action_dim: {action_dim} | Algo: {args.algo.upper()}')

    AgentClass = get_agent_class(args.algo)
    
    agent = AgentClass(
        state_shape,
        action_dim,
        num_agents=args.num_agents,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma
    )

    replay_buffer = ReplayBuffer(capacity=args.replay_size)
    episode_memory = EpisodeMemory(num_agents=args.num_agents)

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
        obs_dict, _ = env.reset()
        obs = np.array([np.array(o['image'], dtype=np.float32) for _, o in obs_dict.items()])
        episode_memory.clear()
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
                try:
                    actions = agent.select_action(obs, evaluate=False)
                    log_probs = [None] * args.num_agents
                except TypeError:
                    actions = agent.select_action(obs)
                    log_probs = [None] * args.num_agents
                if isinstance(actions, tuple):
                    actions, log_probs = actions

            input_actions = {i: int(actions[i]) for i in range(len(actions))}
            next_obs_dict, rewards_dict, terminated, truncated, _ = env.step(input_actions)
            dones = np.array([terminated[i] or truncated[i] for i in terminated.keys()])
            rewards_list = [rewards_dict[i] for i in range(args.num_agents)]
            shared_reward = float(sum(rewards_list))
            next_obs = np.array([np.array(o['image'], dtype=np.float32) for _, o in next_obs_dict.items()])
            done = dones.all() or (step == args.steps_per_episode - 1)
            
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

        update_metrics(metrics, replay_buffer, agent, args.batch_size, args.steps_per_update, args.updates_num, ep)

        update_policy_fn = getattr(agent, 'update_reinforce', None)
        if update_policy_fn is None:
            update_policy_fn = getattr(agent, 'update_policies', None)
            
        if update_policy_fn:
            try:
                stats = update_policy_fn(episode_memory)
                if stats:
                    metrics['actor_losses'].append(stats.get('policy_loss', 0.0))
                    metrics['entropies'].append(stats.get('entropy', 0.0))
            except Exception as e:
                print(f'{args.algo} actor update failed:', e)

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
            plot_metrics(metrics, run_path, moving_average)

    if hasattr(agent, 'save'):
        agent.save(str(run_path / 'model_final.pth'))
    
    save_rewards_json(metrics['episode_rewards'], run_path / 'episode_rewards.json')

    print('Training finished. Saved final model.')
    plot_metrics(metrics, run_path, moving_average)
    return metrics
