import time
import warnings
import json
from pathlib import Path
import gymnasium as gym
import multigrid.envs
import numpy as np
import torch
from CMASAC import ReplayBuffer, SAC_REINFORCE
from plot_utils import save_plots
try:
	import imageio
except Exception:
	imageio = None
def run_training(
	env_id='MultiGrid-PassSparse-8x8-v0',
	num_agents=2,
	episodes=5000,
	steps_per_episode=40,
	replay_size=int(1e6),
	batch_size=64,
	start_steps=100,
	steps_per_update=1,
	updates_num=1,
	save_every=500,
	model_dir='runs/cmasac_reinforce',
	render=False,
	record_video=False,
	video_every=500,
	plot_every=50,
):
	run_path = Path(model_dir)
	run_path.mkdir(parents=True, exist_ok=True)
	if record_video:
		try:
			env = gym.make(env_id, num_agents=num_agents, render_mode='rgb_array', max_steps = steps_per_episode)
		except TypeError:
			env = gym.make(env_id, num_agents=num_agents, max_steps = steps_per_episode)
	else:
		env = gym.make(env_id, num_agents=num_agents, max_steps = steps_per_episode)
	obs0, info = env.reset()
	first_obs = list(obs0.values())[0]
	if isinstance(first_obs, dict) and 'image' in first_obs:
		obs_shape = first_obs['image'].shape
	else:
		obs_shape = env.observation_space[0]['image'].shape
	state_shape = info['state'].shape
	action_dim = env.action_space[0].n
	print('Env:', env_id, 'num_agents:', num_agents, 'obs_shape:', obs_shape, 'state_shape:', state_shape, 'action_dim:', action_dim)
	agent = SAC_REINFORCE(
		obs_shape,
		state_shape,
		action_dim,
		num_agents=num_agents,
		hidden_dim=128,
		lr=3e-4,
		gamma=0.99,
		tau=0.02,
		alpha1=0.02,
		alpha2=0.02,
		auto_entropy_tuning=True,
	)
	replay_buffer = ReplayBuffer(capacity=replay_size)
	total_steps = 0
	metrics = {
		'episode_rewards': [],
		'episode_lengths': [],
		'critic_losses': [],
		'actor_losses': [],
		'entropies': [],
		'q_values': [],
		'alpha1_losses': [],
		'alpha2_losses': [],
		'alpha1_values': [],
		'alpha2_values': [],
		'information_gains': [],
	}
	start_time = time.time()

	def _mean_or_none(values):
		if not values:
			return None
		return float(np.mean(values))

	def _fmt(value, precision=4):
		if value is None:
			return 'n/a'
		return f'{value:.{precision}f}'

	for ep in range(episodes):
		obs_dict, info = env.reset()
		obs = np.array([np.array(o['image'], dtype=np.float32) for _, o in obs_dict.items()])
		state = np.array(info['state'], dtype=np.float32)
		ep_reward = 0.0
		ep_len = 0
		ep_actor_losses = []
		ep_critic_losses = []
		ep_alpha1_losses = []
		ep_alpha2_losses = []
		ep_q_values = []
		ep_entropies = []
		ep_alpha1_values = []
		ep_alpha2_values = []
		ep_info_gains = []
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
			next_state = np.array(info['state'], dtype=np.float32)
			done = dones.all() or (step == steps_per_episode - 1)
			if record_this_ep:
				try:
					frame = env.render()
					if frame is not None:
						frames.append(frame)
				except Exception:
					pass
			replay_buffer.push(obs, state, actions, shared_reward, next_obs, next_state, done)
			obs = next_obs
			state = next_state
			if len(replay_buffer) > batch_size and (total_steps + 1) % steps_per_update == 0:
				for _ in range(updates_num):
					sac_stats = agent.update_sac(replay_buffer, batch_size)
					if sac_stats:
						ep_critic_losses.append(sac_stats.get('critic_loss', 0.0))
						ep_actor_losses.append(sac_stats.get('policy_loss', 0.0))
						ep_entropies.append(sac_stats.get('entropy', 0.0))
						ep_q_values.append(sac_stats.get('q_value', 0.0))
						ep_info_gains.append(sac_stats.get('information_gain', 0.0))
						if 'alpha1_loss' in sac_stats:
							ep_alpha1_losses.append(sac_stats['alpha1_loss'])
						if 'alpha2_loss' in sac_stats:
							ep_alpha2_losses.append(sac_stats['alpha2_loss'])
						if 'alpha1_value' in sac_stats:
							ep_alpha1_values.append(sac_stats['alpha1_value'])
						if 'alpha2_value' in sac_stats:
							ep_alpha2_values.append(sac_stats['alpha2_value'])
			ep_reward += shared_reward
			ep_len = step + 1
			if done:
				break
		metrics['episode_rewards'].append(ep_reward)
		metrics['episode_lengths'].append(ep_len)
		metrics['actor_losses'].append(_mean_or_none(ep_actor_losses))
		metrics['critic_losses'].append(_mean_or_none(ep_critic_losses))
		metrics['alpha1_losses'].append(_mean_or_none(ep_alpha1_losses))
		metrics['alpha2_losses'].append(_mean_or_none(ep_alpha2_losses))
		metrics['q_values'].append(_mean_or_none(ep_q_values))
		metrics['entropies'].append(_mean_or_none(ep_entropies))
		metrics['alpha1_values'].append(_mean_or_none(ep_alpha1_values))
		metrics['alpha2_values'].append(_mean_or_none(ep_alpha2_values))
		metrics['information_gains'].append(_mean_or_none(ep_info_gains))
		if (ep + 1) % 10 == 0:
			recent_rewards = metrics['episode_rewards'][-10:]
			recent_lengths = metrics['episode_lengths'][-10:]
			recent_actor = [x for x in metrics['actor_losses'][-10:] if x is not None]
			recent_critic = [x for x in metrics['critic_losses'][-10:] if x is not None]
			recent_alpha1_loss = [x for x in metrics['alpha1_losses'][-10:] if x is not None]
			recent_alpha2_loss = [x for x in metrics['alpha2_losses'][-10:] if x is not None]
			recent_q = [x for x in metrics['q_values'][-10:] if x is not None]
			recent_entropy = [x for x in metrics['entropies'][-10:] if x is not None]
			recent_alpha1 = [x for x in metrics['alpha1_values'][-10:] if x is not None]
			recent_alpha2 = [x for x in metrics['alpha2_values'][-10:] if x is not None]

			avg_recent_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
			avg_recent_len = float(np.mean(recent_lengths)) if recent_lengths else 0.0
			elapsed = time.time() - start_time

			print(f'Reward: {ep_reward:.3f} (Avg 10: {avg_recent_reward:.3f}) | Length: {ep_len} (Avg 10: {avg_recent_len:.1f})')
			print(
				f'{elapsed:.1f}s\t{ep+1}\t'
				f'Actor Loss: {_fmt(_mean_or_none(recent_actor), 6)} | '
				f'Critic Loss: {_fmt(_mean_or_none(recent_critic), 6)} | '
				f'Alpha1 Loss: {_fmt(_mean_or_none(recent_alpha1_loss), 6)} | '
				f'Alpha2 Loss: {_fmt(_mean_or_none(recent_alpha2_loss), 6)}'
			)
			print(
				f'{elapsed:.1f}s\t{ep+1}\t'
				f'Q-Value: {_fmt(_mean_or_none(recent_q), 4)} | '
				f'Entropy: {_fmt(_mean_or_none(recent_entropy), 4)} | '
				f'Alpha1: {_fmt(_mean_or_none(recent_alpha1), 4)} | '
				f'Alpha2: {_fmt(_mean_or_none(recent_alpha2), 4)}'
			)
			print(f'{elapsed:.1f}s\t{ep+1}\tBuffer Size: {len(replay_buffer)} | Total Steps: {total_steps}')
		if (ep + 1) % save_every == 0:
			model_path = run_path / f'model_ep{ep+1}.pth'
			agent.save(str(model_path))
			print('Saved model to', model_path)
		# Generate plots independently based on plot_every frequency
		if (ep + 1) % plot_every == 0 or ep == episodes - 1:
			save_plots(metrics, run_path, ep + 1)
		# Save video if frames were recorded
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
	agent.save(str(run_path / 'model_final.pth'))
	rewards_path = run_path / 'episode_rewards.json'
	with open(rewards_path, 'w') as f:
		json.dump({
			'episodes': list(range(len(metrics['episode_rewards']))),
			'rewards': metrics['episode_rewards']
		}, f, indent=2)
	print(f"Rewards saved to {rewards_path}")
	print('Training finished. Saved final model.')
	return metrics
if __name__ == '__main__':
	run_training(
		episodes=5000,
		steps_per_episode=40,
		batch_size=1024,
		updates_num=1,
		save_every=100,
		video_every=100,
		plot_every=10,
		record_video=True,
	)