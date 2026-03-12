import time
import warnings
import json
import argparse
from pathlib import Path
import gymnasium as gym
import multigrid.envs
import numpy as np
import torch
import CMASAC
from CMASAC import ReplayBuffer, SAC_REINFORCE
from plot_utils import save_plots
try:
	import imageio
except Exception:
	imageio = None
def run_training(
	env_id='MultiGrid-MultiTargetEmpty-16x16-v0',
	num_agents=2,
	obs_state_mode='simple',
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
	scaled_information_gain_coef=0.0,
	scaled_entropy_coef=0.0,
	alpha_kl=0.1,
	policy_update_steps=3,
):
	CMASAC.scaled_information_gain_coef = float(scaled_information_gain_coef)
	CMASAC.scaled_entropy_coef = float(scaled_entropy_coef)
	print('scaled_information_gain_coef:', CMASAC.scaled_information_gain_coef)
	print('scaled_entropy_coef:', CMASAC.scaled_entropy_coef)

	run_path = Path(model_dir)
	run_path.mkdir(parents=True, exist_ok=True)
	if record_video:
		try:
			env = gym.make(env_id, num_agents=num_agents, render_mode='rgb_array', max_steps = steps_per_episode)
		except TypeError:
			env = gym.make(env_id, num_agents=num_agents, max_steps = steps_per_episode)
	else:
		env = gym.make(env_id, num_agents=num_agents, max_steps = steps_per_episode)
	if obs_state_mode not in ('full', 'simple'):
		raise ValueError(f"obs_state_mode must be 'full' or 'simple', got: {obs_state_mode}")

	def _extract_obs_state(obs_dict, info_dict):
		if obs_state_mode == 'full':
			obs_out = np.array([np.array(o['image'], dtype=np.float32) for _, o in obs_dict.items()])
			state_out = np.array(info_dict['state'], dtype=np.float32)
			return obs_out, state_out

		# simple mode: each agent obs = [x, y, direction], state = concat of all agents obs
		agent_states = env.unwrapped.agent_states
		obs_simple = []
		for i in range(num_agents):
			x, y = agent_states.pos[i]
			direction = agent_states.dir[i]
			obs_simple.append(np.array([x, y, direction], dtype=np.float32))
		obs_out = np.stack(obs_simple, axis=0)
		state_out = obs_out.reshape(-1).astype(np.float32)
		return obs_out, state_out

	obs0, info = env.reset()
	obs0_arr, state0_arr = _extract_obs_state(obs0, info)
	obs_shape = obs0_arr.shape[1:]
	state_shape = state0_arr.shape
	action_dim = env.action_space[0].n
	print('Env:', env_id, 'num_agents:', num_agents, 'mode:', obs_state_mode, 'obs_shape:', obs_shape, 'state_shape:', state_shape, 'action_dim:', action_dim)
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
		alpha_kl=alpha_kl,
		policy_update_steps=policy_update_steps,
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
		'kl_divergences': [],
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
		obs, state = _extract_obs_state(obs_dict, info)
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
		ep_kl_divergences = []
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
			next_obs, next_state = _extract_obs_state(next_obs_dict, info)
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
						ep_kl_divergences.append(sac_stats.get('kl_divergence', 0.0))
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
		metrics['kl_divergences'].append(_mean_or_none(ep_kl_divergences))
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
			recent_kl = [x for x in metrics['kl_divergences'][-10:] if x is not None]

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
				f'KL: {_fmt(_mean_or_none(recent_kl), 6)} | '
				f'Alpha1: {_fmt(_mean_or_none(recent_alpha1), 4)} | '
				f'Alpha2: {_fmt(_mean_or_none(recent_alpha2), 4)} | '
				f'AlphaKL: {alpha_kl:.4f}'
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
	parser = argparse.ArgumentParser(description='Train CMASAC with configurable observation mode and information bonus coefficient.')
	parser.add_argument('--simple', action='store_true', help='Use simple obs/state: obs=[x,y,dir], state=concat of all agent obs.')
	parser.add_argument('--info_coef', type=float, default=0.0, help='Set CMASAC.scaled_information_gain_coef (e.g. --info_coef 0).')
	parser.add_argument('--entropy_coef', type=float, default=0.0, help='Set CMASAC.scaled_entropy_coef (e.g. --entropy_coef 0).')
	parser.add_argument('--alpha_kl', type=float, default=1.0, help='KL regularization coefficient for actor updates.')
	parser.add_argument('--policy_update_steps', type=int, default=3, help='Number of actor updates per sampled batch using a fixed old policy snapshot.')
	args = parser.parse_args()

	obs_mode = 'simple' if args.simple else 'full'

	run_training(
		obs_state_mode=obs_mode,
		episodes=5000,
		steps_per_episode=40,
		batch_size=1024,
		updates_num=1,
		save_every=100,
		video_every=100,
		plot_every=10,
		record_video=True,
		scaled_information_gain_coef=args.info_coef,
		scaled_entropy_coef=args.entropy_coef,
		alpha_kl=args.alpha_kl,
		policy_update_steps=args.policy_update_steps,
	)