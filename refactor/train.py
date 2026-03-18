import time
from pathlib import Path
import numpy as np
import gymnasium as gym
import multigrid
import multigrid.envs
import imageio

from algorithms.MAIC import SAC_REINFORCE
from algorithms.buffer import ReplayBuffer
from plot import save_plots
from train_helpers import TrainingMetrics


def run_training(
	env_id='MultiGrid-MultiTargetEmpty-16x16-v0',
	num_agents=2,
	obs_state_mode='simple',
	model_config=None,
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
	metrics_save_every=10,
	scaled_information_gain_coef=0.0,
	scaled_entropy_coef=0.0,
):
	SAC_REINFORCE.scaled_information_gain_coef = float(scaled_information_gain_coef)
	SAC_REINFORCE.scaled_entropy_coef = float(scaled_entropy_coef)
	print('scaled_information_gain_coef:', SAC_REINFORCE.scaled_information_gain_coef)
	print('scaled_entropy_coef:', SAC_REINFORCE.scaled_entropy_coef)

	resolved_model_config = {
		'hidden_dim': 128,
		'lr': 3e-4,
		'gamma': 0.99,
		'tau': 0.02,
		'alpha1': 0.02,
		'alpha2': 0.02,
		'alpha_kl': 0.1,
		'policy_update_steps': 3,
		'auto_entropy_tuning': True,
		'target_entropy_scale': 0.2,
	}
	if model_config is not None:
		resolved_model_config.update(model_config)

	run_path = Path(model_dir)
	run_path.mkdir(parents=True, exist_ok=True)
	if record_video:
		env = gym.make(env_id, num_agents=num_agents, render_mode='rgb_array', max_steps=steps_per_episode)
	else:
		env = gym.make(env_id, num_agents=num_agents, max_steps=steps_per_episode)
	if obs_state_mode not in ('full', 'simple'):
		raise ValueError(f"obs_state_mode must be 'full' or 'simple', got: {obs_state_mode}")

	def _extract_obs_state(obs_dict, info_dict):
		if obs_state_mode == 'full':
			obs_out = np.array([np.array(o['image'], dtype=np.float32) for _, o in obs_dict.items()])
			state_out = np.array(info_dict['state'], dtype=np.float32)
			return obs_out, state_out

		# simple mode: compact per-agent features + key env-level flags
		agent_states = env.unwrapped.agent_states
		obs_simple = []
		for i in range(num_agents):
			x, y = agent_states.pos[i]
			direction = agent_states.dir[i]
			carrying_flag = 1.0 if env.unwrapped.agents[i].state.carrying is not None else 0.0
			obs_simple.append(np.array([x, y, direction, carrying_flag], dtype=np.float32))
		obs_out = np.stack(obs_simple, axis=0)

		env_features = []
		env_unwrapped = env.unwrapped

		# Door-centric features (important for PassSparse / door-based tasks)
		door_obj = getattr(env_unwrapped, 'door_obj', None)
		if door_obj is not None:
			env_features.append(1.0 if bool(getattr(door_obj, 'is_open', False)) else 0.0)

		# PushBox-centric feature (if the env exposes the moving box center)
		box_center = getattr(env_unwrapped, 'box_center', None)
		if box_center is not None:
			env_features.extend([float(box_center[0]), float(box_center[1])])

		flat_obs = obs_out.reshape(-1).astype(np.float32)
		if env_features:
			state_out = np.concatenate([flat_obs, np.array(env_features, dtype=np.float32)], axis=0)
		else:
			state_out = flat_obs
		return obs_out, state_out

	def _state_signature(state_arr):
		s = np.asarray(state_arr)
		if np.issubdtype(s.dtype, np.floating):
			s = np.round(s, 4)
		s = np.ascontiguousarray(s)
		return s.tobytes()

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
		hidden_dim=resolved_model_config['hidden_dim'],
		lr=resolved_model_config['lr'],
		gamma=resolved_model_config['gamma'],
		tau=resolved_model_config['tau'],
		alpha1=resolved_model_config['alpha1'],
		alpha2=resolved_model_config['alpha2'],
		alpha_kl=resolved_model_config['alpha_kl'],
		policy_update_steps=resolved_model_config['policy_update_steps'],
		auto_entropy_tuning=resolved_model_config['auto_entropy_tuning'],
		target_entropy_scale=resolved_model_config['target_entropy_scale'],
	)
	replay_buffer = ReplayBuffer(capacity=replay_size)
	total_steps = 0
	metrics = TrainingMetrics()
	seen_state_signatures = {_state_signature(state0_arr)}
	start_time = time.time()

	for ep in range(episodes):
		obs_dict, info = env.reset()
		obs, state = _extract_obs_state(obs_dict, info)
		seen_state_signatures.add(_state_signature(state))
		ep_reward = 0.0
		ep_len = 0
		ep_sac_stats = []
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
			seen_state_signatures.add(_state_signature(next_state))
			done = dones.all() or (step == steps_per_episode - 1)
			if record_this_ep:
				frame = env.render()
				if frame is not None:
					frames.append(frame)
			replay_buffer.push(obs, state, actions, shared_reward, next_obs, next_state, done)
			obs = next_obs
			state = next_state
			if len(replay_buffer) > batch_size and (total_steps + 1) % steps_per_update == 0:
				for _ in range(updates_num):
					sac_stats = agent.update_sac(replay_buffer, batch_size)
					if sac_stats:
						ep_sac_stats.append(sac_stats)
			ep_reward += shared_reward
			ep_len = step + 1
			if done:
				break
		metrics.append_episode(ep_reward, ep_len, ep_sac_stats, unique_states_seen=len(seen_state_signatures))
		if (ep + 1) % 10 == 0:
			elapsed = time.time() - start_time
			metrics.print_episode_metrics(ep, ep_reward, ep_len, elapsed, resolved_model_config['alpha_kl'], len(replay_buffer), total_steps)
		if (ep + 1) % save_every == 0:
			model_path = run_path / f'model_ep{ep+1}.pth'
			agent.save(str(model_path))
			print('Saved model to', model_path)
		if (ep + 1) % metrics_save_every == 0 or ep == episodes - 1:
			metrics.save_json(run_path, ep + 1)
		if (ep + 1) % plot_every == 0 or ep == episodes - 1:
			save_plots(metrics.as_dict(), run_path, ep + 1)
		if record_this_ep and frames:
			frames_u8 = [np.asarray(f).astype(np.uint8) for f in frames]
			vid_path = run_path / f'run_ep{ep+1}.mp4'
			imageio.mimsave(str(vid_path), frames_u8, fps=8)
			print('Saved video to', vid_path)
			
	agent.save(str(run_path / 'model_final.pth'))
	metrics.save_json(run_path, episodes)
	metrics.save_rewards_json(run_path)
	print('Training finished. Saved final model.')
	return metrics.as_dict()