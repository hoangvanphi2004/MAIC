import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio
import multigrid
import multigrid.envs
import numpy as np
import torch

from algorithms.MAIC import MAIC


def _resolve_path(path_like: str) -> Path:
	path = Path(path_like)
	if path.is_absolute() or path.is_file():
		return path

	alt_path = Path(__file__).resolve().parent / path
	if alt_path.is_file() or alt_path.exists():
		return alt_path

	return path


def load_train_config(config_path: str) -> dict:
	path = _resolve_path(config_path)

	if not path.is_file():
		raise FileNotFoundError(f'Config file not found: {path}')

	with path.open('r', encoding='utf-8') as f:
		cfg = json.load(f)

	if not isinstance(cfg, dict):
		raise ValueError(f'Config file must be a JSON object: {path}')

	return cfg





def _extract_obs_state(env, obs_dict, info_dict, num_agents, obs_state_mode):
	if obs_state_mode == 'full':
		obs_out = np.array([np.array(o['image'], dtype=np.float32) for _, o in obs_dict.items()])
		state_out = np.array(info_dict['state'], dtype=np.float32)
		return obs_out, state_out

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

	door_obj = getattr(env_unwrapped, 'door_obj', None)
	if door_obj is not None:
		env_features.append(1.0 if bool(getattr(door_obj, 'is_open', False)) else 0.0)

	box_center = getattr(env_unwrapped, 'box_center', None)
	if box_center is not None:
		env_features.extend([float(box_center[0]), float(box_center[1])])

	flat_obs = obs_out.reshape(-1).astype(np.float32)
	if env_features:
		state_out = np.concatenate([flat_obs, np.array(env_features, dtype=np.float32)], axis=0)
	else:
		state_out = flat_obs
	return obs_out, state_out


def run_inference(
	env_id='MultiGrid-PushBox-16x16-v0',
	checkpoint='refactor/model_ep2100.pth',
	episodes=1,
	save_video='refactor/inference.mp4',
	fps=8,
	num_agents=2,
	obs_state_mode='simple',
	seed=None,
	steps_per_episode=40,
	model_config=None,
	**_unused,
):
	# Do not set global seeds; run with default nondeterministic RNG state

	def _infer_hidden_dim_from_checkpoint(path: Path):
		# Load checkpoint on CPU and inspect first actor state dict
		ckpt = torch.load(str(path), map_location='cpu')
		actor_states = ckpt.get('actors', None)
		if not actor_states:
			return None
		st = actor_states[0]
		# Typical keys for non-image Actor: 'fc1.weight', 'fc2.weight', 'out.weight'
		if 'fc1.weight' in st:
			return int(st['fc1.weight'].shape[0])
		if 'fc2.weight' in st:
			return int(st['fc2.weight'].shape[0])
		if 'out.weight' in st:
			# out.weight shape is (action_dim, hidden_dim)
			return int(st['out.weight'].shape[1])
		# Fallback: try any linear weight with 2 dims
		for k, v in st.items():
			if isinstance(v, torch.Tensor) and v.ndim == 2:
				# prefer hidden-like dims (second dim)
				return int(v.shape[1])
		return None


	# Infer hidden_dim from checkpoint; require it to exist.
	checkpoint_path = _resolve_path(checkpoint)
	if not checkpoint_path.is_file():
		raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

	try:
		inferred_hidden = _infer_hidden_dim_from_checkpoint(checkpoint_path)
		if inferred_hidden is None:
			raise RuntimeError('Could not infer hidden_dim from checkpoint; actor linear weights not found.')
		resolved_model_config = {
			'hidden_dim': int(inferred_hidden),
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
	except Exception as e:
		raise RuntimeError(f'Failed to infer hidden_dim from checkpoint: {e}')
	if model_config is not None:
		resolved_model_config.update(model_config)

	if obs_state_mode not in ('full', 'simple'):
		raise ValueError(f"obs_state_mode must be 'full' or 'simple', got: {obs_state_mode}")


	record_video = bool(save_video)
	if record_video:
		env = gym.make(env_id, num_agents=num_agents, render_mode='rgb_array', max_steps=steps_per_episode)
	else:
		env = gym.make(env_id, num_agents=num_agents, max_steps=steps_per_episode)

	# Do not seed action spaces or env.reset here to allow natural randomness
	obs0, info = env.reset()
	obs0_arr, state0_arr = _extract_obs_state(env, obs0, info, num_agents, obs_state_mode)
	obs_shape = obs0_arr.shape[1:]
	state_shape = state0_arr.shape
	action_dim = env.action_space[0].n
	print('Env:', env_id, 'num_agents:', num_agents, 'mode:', obs_state_mode, 'obs_shape:', obs_shape, 'state_shape:', state_shape, 'action_dim:', action_dim)

	agent = MAIC(
		obs_shape,
		state_shape,
		action_dim,
		num_agents=num_agents,
		# do not pass a model init seed here (use MAIC default)
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
	agent.load(str(checkpoint_path))
	agent.actors = [a.eval() for a in agent.actors]
	agent.critic.eval()
	agent.critic_target.eval()

	video_frames = []
	results = []

	for ep in range(episodes):
		obs_dict, info = env.reset()
		obs, state = _extract_obs_state(env, obs_dict, info, num_agents, obs_state_mode)
		ep_reward = 0.0
		ep_len = 0
		frames = []

		for step in range(steps_per_episode):
			actions, _ = agent.select_action(obs, evaluate=True)
			input_actions = {i: int(actions[i]) for i in range(len(actions))}
			next_obs_dict, rewards_dict, terminated, truncated, info = env.step(input_actions)
			rewards_list = [rewards_dict[i] for i in range(num_agents)]
			shared_reward = float(sum(rewards_list))
			ep_reward += shared_reward
			ep_len = step + 1

			if record_video:
				frame = env.render()
				if frame is not None:
					frames.append(np.asarray(frame).astype(np.uint8))

			dones = np.array([terminated[i] or truncated[i] for i in terminated.keys()])
			obs, state = _extract_obs_state(env, next_obs_dict, info, num_agents, obs_state_mode)
			if dones.all() or (step == steps_per_episode - 1):
				break

		results.append({'episode': ep + 1, 'reward': ep_reward, 'length': ep_len})
		print(f'Episode {ep + 1}: reward={ep_reward:.3f}, length={ep_len}')
		if record_video and frames:
			video_frames.extend(frames)

	if record_video and video_frames:
		video_path = Path(save_video)
		if not video_path.is_absolute():
			video_path = _resolve_path(save_video)
		video_path.parent.mkdir(parents=True, exist_ok=True)
		imageio.mimsave(str(video_path), video_frames, fps=int(fps))
		print('Saved video to', video_path)

	return results


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run MAIC inference from a JSON config.')
	parser.add_argument('--config', type=str, default='inference_config.json', help='Path to a single JSON config containing all inference settings.')
	args = parser.parse_args()

	infer_cfg = load_train_config(args.config)
	run_inference(**infer_cfg)