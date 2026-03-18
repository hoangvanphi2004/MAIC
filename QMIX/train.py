
from __future__ import annotations
import random
import random
from pathlib import Path
import numpy as np
import torch
import json
import os
from multigrid_gym_adapter import make_refactor_multigrid_gym_env
from QMIX import QMIXAgent
from utils import ReplayBuffer, TrainingMetrics


def linear_epsilon(step: int, start_steps: int, decay_steps: int, eps_start: float, eps_end: float) -> float:
	if step < start_steps:
		return eps_start
	progress = min(1.0, (step - start_steps) / max(1, decay_steps))
	return eps_start + progress * (eps_end - eps_start)


def _state_signature(state_arr: np.ndarray) -> bytes:
	s = np.asarray(state_arr)
	if np.issubdtype(s.dtype, np.floating):
		s = np.round(s, 4)
	s = np.ascontiguousarray(s)
	return s.tobytes()


def run_training(
	env_id: str = "MultiGrid-MultiTargetEmpty-4x4-v0",
	num_agents: int = 2,
	episodes: int = 10000,
	max_steps: int = 40,
	buffer_size: int = 100000,
	batch_size: int = 64,
	lr: float = 3e-4,
	gamma: float = 0.99,
	tau: float = 0.01,
	start_steps: int = 1000,
	epsilon_start: float = 1.0,
	epsilon_end: float = 0.05,
	epsilon_decay_steps: int = 5000,
	train_every: int = 1,
	updates_per_step: int = 1,
	save_every: int = 100,
	seed: int = 0,
	model_dir: str = "runs/qmix",
) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	env = make_refactor_multigrid_gym_env(
		env_id=env_id,
		num_agents=num_agents,
		max_steps=max_steps,
		render_mode=None,
		seed=seed,
	)

	obs0, info0 = env.reset(seed=seed)
	state0 = np.asarray(info0["state"], dtype=np.float32)
	seen_state_signatures = {_state_signature(state0)}
	obs_dim = int(obs0.shape[1])  # obs_dim = obs vector per agent
	state_dim = int(state0.shape[0])
	action_dim = int(env.action_space.nvec[0])

	agent = QMIXAgent(
		n_agents=num_agents,
		obs_dim=obs_dim,
		state_dim=state_dim,
		action_dim=action_dim,
		lr=lr,
		gamma=gamma,
		tau=tau,
	)
	replay = ReplayBuffer(capacity=buffer_size)
	metrics = TrainingMetrics()

	run_path = Path(model_dir)
	run_path.mkdir(parents=True, exist_ok=True)

	total_steps = 0
	reward_history: list[float] = []

	print(
		f"Env={env_id} | agents={num_agents} | obs_dim={obs_dim} | state_dim={state_dim} | action_dim={action_dim}"
	)
	print(f"Device={agent.device}")

	for ep in range(1, episodes + 1):

		obs, info = env.reset(seed=seed + ep)
		state = np.asarray(info["state"], dtype=np.float32)
		seen_state_signatures.add(_state_signature(state))
		obs_state = np.asarray(obs, dtype=np.float32)  # shape: (num_agents, obs_dim)

		ep_reward = 0.0
		loss_values: list[float] = []
		q_total_values: list[float] = []
		target_values: list[float] = []
		ep_len = 0

		for _ in range(max_steps):
			epsilon = linear_epsilon(
				step=total_steps,
				start_steps=0,
				decay_steps=epsilon_decay_steps,
				eps_start=epsilon_start,
				eps_end=epsilon_end,
			)
			actions = agent.select_actions(obs_state, epsilon=epsilon)
			next_obs, reward, terminated, truncated, next_info = env.step(actions)
			next_state = np.asarray(next_info["state"], dtype=np.float32)
			seen_state_signatures.add(_state_signature(next_state))
			next_obs_state = np.asarray(next_obs, dtype=np.float32)  # shape: (num_agents, obs_dim)

			# Treat both true terminal and time-limit truncation as terminal for TD bootstrap.
			done_for_bootstrap = bool(terminated or truncated)
			done_for_episode = bool(terminated or truncated)

			replay.push(
				obs=obs_state,
				state=state,
				actions=actions,
				reward=float(reward),
				next_obs=next_obs_state,
				next_state=next_state,
				done=done_for_bootstrap,
			)

			obs_state = next_obs_state
			state = next_state
			ep_reward += float(reward)
			ep_len += 1
			total_steps += 1

			if len(replay) >= batch_size and total_steps >= start_steps and total_steps % train_every == 0:
				for _ in range(updates_per_step):
					batch = replay.sample(batch_size)
					stats = agent.train_step(batch)
					loss_values.append(stats["loss"])
					q_total_values.append(stats["q_total_mean"])
					target_values.append(stats["target_mean"])

			if done_for_episode:
				break
		
		# Test Q value: truyền đúng shape (1, obs_dim)
		test_input = np.asarray(obs_state, dtype=np.float32)  # shape: (num_agents, obs_dim)
		q_val = agent.online.agent_nets[0].forward(torch.as_tensor(test_input, dtype=torch.float32, device=agent.device)).detach().cpu().numpy()
		print(f" Q value at obs: {q_val}")

		reward_history.append(ep_reward)
		mean10 = float(np.mean(reward_history[-10:]))
		mean_loss = float(np.mean(loss_values)) if loss_values else float("nan")
		mean_q_total = float(np.mean(q_total_values)) if q_total_values else float("nan")
		mean_target = float(np.mean(target_values)) if target_values else float("nan")
		metrics.append_episode(
			episode=ep,
			episode_length=ep_len,
			unique_states_seen=len(seen_state_signatures),
			reward=ep_reward,
			avg10_reward=mean10,
			loss=mean_loss,
			q_total_mean=mean_q_total,
			target_mean=mean_target,
			epsilon=epsilon,
			buffer_size=len(replay),
		)
		print(
			f"Episode {ep:4d} | Reward {ep_reward:8.3f} | Avg10 {mean10:8.3f} | "
			f"Len {ep_len:3d} | Buffer {len(replay):6d} | Eps {epsilon:.3f} | "
			f"Loss {mean_loss:.6f} | Q {mean_q_total:.6f} | Target {mean_target:.6f} | "
			f"UniqueStates {len(seen_state_signatures)}"
		)

		if ep % 10 == 0:
			metrics_json = metrics.save_json(run_path, filename="metrics_latest.json")
			metrics_png = metrics.plot(run_path, filename="metrics_latest.png")
			print(f"Saved metrics JSON: {metrics_json}")
			if metrics_png is not None:
				print(f"Saved metrics plot: {metrics_png}")

		if ep % save_every == 0:
			ckpt = run_path / f"qmix_ep{ep}.pth"
			agent.save(ckpt)
			print(f"Saved checkpoint: {ckpt}")

	final_path = run_path / "qmix_final.pth"
	agent.save(final_path)
	metrics.save_json(run_path, filename="metrics_final.json")
	metrics.plot(run_path, filename="metrics_final.png")
	print(f"Training finished. Final checkpoint: {final_path}")
	env.close()

if __name__ == "__main__":
	config_path = os.path.join(os.path.dirname(__file__), "config.json")
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with open(config_path, "r", encoding="utf-8") as f:
		config = json.load(f)

	run_training(
		env_id=config.get("env_id", "MultiGrid-MultiTargetEmpty-4x4-v0"),
		num_agents=config.get("num_agents", 2),
		episodes=config.get("episodes", 10000),
		max_steps=config.get("max_steps", 40),
		buffer_size=config.get("buffer_size", 100000),
		batch_size=config.get("batch_size", 64),
		lr=config.get("learning_rate", 3e-4),
		gamma=config.get("gamma", 0.99),
		tau=config.get("tau", 0.01),
		start_steps=config.get("start_steps", 1000),
		epsilon_start=config.get("epsilon_start", 1.0),
		epsilon_end=config.get("epsilon_end", 0.05),
		epsilon_decay_steps=config.get("epsilon_decay_steps", 5000),
		train_every=config.get("train_every", 1),
		updates_per_step=config.get("updates_per_step", 1),
		save_every=config.get("save_every", 100),
		seed=config.get("seed", 0),
		model_dir=config.get("save_path", "runs/qmix"),
	)
