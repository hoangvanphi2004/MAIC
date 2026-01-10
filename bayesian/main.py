import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
import torch
from HUCRL import HUCRL, ReplayBuffer, EpisodeMemory

def train_sac_reinforce_cartpole():
	# Select environment: "mountaincar" or "cartpole"
	ENV_OPTION = "cartpole"
	if ENV_OPTION == "mountaincar":
		ENV_NAME = 'MountainCar-v0'
		MAX_STEPS = 200
		plot_title = "MountainCar Reward per Episode"
		plot_file = "mountaincar_rewards.png"
		# Reward: +1 if goal reached, -1 otherwise
		def reward_fn(state, action):
			position = float(np.array(state)[0])
			return 1.0 if position >= 0.5 else -1.0
		# Done when reaching goal position
		def done_fn(state):
			s = np.array(state).reshape(-1)
			return float(s[0]) >= 0.5
	elif ENV_OPTION == "cartpole":
		ENV_NAME = 'CartPole-v1'
		MAX_STEPS = 500
		plot_title = "CartPole Reward per Episode"
		plot_file = "cartpole_rewards.png"
		# Reward: +1 per step
		def reward_fn(state, action):
			return 1.0
		# Done when cart or pole exceeds thresholds
		def done_fn(state):
			s = np.array(state).reshape(-1)
			x = float(s[0])
			theta = float(s[2])
			x_threshold = 2.4
			theta_threshold = 12 * 2 * np.pi / 360
			return (abs(x) > x_threshold) or (abs(theta) > theta_threshold)
	else:
		raise ValueError("ENV_OPTION must be 'mountaincar' or 'cartpole'")

	EPISODES = 1000
	BATCH_SIZE = 64
	BUFFER_SIZE = 1000
	UPDATE_CRITIC_FREQ = 10
	HALLUCINATED_UPDATES = 20  # Number of hallucinated rollouts per episode

	env = gym.make(ENV_NAME, render_mode="rgb_array")
	env = RecordVideo(
		env,
		video_folder="videos",
		episode_trigger=lambda episode_id: episode_id >= 10 and (episode_id % 10 == 0)
	)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n

	agent = HUCRL(
		state_dim,
		action_dim,
		hidden_dim=128,
		lr=5e-4,
		gamma=0.99,
		tau=0.01,
		alpha1=0.01,
		alpha2=0.01,
		auto_entropy_tuning=False,
		reward_function=reward_fn,
		done_function=done_fn,
		num_ensembles=5,
		beta=1,
	)
	replay_buffer = ReplayBuffer(BUFFER_SIZE)
	hallucinated_buffer = ReplayBuffer(BUFFER_SIZE)
	hallucinated_memory = EpisodeMemory()
	reward_history = []
	for episode in range(EPISODES):
		state, _ = env.reset()
		episode_reward = 0
		hallucinated_memory.clear()
		# Multiple hallucinated rollouts and policy updates per episode
		for _ in range(HALLUCINATED_UPDATES):
			hallucinated_memory.clear()
			hallucinated_memory = agent.roll_out_hallucinated_next_state(
				state,
				horizons=MAX_STEPS,
				hallucinated_memory=hallucinated_memory,
				hallucinated_buffer=hallucinated_buffer,
			)
			agent.update_reinforce(hallucinated_memory)
			# Also update SAC inside hallucination loop
			if len(hallucinated_buffer) > BATCH_SIZE and (episode * HALLUCINATED_UPDATES + _ + 1) % UPDATE_CRITIC_FREQ == 0:
				for _ in range(HALLUCINATED_UPDATES):
					agent.update_sac(hallucinated_buffer, batch_size=BATCH_SIZE)
		for t in range(MAX_STEPS):
			action, _ = agent.select_action(state)
			next_state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			replay_buffer.push(state, action, reward, next_state, float(done))
			if(t % 200 == 0):
				input_ensemble = np.concatenate([state, np.eye(action_dim)[action]], axis=-1)
				mu, std = agent.ensemble_regressor.mixture_mean_var(torch.as_tensor(input_ensemble, dtype=torch.float32, device=agent.device).unsqueeze(0))
				print(f"Episode: {episode}, \n next_state prediction {mu.squeeze(0).detach().cpu().numpy()}, \n next_state actual: {next_state}")
			state = next_state
			episode_reward += reward
			if done:
				break
		agent.train_ensemble_model(replay_buffer, batch_size=BATCH_SIZE, epochs=20)
		reward_history.append(episode_reward)
		print(f"Episode: {episode}, Reward: {episode_reward}")

		# Overwrite the same plot files every 10 episodes
		if (episode + 1) % 10 == 0:
			# Reward curve
			plt.figure(figsize=(8, 4))
			plt.plot(reward_history, label="Episode reward")
			plt.xlabel("Episode")
			plt.ylabel("Reward")
			plt.title(plot_title)
			plt.legend()
			plt.tight_layout()
			plt.savefig(plot_file)
			plt.close()
	# Save ensemble uncertainty plot once at the end
	env.close()
	# Plot reward curve
	plt.figure(figsize=(8, 4))
	plt.plot(reward_history, label="Episode reward")
	plt.xlabel("Episode")
	plt.ylabel("Reward")
	plt.title(plot_title)
	plt.legend()
	plt.tight_layout()
	plt.savefig(plot_file)
	plt.close()

if __name__ == "__main__":
	train_sac_reinforce_cartpole()
