import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
import torch
from HUCRL import HUCRL, ReplayBuffer, EpisodeMemory, HallucinatedReplayBuffer, HallucinateMemory

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
	BATCH_SIZE = 32
	BUFFER_SIZE = 1000
	UPDATE_CRITIC_FREQ = 10
	HALLUCINATED_UPDATES = 30  # Number of hallucinated rollouts per episode

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
	hallucinated_buffer = HallucinatedReplayBuffer(BUFFER_SIZE)
	hallucinated_memory = HallucinateMemory()
	reward_history = []
	plot_count = 0
	for episode in range(EPISODES):
		state, _ = env.reset()
		episode_reward = 0
		replay_buffer = ReplayBuffer(BUFFER_SIZE)
		for t in range(MAX_STEPS):
			action, _ = agent.select_action(state)
			action = action[0].item()
			next_state, reward, terminated, truncated, _ = env.step(action)
			if t % 100 == 0:
				state_tensor = torch.FloatTensor(state).unsqueeze(0)
				action_one_hot = torch.zeros((1, action_dim))
				action_one_hot[0, action] = 1.0
				ensemble_input = torch.cat([state_tensor, action_one_hot], dim=-1)
				if(agent.dynamics_model.model is not None):
					predicted_next_state = agent.dynamics_model.sample(ensemble_input)
					print(f"Step {t}: Predicted next state: {predicted_next_state}, Actual next state: {next_state}")
			done = terminated or truncated
			replay_buffer.push(state, action, reward, next_state, float(done))
			state = next_state
			episode_reward += reward
			if done:
				break
		agent.train_ensemble_model(replay_buffer, batch_size=len(replay_buffer), epochs=30)
		reward_history.append(episode_reward)
		print(f"Episode: {episode}, Reward: {episode_reward}")

		state, _ = env.reset()
		hallucinated_memory.clear()
		# Multiple hallucinated rollouts and policy updates per episode
		for _ in range(HALLUCINATED_UPDATES):
			hallucinated_memory.clear()
			state, _ = env.reset()
			hallucinated_memory = agent.roll_out_hallucinated_next_state(
				state,
				horizons=MAX_STEPS,
				hallucinated_memory=hallucinated_memory,
				hallucinated_buffer=hallucinated_buffer,
			)
			#print(f"Hallucinated rollout length: {len(hallucinated_memory)}")
			agent.update_reinforce(hallucinated_memory)
			agent.update_sac(hallucinated_memory, batch_size=BATCH_SIZE)

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
