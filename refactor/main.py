
import argparse
import torch
from train import run_training


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train CMASAC with configurable observation mode and information bonus coefficient.')
	parser.add_argument('--env', type=str, default='MultiGrid-MultiTargetEmpty-16x16-v0', help='Gym environment id (e.g. MultiGrid-PassSparse-8x8-v0).')
	parser.add_argument('--num_agents', type=int, default=2, help='Number of agents to pass to the environment constructor.')
	parser.add_argument('--simple', action='store_true', help='Use simple obs/state: obs=[x,y,dir], state=concat of all agent obs.')
	parser.add_argument('--info_coef', type=float, default=0.0, help='Set CMASAC.scaled_information_gain_coef (e.g. --info_coef 0).')
	parser.add_argument('--entropy_coef', type=float, default=0.0, help='Set CMASAC.scaled_entropy_coef (e.g. --entropy_coef 0).')
	parser.add_argument('--alpha_kl', type=float, default=1.0, help='KL regularization coefficient for actor updates.')
	parser.add_argument('--policy_update_steps', type=int, default=3, help='Number of actor updates per sampled batch using a fixed old policy snapshot.')
	args = parser.parse_args()

	obs_mode = 'simple' if args.simple else 'full'

	run_training(
		env_id=args.env,
		num_agents=args.num_agents,
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