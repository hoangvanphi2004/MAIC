import argparse
from utils.trainer import run_training

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Multi-Agent RL Training Script")
    parser.add_argument('--algo', type=str, default='MAIC', choices=['MAIC', 'CMASAC', 'COMA', 'ISAC', 'MASAC'],
                        help='Which algorithm to use')
    parser.add_argument('--env_id', type=str, default='MultiGrid-MultiTargetEmpty-5x5-v0', 
                        help='Multigrid environment id')
    parser.add_argument('--num_agents', type=int, default=1, help='Number of agents')
    parser.add_argument('--episodes', type=int, default=5000, help='Total training episodes')
    parser.add_argument('--steps_per_episode', type=int, default=40, help='Max steps per episode')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for updates')
    parser.add_argument('--replay_size', type=int, default=int(1e6), help='Replay buffer size')
    parser.add_argument('--start_steps', type=int, default=1000, help='Random steps before training')
    parser.add_argument('--steps_per_update', type=int, default=1, help='Update network every N episodes')
    parser.add_argument('--updates_num', type=int, default=1, help='Updates per update step')
    parser.add_argument('--save_every', type=int, default=500, help='Save model interval')
    parser.add_argument('--record_video', action='store_true', help='Record video during training')
    parser.add_argument('--video_every', type=int, default=500, help='Record video interval')
    parser.add_argument('--plot_every', type=int, default=10, help='Plot metrics interval')
    parser.add_argument('--model_dir', type=str, default='runs/', help='Directory to save metrics/models')
    
    # Algorithm Hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--auto_entropy_tuning', action='store_true', help='Use SAC auto entropy (for supported algos)')

    args = parser.parse_args()
    
    run_training(args)
