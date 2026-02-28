import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from components import Actor, Critic, Normalizer

try:
    from bayesian.Bayesian_sampling import EnsembleRegressor
except ImportError:
    EnsembleRegressor = None

from .information_bonus import information_bonus, train_ensemble_model
from .update import update_sac
from .baseline import compute_counterfactual_baseline

class MAC:
    def __init__(self, obs_shape, state_shape, action_dim, num_agents=2, hidden_dim=128, lr=3e-4, gamma=0.99,
                 tau=0.01, alpha=0.01, auto_entropy_tuning=False,
                 critic_type='centralized',
                 baseline_type='counterfactual', use_information_bonus=False,
                 use_entropy=True, reg_weight=0.0, critic_reg_weight=0.0):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.n_actions = action_dim
        self.num_agents = num_agents
        
        self.critic_type = critic_type
        self.baseline_type = baseline_type
        self.use_information_bonus = use_information_bonus
        self.use_entropy = use_entropy
        
        self.reg_weight = reg_weight
        self.critic_reg_weight = critic_reg_weight

        # Actors (always independent policies)
        self.actors = [Actor(obs_shape, action_dim, hidden_dim).to(self.device) for _ in range(self.num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]

        # Target actors
        self.actors_target = [Actor(obs_shape, action_dim, hidden_dim).to(self.device) for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

        # Critic
        if self.critic_type == 'centralized':
            self.critic = Critic(self.num_agents, state_shape, action_dim_per_agent=self.n_actions, hidden_dim=256).to(self.device)
            self.critic_target = Critic(self.num_agents, state_shape, action_dim_per_agent=self.n_actions, hidden_dim=256).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        elif self.critic_type == 'independent':
            self.critics = [Critic(1, state_shape, action_dim_per_agent=self.n_actions, hidden_dim=256).to(self.device) for _ in range(self.num_agents)]
            self.critic_targets = [Critic(1, state_shape, action_dim_per_agent=self.n_actions, hidden_dim=256).to(self.device) for _ in range(self.num_agents)]
            for i in range(self.num_agents):
                self.critic_targets[i].load_state_dict(self.critics[i].state_dict())
            self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr, weight_decay=1e-5 if self.critic_reg_weight > 0 else 0) for critic in self.critics]

        # Entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning and self.use_entropy:
            self.target_entropy = -float(self.num_agents * action_dim)
            self.log_alpha1 = torch.zeros(1, requires_grad=True, device=self.device)
            self.log_alpha2 = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha1_optimizer = optim.Adam([self.log_alpha1], lr=lr)
            self.alpha2_optimizer = optim.Adam([self.log_alpha2], lr=lr)
            self.alpha1 = self.log_alpha1.exp()
            self.alpha2 = self.log_alpha2.exp()
        else:
            self.alpha1 = alpha
            self.alpha2 = alpha

        # Information Bonus (MAIC)
        if self.use_information_bonus and EnsembleRegressor is not None:
            self.input_ensemble_shape = state_shape[0] * state_shape[1] * state_shape[2] + action_dim * num_agents
            self.output_ensemble_shape = state_shape[0] * state_shape[1] * state_shape[2]
            self.ensemble_regressor = EnsembleRegressor(in_dim=self.input_ensemble_shape, out_dim=self.output_ensemble_shape, M=5, hidden=256, device=self.device)
            self.ensemble_regressor.setup_optimizers(lr)
            
            self.input_normalizer = Normalizer(self.input_ensemble_shape, device=self.device)
            self.output_normalizer = Normalizer(self.output_ensemble_shape, device=self.device)
            self.information_bonus_normalizer = Normalizer(1, device=self.device)

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state_arr = np.array(state)
            if state_arr.ndim == 3:
                state_arr = state_arr[np.newaxis, ...]
            actions = []
            log_probs = []
            for i, actor in enumerate(self.actors):
                s = torch.FloatTensor(state_arr[i]).permute(2, 0, 1).unsqueeze(0).to(self.device)
                if evaluate:
                    action_probs = actor(s)
                    action = torch.argmax(action_probs, dim=-1)
                    actions.append(int(action.item()))
                    log_probs.append(None)
                else:
                    action, log_prob, _ = actor.sample(s)
                    actions.append(int(action.item()))
                    log_probs.append(float(log_prob.item()))
            return actions, log_probs

    def update(self, replay_buffer, batch_size=64):
        return update_sac(self, replay_buffer, batch_size)

    def save(self, filename):
        save_dict = {
            'actors': [a.state_dict() for a in self.actors]
        }
        if self.critic_type == 'centralized':
            save_dict['critic'] = self.critic.state_dict()
            save_dict['critic_target'] = self.critic_target.state_dict()
        else:
            save_dict['critics'] = [c.state_dict() for c in self.critics]
            save_dict['critic_targets'] = [ct.state_dict() for ct in self.critic_targets]
        torch.save(save_dict, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        if 'actors' in checkpoint:
            for a, st in zip(self.actors, checkpoint['actors']):
                a.load_state_dict(st)
        if self.critic_type == 'centralized':
            if 'critic' in checkpoint: self.critic.load_state_dict(checkpoint['critic'])
            if 'critic_target' in checkpoint: self.critic_target.load_state_dict(checkpoint['critic_target'])
        else:
            if 'critics' in checkpoint:
                for c, st in zip(self.critics, checkpoint['critics']): c.load_state_dict(st)
            if 'critic_targets' in checkpoint:
                for ct, st in zip(self.critic_targets, checkpoint['critic_targets']): ct.load_state_dict(st)
