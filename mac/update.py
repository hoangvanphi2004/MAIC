import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .information_bonus import information_bonus, train_ensemble_model
from .baseline import compute_counterfactual_baseline

def update_sac(mac, replay_buffer, batch_size=64):
    if len(replay_buffer) < batch_size:
        return {}
    obs, state, action, reward, next_obs, next_state, done = replay_buffer.sample(batch_size)
    obs = torch.FloatTensor(obs).permute(0, 1, 4, 2, 3).to(mac.device)
    state = torch.FloatTensor(state).permute(0, 3, 1, 2).to(mac.device)
    action = torch.LongTensor(action).to(mac.device)
    reward = torch.FloatTensor(reward).to(mac.device)
    next_obs = torch.FloatTensor(next_obs).permute(0, 1, 4, 2, 3).to(mac.device)
    next_state = torch.FloatTensor(next_state).permute(0, 3, 1, 2).to(mac.device)
    done = torch.FloatTensor(done).to(mac.device)

    action_oh = F.one_hot(action.long(), num_classes=mac.n_actions).float()
    
    if mac.use_information_bonus:
        information_bonus_val = information_bonus(mac, state, action_oh)
        states_flat = state.reshape(state.size(0), -1)
        next_state_flat = next_state.reshape(next_state.size(0), -1)
        actions_oh_flat = action_oh.reshape(action_oh.shape[0], -1)
        ensemble_input = torch.cat([states_flat, actions_oh_flat], dim=-1)
        mac.input_normalizer.update(ensemble_input)
        mac.output_normalizer.update(next_state_flat)
        mac.information_bonus_normalizer.update(information_bonus_val)

    result = {}
    
    # Critic Update
    if mac.critic_type == 'centralized':
        if reward.dim() == 2:
            reward_for_target = reward.mean(dim=1, keepdim=True)
        else:
            reward_for_target = reward.unsqueeze(1)
        
        with torch.no_grad():
            next_actions = []
            next_log_prob = 0
            for i, actor in enumerate(mac.actors):
                nobs_i = next_obs[:, i]
                a_i, logp_i, _ = actor.sample(nobs_i)
                next_actions.append(a_i.unsqueeze(1).long())
                next_log_prob = next_log_prob + logp_i
            
            next_actions_cat = torch.cat(next_actions, dim=1)
            next_actions_oh = F.one_hot(next_actions_cat.long(), num_classes=mac.n_actions).float()

            next_q1_target, next_q2_target = mac.critic_target(next_state, next_actions_oh)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            next_information_bonus_val = information_bonus(mac, next_state, next_actions_oh) if mac.use_information_bonus else 0.0
            
            alpha1 = mac.alpha1.detach() if isinstance(mac.alpha1, torch.Tensor) else mac.alpha1
            alpha2 = mac.alpha2.detach() if isinstance(mac.alpha2, torch.Tensor) else mac.alpha2
            entropy_term = alpha1 * next_log_prob.unsqueeze(1) if mac.use_entropy else 0.0
            information_bonus_term = alpha2 * next_information_bonus_val if mac.use_information_bonus else 0.0
            target_q = next_q_target - entropy_term + information_bonus_term
            target_q_value = reward_for_target + (1 - done.unsqueeze(1)) * mac.gamma * target_q
            
        q1, q2 = mac.critic(state, action_oh)
        critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value)
        
        mac.critic_optimizer.zero_grad()
        critic_loss.backward()
        mac.critic_optimizer.step()
        
        for target_param, param in zip(mac.critic_target.parameters(), mac.critic.parameters()):
            target_param.data.copy_(mac.tau * param.data + (1 - mac.tau) * target_param.data)
            
        result['critic_loss'] = critic_loss.item()
        result['q_value'] = q1.mean().item()

        if mac.use_information_bonus:
            train_ensemble_model(mac, state, action_oh, next_state)

    elif mac.critic_type == 'independent':
        total_critic_loss = 0.0
        total_q_value = 0.0
        total_critic_reg = 0.0
        
        for agent_id in range(mac.num_agents):
            obs_i = obs[:, agent_id].unsqueeze(1)
            action_i = action[:, agent_id]
            reward_i = reward[:, agent_id] if reward.dim() == 2 else reward
            next_obs_i = next_obs[:, agent_id].unsqueeze(1)
            done_i = done[:, agent_id] if done.dim() == 2 else done
            
            with torch.no_grad():
                next_action_i, next_log_prob_i, _ = mac.actors[agent_id].sample(next_obs_i.squeeze(1))
                next_action_oh_i = F.one_hot(next_action_i.long(), num_classes=mac.n_actions).float()
                
                next_q1_target, next_q2_target = mac.critic_targets[agent_id](next_state, next_action_oh_i)
                next_q_target = torch.min(next_q1_target, next_q2_target)
                
                alpha = mac.alpha.detach() if isinstance(mac.alpha, torch.Tensor) else mac.alpha
                entropy_term = alpha * next_log_prob_i.unsqueeze(1) if mac.use_entropy else 0.0
                target_q = next_q_target - entropy_term
                target_q_value = reward_i.unsqueeze(1) + (1 - done_i.unsqueeze(1)) * mac.gamma * target_q
            
            action_oh_i = F.one_hot(action_i.long(), num_classes=mac.n_actions).float()
            q1, q2 = mac.critics[agent_id](state, action_oh_i)
            
            critic_mse_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value)
            q_reg = (q1 ** 2).mean() + (q2 ** 2).mean() if mac.critic_reg_weight > 0 else 0.0
            critic_loss = critic_mse_loss + mac.critic_reg_weight * q_reg
            
            mac.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(mac.critics[agent_id].parameters(), 1.0)
            mac.critic_optimizers[agent_id].step()
            
            for target_param, param in zip(mac.critic_targets[agent_id].parameters(), mac.critics[agent_id].parameters()):
                target_param.data.copy_(mac.tau * param.data + (1 - mac.tau) * target_param.data)
            
            total_critic_loss += critic_loss.item()
            total_q_value += q1.mean().item()
            total_critic_reg += float(q_reg)
            
        result['critic_loss'] = total_critic_loss / mac.num_agents
        result['q_value'] = total_q_value / mac.num_agents
        if mac.critic_reg_weight > 0:
            result['critic_regularization'] = total_critic_reg / mac.num_agents

    # Actor Update
    with torch.no_grad():
        q1, q2 = mac.critic(state, action_oh)
        q_values = torch.min(q1, q2).squeeze(-1)

    information_bonus_val = information_bonus(mac, state, action_oh).squeeze(-1)

    total_policy_loss = 0.0
    total_entropy = 0.0
    total_advantage = 0.0
    for a_i, actor in enumerate(mac.actors):
        obs_a = obs[:, a_i]
        if mac.baseline_type == 'counterfactual':
            baseline = compute_counterfactual_baseline(mac, obs_a, state, action_oh, a_i)
        else:
            baseline = 0
        advantages = (q_values - baseline).detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        action, log_prob, action_probs = actor.sample(obs_a)
        entropy = Categorical(action_probs).entropy().mean()
        alpha1 = mac.alpha1.detach() if isinstance(mac.alpha1, torch.Tensor) else mac.alpha1
        alpha2 = mac.alpha2.detach() if isinstance(mac.alpha2, torch.Tensor) else mac.alpha2
        policy_loss_a = ((alpha1 * (log_prob + 1) - advantages - alpha2 * information_bonus_val).detach() * log_prob).mean()
        total_policy_loss += policy_loss_a
        total_entropy += entropy.item()
        total_advantage += advantages.mean().item()

    mac.actor_optimizers[a_i].zero_grad()
    total_policy_loss.backward()
    torch.nn.utils.clip_grad_norm_([p for a in mac.actors for p in a.parameters()], 1.0)
    mac.actor_optimizers[a_i].step()

    result['actor_loss'] = (total_policy_loss / mac.num_agents).item()
    result['entropy'] = total_entropy / mac.num_agents

    # Auto entropy tuning (alpha)
    if mac.auto_entropy_tuning and mac.use_entropy:
        with torch.no_grad():
            sampled_logp = 0
            sampled_action = []
            for i, actor in enumerate(mac.actors):
                obs_i = obs[:, i]
                action_i, log_prob_i, action_probs_i = actor.sample(obs_i)
                sampled_logp = sampled_logp + log_prob_i
                sampled_action.append(action_i)
            sampled_action = torch.stack(sampled_action, dim=1)
            sampled_action_oh = F.one_hot(sampled_action.long(), num_classes=mac.n_actions).float()
            
            sampled_logp_target = 0
            sampled_action_target = []
            for i, actor in enumerate(mac.actors):
                obs_i = obs[:, i]
                action_i, log_prob_i, action_probs_i = actor.sample(obs_i)
                sampled_logp_target = sampled_logp_target + log_prob_i
                sampled_action_target.append(action_i)
            sampled_action_target = torch.stack(sampled_action_target, dim=1)
            sampled_action_target_oh = F.one_hot(sampled_action_target.long(), num_classes=mac.n_actions).float()
            
        alpha1_loss = -(mac.log_alpha1 * (sampled_logp + mac.target_entropy).detach()).mean()

        mac.alpha1_optimizer.zero_grad()
        alpha1_loss.backward()
        mac.alpha1_optimizer.step()

        if mac.use_information_bonus:
            information_bonus_val = information_bonus(mac, state, sampled_action_oh).squeeze(-1)
            information_bonus_val_target = information_bonus(mac, state, sampled_action_target_oh).squeeze(-1)
            alpha2_loss = (mac.log_alpha2 * (information_bonus_val - information_bonus_val_target).detach()).mean()

            mac.alpha2_optimizer.zero_grad()
            alpha2_loss.backward()
            mac.alpha2_optimizer.step()

        mac.alpha1 = mac.log_alpha1.exp()
        mac.alpha2 = mac.log_alpha2.exp()
        result['alpha1_loss'] = alpha1_loss.item()
        result['alpha2_loss'] = alpha2_loss.item()
        result['alpha1_value'] = mac.alpha1.item()
        result['alpha2_value'] = mac.alpha2.item()

    return result
