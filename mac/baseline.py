import torch
import torch.nn.functional as F

def compute_counterfactual_baseline(mac, obs_a, state, actions_oh, agent_idx):
    with torch.no_grad():
        B = state.size(0)
        _, _, action_probs = mac.actors[agent_idx].sample(obs_a)
        q_values_per_action = []
        for a in range(mac.n_actions):
            candidate_actions = actions_oh.clone()
            candidate_actions[:, agent_idx, :] = 0.0
            candidate_actions[:, agent_idx, a] = 1.0
            q1, q2 = mac.critic(state, candidate_actions)
            q_val = torch.min(q1, q2).squeeze(-1)
            q_values_per_action.append(q_val)
        q_stack = torch.stack(q_values_per_action, dim=1)
        baseline = (action_probs * q_stack).sum(dim=1)
    return baseline
