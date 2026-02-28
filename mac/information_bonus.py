import torch
import torch.nn.functional as F

try:
    from bayesian.Bayesian_sampling import EnsembleRegressor
except ImportError:
    EnsembleRegressor = None

def information_bonus(mac, states, actions_oh):
    if not mac.use_information_bonus or EnsembleRegressor is None:
        return torch.zeros(states.size(0), 1).to(mac.device)
    states_flat = states.reshape(states.size(0), -1)
    actions_oh_flat = actions_oh.reshape(actions_oh.shape[0], -1)
    ensemble_input = torch.cat([states_flat, actions_oh_flat], dim=-1)
    ensemble_input_norm = mac.input_normalizer.normalize(ensemble_input)
    try:
        mean_ens, var_total, std_total, std_ale, std_epi = mac.ensemble_regressor.mixture_mean_var(ensemble_input, return_decomposed=True)
        info_gain = torch.sum(torch.log(1 + (std_epi ** 2) / (std_ale ** 2 + 1e-8)), dim=-1, keepdim=True)
        info_gain = mac.information_bonus_normalizer.normalize(info_gain)
    except:
        info_gain = torch.zeros(states.size(0), 1, device=mac.device)
    return info_gain

def train_ensemble_model(mac, state, actions_oh, next_state):
    if not mac.use_information_bonus or EnsembleRegressor is None:
        return
    states_flat = state.reshape(state.size(0), -1)
    next_state_flat = next_state.reshape(next_state.size(0), -1)
    actions_oh_flat = actions_oh.reshape(actions_oh.shape[0], -1)
    
    next_state_norm = mac.output_normalizer.normalize(next_state_flat)

    ensemble_input = torch.cat([states_flat, actions_oh_flat], dim=-1)
    ensemble_input_norm = mac.input_normalizer.normalize(ensemble_input)
    
    mac.ensemble_regressor.train_batch(ensemble_input_norm, next_state_norm)
