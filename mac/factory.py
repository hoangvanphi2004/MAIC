from .core import MAC

def create_maic(*args, **kwargs):
    return MAC(*args, **kwargs, critic_type='centralized',
               baseline_type='counterfactual', use_information_bonus=True, use_entropy=True)

def create_cmasac(*args, **kwargs):
    return MAC(*args, **kwargs, critic_type='centralized',
               baseline_type='counterfactual', use_information_bonus=False, use_entropy=True)

def create_isac(*args, **kwargs):
    return MAC(*args, **kwargs, critic_type='independent',
               baseline_type=None, use_information_bonus=False, use_entropy=True,
               reg_weight=0.001, critic_reg_weight=0.001)

def create_masac(*args, **kwargs):
    return MAC(*args, **kwargs, critic_type='centralized',
               baseline_type=None, use_information_bonus=False, use_entropy=True)

def create_coma(*args, **kwargs):
    return MAC(*args, **kwargs, critic_type='centralized',
               baseline_type='counterfactual', use_information_bonus=False, use_entropy=False)
