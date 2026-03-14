from typing import List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class CNNBaseNet(nn.Module):
    """Simplified CNN dynamics model: (state, action) -> next_state distribution."""
    
    def __init__(self, state_shape: Tuple[int, int, int], action_dim: int, out_dim: int = 1, 
                 hidden: int = 64, nlayers: int = 2, use_random_prior: bool = True, 
                 init_seed: Optional[int] = None):
        super().__init__()
        
        # Set unique random seed for diverse initialization
        if init_seed is not None:
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
        
        self.state_shape = state_shape  # (C, H, W)
        self.action_dim = action_dim
        self.state_channels = state_shape[0]
        self.state_h = state_shape[1]
        self.state_w = state_shape[2]
        self.flat_state_dim = self.state_channels * self.state_h * self.state_w
        
        # Simplified encoder: just 2 conv layers
        self.conv1 = nn.Conv2d(self.state_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32 * self.state_h * self.state_w),
        )

        # Decoder
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 64 = 32 (conv) + 32 (action)
        
        # Output heads
        self.mu_head = nn.Conv2d(32, self.state_channels, kernel_size=1)
        self.var_head = nn.Conv2d(32, self.state_channels, kernel_size=1)

        # Optional projection if caller requests out_dim != flattened state size.
        self.out_dim = out_dim
        self.mu_proj = None
        self.var_proj = None
        if out_dim != self.flat_state_dim:
            self.mu_proj = nn.Linear(self.flat_state_dim, out_dim)
            self.var_proj = nn.Linear(self.flat_state_dim, out_dim)

        # Random prior layer (frozen) for diversity on unseen states
        self.use_random_prior = use_random_prior
        
        if use_random_prior:
            self.random_projection = nn.Conv2d(self.state_channels, self.state_channels, kernel_size=1, bias=True)
            # Initialize with diverse weights and FREEZE
            nn.init.orthogonal_(self.random_projection.weight, gain=np.random.uniform(0.5, 2.0))
            nn.init.uniform_(self.random_projection.bias, -1.0, 1.0)
            # Freeze this layer
            for param in self.random_projection.parameters():
                param.requires_grad = False
        else:
            self.random_projection = None
        
        # Diverse initialization with different scales
        init_scale = np.random.uniform(0.5, 2.0) if init_seed is not None else 1.0
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.random_projection:
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2) * init_scale)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2) * init_scale)
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.var_head.bias, -1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, flat_state_dim + action_dim)
        # Split into state and action.
        batch_size = x.shape[0]
        state_dim = self.flat_state_dim
        
        state_flat = x[:, :state_dim]  # (batch, C*H*W)
        action = x[:, state_dim:]  # (batch, action_dim)
        
        # Reshape state image
        state_img = state_flat.view(batch_size, *self.state_shape)
        model_in = state_img

        # Apply frozen random prior in image space.
        if self.use_random_prior:
            model_in = model_in + torch.tanh(self.random_projection(model_in))

        # Simple encoder
        x1 = F.relu(self.conv1(model_in))
        x2 = F.relu(self.conv2(x1))
        
        # Encode action and reshape to match spatial dimensions
        action_encoded = self.action_encoder(action)
        action_map = action_encoded.view(batch_size, 32, self.state_h, self.state_w)
        
        # Concatenate features and action
        combined = torch.cat([x2, action_map], dim=1)
        
        # Decoder
        x3 = F.relu(self.conv3(combined))

        # Output heads
        mu_map = self.mu_head(x3)
        var_map = F.softplus(self.var_head(x3)) + 1e-6

        mu = mu_map.flatten(start_dim=1)
        var = var_map.flatten(start_dim=1)

        if self.mu_proj is not None and self.var_proj is not None:
            mu = self.mu_proj(mu)
            var = F.softplus(self.var_proj(var)) + 1e-6

        return mu, var


def gaussian_nll(mu: torch.Tensor, var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 0.5 * log(2*pi) is approx 0.9189, using math.log(2 * math.pi) directly
    return 0.5 * (math.log(2 * math.pi) + torch.log(var) + (y - mu) ** 2 / var)


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = x.clone().detach().requires_grad_(True)
    mu, var = model(x_adv) # Updated to return var instead of logvar
    loss = gaussian_nll(mu, var, y).mean()
    # Use torch.autograd.grad to prevent accumulating gradients in model parameters
    grad = torch.autograd.grad(loss, x_adv)[0]
    if grad is None:
        return x.detach()
    x_adv = x_adv + eps * grad.sign()
    return x_adv.detach()


class CNNEnsembleRegressor:
    """Ensemble regressor using CNN-based models for image states"""
    
    def __init__(self, M: int, state_shape: Tuple[int, int, int], action_dim: int, 
                 out_dim: int = 1, hidden: int = 128, nlayers: int = 2, 
                 device: Optional[torch.device] = None, use_random_prior: bool = True, 
                 base_seed: int = 42):
        self.M = M
        self.out_dim = out_dim
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # Create models with diverse initializations
        self.models: List[CNNBaseNet] = []
        for i in range(M):
            # Each model gets a unique seed for diverse initialization
            model_seed = base_seed + i * 1000
            model = CNNBaseNet(state_shape, action_dim, out_dim, hidden, nlayers, 
                             use_random_prior=use_random_prior, 
                             init_seed=model_seed).to(self.device)
            self.models.append(model)
        
        self.optimizers: Optional[List[optim.Optimizer]] = None

    def parameters(self):
        params = []
        for m in self.models:
            params += list(m.parameters())
        return params

    def to(self, device: torch.device):
        for m in self.models:
            m.to(device)
        self.device = device

    def eval(self):
        for m in self.models:
            m.eval()

    def train(self):
        for m in self.models:
            m.train()

    def predict_per_model(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        mus = []
        vars_ = []
        for m in self.models:
            mu, var = m(x)
            mus.append(mu.unsqueeze(0))
            vars_.append(var.unsqueeze(0))
        mus = torch.cat(mus, dim=0)
        vars_ = torch.cat(vars_, dim=0)
        return mus, vars_

    def mixture_mean_var(self, x: torch.Tensor, return_decomposed: bool = False):
        mus, vars_ = self.predict_per_model(x)
        mean_ens = mus.mean(dim=0)

        var_ale = vars_.mean(dim=0)
        var_epi = ((mus - mean_ens) ** 2).mean(dim=0)
        var_total = var_ale + var_epi

        var_total = torch.clamp(var_total, min=1e-6)

        if not return_decomposed:
            return mean_ens, var_total

        std_total = torch.sqrt(torch.clamp(var_total, min=1e-12))
        std_ale = torch.sqrt(torch.clamp(var_ale, min=1e-12))
        std_epi = torch.sqrt(torch.clamp(var_epi, min=1e-12))
        return mean_ens, var_total, std_total, std_ale, std_epi

    def sample_from_mixture(self, x: torch.Tensor, n_samples: int = 1, use_gaussian_approx: bool = False) -> torch.Tensor:
        x = x.to(self.device)
        batch = x.shape[0]
        if use_gaussian_approx:
            mean_ens, var_ens = self.mixture_mean_var(x)
            std = torch.sqrt(var_ens)
            eps = torch.randn(n_samples, batch, self.out_dim, device=self.device)
            return mean_ens.unsqueeze(0) + eps * std.unsqueeze(0)

        mus, vars_ = self.predict_per_model(x)
        comps = torch.randint(0, self.M, size=(n_samples, batch), device=self.device)
        comps_exp = comps.unsqueeze(-1).expand(-1, -1, self.out_dim)
        mu_samples = torch.gather(mus, 0, comps_exp)
        var_samples = torch.gather(vars_, 0, comps_exp)
        std_samples = torch.sqrt(var_samples)
        eps = torch.randn(n_samples, batch, self.out_dim, device=self.device)
        return mu_samples + std_samples * eps

    def setup_optimizers(self, lr: float = 1e-3, weight_decay: float = 1e-5):
        self.optimizers = [optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay) for m in self.models]

    def train_batch(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
        eps_adv: float = 0.0,
        y_mean: Optional[torch.Tensor] = None,
        y_std: Optional[torch.Tensor] = None,
        normalize_for_loss: bool = False,
    ) -> float:
        if self.optimizers is None:
            raise RuntimeError("Optimizers not initialized. Call setup_optimizers() first.")

        xb = xb.to(self.device)
        yb = yb.to(self.device).float()

        if normalize_for_loss:
            if y_mean is None or y_std is None:
                raise ValueError("y_mean and y_std must be provided when normalize_for_loss=True")
            y_mean = y_mean.to(self.device).float()
            y_std = y_std.to(self.device).float()
            norm_denom = torch.clamp(y_std, min=1e-8)

        total_loss = 0.0
        batch_size = xb.shape[0]

        for i, m in enumerate(self.models):
            m.train()
            self.optimizers[i].zero_grad()

            # Bootstrapping: Randomly sample (with replacement) for diversity
            indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
            xb_model = xb[indices]
            yb_model = yb[indices]

            mu, var = m(xb_model)
            if normalize_for_loss:
                yb_model_norm = (yb_model - y_mean) / norm_denom
                mu_norm = (mu - y_mean) / norm_denom
                var_norm = torch.clamp(var / (norm_denom ** 2), min=1e-8)
                loss = gaussian_nll(mu_norm, var_norm, yb_model_norm).mean()
            else:
                loss = gaussian_nll(mu, var, yb_model).mean()

            if eps_adv > 0.0:
                xb_adv = fgsm_attack(m, xb_model, yb_model, eps_adv)
                mu2, var2 = m(xb_adv)
                if normalize_for_loss:
                    mu2_norm = (mu2 - y_mean) / norm_denom
                    var2_norm = torch.clamp(var2 / (norm_denom ** 2), min=1e-8)
                    loss = loss + gaussian_nll(mu2_norm, var2_norm, yb_model_norm).mean()
                else:
                    loss = loss + gaussian_nll(mu2, var2, yb_model).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            self.optimizers[i].step()
            total_loss += loss.item()

        return total_loss