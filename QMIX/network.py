import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentQNetwork(nn.Module):
    """Per-agent feed-forward Q-network for current-step inference."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc_q(x)
        return q_values


class MixingNetwork(nn.Module):
    """State-conditioned monotonic mixer in QMIX."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_hidden_dim = mixing_hidden_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * mixing_hidden_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, mixing_hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: [batch_size, n_agents]
            state: [batch_size, state_dim]
        Returns:
            q_total: [batch_size, 1]
        """
        batch_size = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_hidden_dim)

        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze(1)


class QMIXModel(nn.Module):
    """Model-only QMIX container (current state only, no recurrence)."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        agent_hidden_dim: int = 64,
        mixing_hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.agent_nets = nn.ModuleList(
            [
                AgentQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=agent_hidden_dim)
                for _ in range(n_agents)
            ]
        )
        self.mixer = MixingNetwork(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_hidden_dim=mixing_hidden_dim,
            hypernet_hidden_dim=hypernet_hidden_dim,
        )

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [batch_size, n_agents, obs_dim]
            state: [batch_size, state_dim]
            actions: [batch_size, n_agents] or [batch_size, n_agents, 1]

        Returns:
            agent_qs: [batch_size, n_agents, action_dim]
            q_total: [batch_size, 1]
        """
        batch_size = obs.size(0)

        per_agent_qs = []
        for agent_idx, agent_net in enumerate(self.agent_nets):
            agent_obs = obs[:, agent_idx, :]
            agent_q = agent_net(agent_obs)
            per_agent_qs.append(agent_q)

        agent_qs = torch.stack(per_agent_qs, dim=1)

        if actions is None:
            chosen_agent_qs = agent_qs.max(dim=-1).values
        else:
            if actions.dim() == 3 and actions.size(-1) == 1:
                actions = actions.squeeze(-1)
            chosen_agent_qs = agent_qs.gather(dim=-1, index=actions.long().unsqueeze(-1)).squeeze(-1)

        q_total = self.mixer(chosen_agent_qs, state)

        return agent_qs, q_total

