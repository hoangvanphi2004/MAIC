from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np


def _ensure_refactor_multigrid_on_path() -> None:
    """Allow importing multigrid from refactor/multigrid without editing that folder."""
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "refactor" / "multigrid"
    candidate_str = str(candidate)
    if candidate.is_dir() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


_ensure_refactor_multigrid_on_path()

import multigrid  # noqa: F401  # Registers environments
import multigrid.envs  # noqa: F401


class RefactorMultigridGymAdapter(gym.Env):
    """
    Wrap refactor.multigrid multi-agent env into a single Gym-style env.

    Observation:
        np.ndarray with shape (num_agents, 4), using the same simple features as refactor/train.py:
        [x, y, direction, carrying_flag]

    Action:
        gym.spaces.MultiDiscrete([action_dim] * num_agents)

    Reward:
        Shared scalar reward = sum of per-agent rewards.

    Extra info:
        info['state'] contains the simple global state vector compatible with refactor/train.py.
        info['per_agent_rewards'] contains per-agent rewards as np.ndarray.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        env_id: str = "MultiGrid-MultiTargetEmpty-16x16-v0",
        num_agents: int = 2,
        max_steps: int = 40,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.env_id = env_id
        self.num_agents = int(num_agents)

        make_kwargs: dict[str, Any] = {
            "num_agents": self.num_agents,
            "max_steps": int(max_steps),
        }
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode

        self._env = gym.make(self.env_id, **make_kwargs)

        # Probe once to lock observation/state structure for Gym spaces.
        obs_dict, _ = self._env.reset(seed=seed)
        self._use_door_feature = getattr(self._env.unwrapped, "door_obj", None) is not None
        self._use_box_feature = getattr(self._env.unwrapped, "box_center", None) is not None
        obs_arr, state_arr = self._extract_simple_obs_state(obs_dict)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_arr.shape,
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=state_arr.shape,
            dtype=np.float32,
        )

        action_dim = int(self._env.action_space[0].n)
        self.action_space = gym.spaces.MultiDiscrete(np.full(self.num_agents, action_dim, dtype=np.int64))

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def _extract_simple_obs_state(self, _obs_dict: dict[int, Any]) -> tuple[np.ndarray, np.ndarray]:
        # Keep the same simple feature logic used in refactor/train.py.
        agent_states = self._env.unwrapped.agent_states
        obs_simple = []
        for i in range(self.num_agents):
            x, y = agent_states.pos[i]
            direction = agent_states.dir[i]
            carrying_flag = 1.0 if self._env.unwrapped.agents[i].state.carrying is not None else 0.0
            obs_simple.append(np.array([x, y, direction, carrying_flag], dtype=np.float32))
        obs_out = np.stack(obs_simple, axis=0)

        env_unwrapped = self._env.unwrapped
        env_features: list[float] = []

        if self._use_door_feature:
            door_obj = getattr(env_unwrapped, "door_obj", None)
            env_features.append(1.0 if bool(getattr(door_obj, "is_open", False)) else 0.0)

        if self._use_box_feature:
            box_center = getattr(env_unwrapped, "box_center", None)
            if box_center is None:
                env_features.extend([0.0, 0.0])
            else:
                env_features.extend([float(box_center[0]), float(box_center[1])])

        flat_obs = obs_out.reshape(-1).astype(np.float32)
        if env_features:
            state_out = np.concatenate([flat_obs, np.array(env_features, dtype=np.float32)], axis=0)
        else:
            state_out = flat_obs

        return obs_out, state_out

    def _normalize_actions(self, actions: dict[int, int] | list[int] | np.ndarray) -> dict[int, int]:
        if isinstance(actions, dict):
            return {int(k): int(v) for k, v in actions.items()}

        action_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
        if action_arr.shape[0] != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {action_arr.shape[0]}")
        return {i: int(action_arr[i]) for i in range(self.num_agents)}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs_dict, info = self._env.reset(seed=seed, options=options)
        obs, state = self._extract_simple_obs_state(obs_dict)

        info_out = dict(info)
        info_out["state"] = state
        return obs, info_out

    def step(self, actions: dict[int, int] | list[int] | np.ndarray):
        action_dict = self._normalize_actions(actions)
        obs_dict, rewards_dict, terminations, truncations, info = self._env.step(action_dict)

        obs, state = self._extract_simple_obs_state(obs_dict)
        per_agent_rewards = np.array([float(rewards_dict[i]) for i in range(self.num_agents)], dtype=np.float32)
        reward = float(per_agent_rewards.sum())
        terminated = bool(all(bool(v) for v in terminations.values()))
        truncated = bool(all(bool(v) for v in truncations.values()))

        info_out = dict(info)
        info_out["state"] = state
        info_out["per_agent_rewards"] = per_agent_rewards

        return obs, reward, terminated, truncated, info_out

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def make_refactor_multigrid_gym_env(
    env_id: str = "MultiGrid-MultiTargetEmpty-16x16-v0",
    num_agents: int = 2,
    max_steps: int = 40,
    render_mode: str | None = None,
    seed: int | None = None,
) -> RefactorMultigridGymAdapter:
    return RefactorMultigridGymAdapter(
        env_id=env_id,
        num_agents=num_agents,
        max_steps=max_steps,
        render_mode=render_mode,
        seed=seed,
    )


if __name__ == "__main__":
    env = make_refactor_multigrid_gym_env(
        env_id="MultiGrid-MultiTargetEmpty-4x4-v0",
        num_agents=2,
        max_steps=40,
    )
    obs, info = env.reset(seed=0)
    print("obs shape:", obs.shape)
    print("state shape:", info["state"].shape)

    done = False
    ep_reward = 0.0
    while not done:
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)
        # print("obs:", obs, "actions:", actions, "reward:", reward, "terminated:", terminated, "truncated:", truncated)
        ep_reward += reward
        done = terminated or truncated

    print("episode reward:", ep_reward)
    env.close()