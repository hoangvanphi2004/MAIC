from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    reward: float
    next_obs: np.ndarray
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32),
                state=np.asarray(state, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=float(done),
            )
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = random.sample(self._buffer, batch_size)
        return {
            "obs": np.stack([t.obs for t in batch], axis=0),
            "state": np.stack([t.state for t in batch], axis=0),
            "actions": np.stack([t.actions for t in batch], axis=0),
            "rewards": np.array([t.reward for t in batch], dtype=np.float32),
            "next_obs": np.stack([t.next_obs for t in batch], axis=0),
            "next_state": np.stack([t.next_state for t in batch], axis=0),
            "dones": np.array([t.done for t in batch], dtype=np.float32),
        }


class TrainingMetrics:
    def __init__(self):
        self.data: dict[str, list[float | int]] = {
            "episodes": [],
            "episode_lengths": [],
            "unique_states_seen": [],
            "episode_rewards": [],
            "avg10_rewards": [],
            "episode_losses": [],
            "q_total_means": [],
            "target_means": [],
            "epsilons": [],
            "buffer_sizes": [],
        }

    def append_episode(
        self,
        episode: int,
        episode_length: int,
        unique_states_seen: int,
        reward: float,
        avg10_reward: float,
        loss: float,
        q_total_mean: float,
        target_mean: float,
        epsilon: float,
        buffer_size: int,
    ) -> None:
        self.data["episodes"].append(int(episode))
        self.data["episode_lengths"].append(int(episode_length))
        self.data["unique_states_seen"].append(int(unique_states_seen))
        self.data["episode_rewards"].append(float(reward))
        self.data["avg10_rewards"].append(float(avg10_reward))
        self.data["episode_losses"].append(float(loss))
        self.data["q_total_means"].append(float(q_total_mean))
        self.data["target_means"].append(float(target_mean))
        self.data["epsilons"].append(float(epsilon))
        self.data["buffer_sizes"].append(int(buffer_size))

    def save_json(self, run_path: Path, filename: str = "metrics_latest.json") -> Path:
        run_path.mkdir(parents=True, exist_ok=True)
        out_path = run_path / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        return out_path

    def plot(self, run_path: Path, filename: str = "metrics_latest.png") -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skip plotting metrics.")
            return None

        if not self.data["episodes"]:
            return None

        run_path.mkdir(parents=True, exist_ok=True)
        episodes = self.data["episodes"]

        fig, axes = plt.subplots(4, 2, figsize=(13, 14))

        axes[0, 0].plot(episodes, self.data["episode_rewards"], label="Reward", linewidth=1.6)
        axes[0, 0].plot(episodes, self.data["avg10_rewards"], label="Avg10", linewidth=1.8)
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(episodes, self.data["episode_losses"], color="tab:red", linewidth=1.6)
        axes[0, 1].set_title("Episode Loss")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(episodes, self.data["q_total_means"], color="tab:orange", linewidth=1.6)
        axes[1, 0].plot(episodes, self.data["target_means"], color="tab:brown", linewidth=1.6)
        axes[1, 0].set_title("Q_total vs TD Target")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].legend(["Q_total", "TD Target"])
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(episodes, self.data["episode_lengths"], color="tab:cyan", linewidth=1.6)
        axes[1, 1].set_title("Episode Length")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].grid(alpha=0.3)

        axes[2, 0].plot(episodes, self.data["unique_states_seen"], color="tab:blue", linewidth=1.6)
        axes[2, 0].set_title("Unique States Seen (Cumulative)")
        axes[2, 0].set_xlabel("Episode")
        axes[2, 0].grid(alpha=0.3)

        axes[2, 1].plot(episodes, self.data["epsilons"], color="tab:green", linewidth=1.6)
        axes[2, 1].set_title("Epsilon")
        axes[2, 1].set_xlabel("Episode")
        axes[2, 1].grid(alpha=0.3)

        axes[3, 0].plot(episodes, self.data["buffer_sizes"], color="tab:purple", linewidth=1.6)
        axes[3, 0].set_title("Replay Buffer Size")
        axes[3, 0].set_xlabel("Episode")
        axes[3, 0].grid(alpha=0.3)

        axes[3, 1].axis("off")

        fig.tight_layout()
        out_path = run_path / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
