from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.q_network import QNetwork
from src.agents.replay_buffer import ReplayBuffer


def apply_mask(q_values: torch.Tensor, valid_masks: torch.Tensor) -> torch.Tensor:
    invalid = valid_masks <= 0.5
    masked = q_values.masked_fill(invalid, -1e9)
    all_invalid = invalid.all(dim=1, keepdim=True)
    return torch.where(all_invalid, q_values, masked)


def compute_next_target(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q_values: torch.Tensor,
    next_valid_masks: torch.Tensor,
    gamma: float,
    use_target_mask: bool,
) -> torch.Tensor:
    if use_target_mask:
        next_q_values = apply_mask(next_q_values, next_valid_masks)
    next_max = next_q_values.max(dim=1, keepdim=True).values
    return rewards + gamma * next_max * (1.0 - dones)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50000
    min_buffer_size: int = 1000
    target_update_freq: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30000
    use_temporal_difference: bool = False
    use_behavior_mask: bool = False
    use_target_mask: bool = False
    device: str = "cpu"
    seed: int = 0


class DQNAgent:
    def __init__(self, *, config: DQNConfig, state_dim: int = 60, action_dim: int = 5) -> None:
        self.cfg = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(config.device)
        self._seed_all(config.seed)

        self.q_net = QNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_q_net = QNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=config.buffer_size)
        self.rng = np.random.default_rng(config.seed)
        self.global_step = 0

    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_epsilon(self) -> float:
        t = min(1.0, self.global_step / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + t * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, state: np.ndarray, valid_mask: np.ndarray | None = None, greedy: bool = False) -> int:
        action, _ = self.act_with_info(state, valid_mask=valid_mask, greedy=greedy)
        return action

    def act_with_info(
        self,
        state: np.ndarray,
        valid_mask: np.ndarray | None = None,
        greedy: bool = False,
    ) -> tuple[int, int]:
        if (not greedy) and self.rng.random() < self.get_epsilon():
            candidate = int(self.rng.integers(self.action_dim))
            intervention = 0
            if self.cfg.use_behavior_mask and valid_mask is not None:
                valid_actions = np.flatnonzero(valid_mask > 0.5)
                if len(valid_actions) > 0 and valid_mask[candidate] <= 0.5:
                    candidate = int(self.rng.choice(valid_actions))
                    intervention = 1
            return candidate, intervention

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            unmasked_argmax = int(torch.argmax(q_values, dim=1).item())
            if self.cfg.use_behavior_mask and valid_mask is not None:
                q_values = apply_mask(
                    q_values,
                    torch.tensor(valid_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
                )
                masked_argmax = int(torch.argmax(q_values, dim=1).item())
                intervention = int(valid_mask[unmasked_argmax] <= 0.5 and masked_argmax != unmasked_argmax)
                return masked_argmax, intervention
        return int(torch.argmax(q_values, dim=1).item()), 0

    def store(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        valid_mask: np.ndarray,
        next_valid_mask: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            valid_mask=valid_mask,
            next_valid_mask=next_valid_mask,
            info=info,
        )

    def update(self) -> dict[str, float] | None:
        if len(self.replay_buffer) < max(self.cfg.min_buffer_size, self.cfg.batch_size):
            return None
        batch = self.replay_buffer.sample(self.cfg.batch_size)
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_valid_masks = torch.tensor(batch["next_valid_masks"], dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states)
            target = compute_next_target(
                rewards=rewards,
                dones=dones,
                next_q_values=next_q_values,
                next_valid_masks=next_valid_masks,
                gamma=self.cfg.gamma,
                use_target_mask=self.cfg.use_target_mask,
            )
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.cfg.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        return {"loss": float(loss.item())}

    def save(self, path: str) -> None:
        torch.save({"q_net": self.q_net.state_dict(), "target_q_net": self.target_q_net.state_dict()}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(payload["q_net"])
        self.target_q_net.load_state_dict(payload["target_q_net"])
