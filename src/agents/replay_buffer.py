from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 50000) -> None:
        self.buffer: deque[dict[str, Any]] = deque(maxlen=capacity)

    def add(
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
        self.buffer.append(
            {
                "state": np.asarray(state, dtype=np.float32).copy(),
                "action": int(action),
                "reward": float(reward),
                "next_state": np.asarray(next_state, dtype=np.float32).copy(),
                "done": float(done),
                "valid_mask": np.asarray(valid_mask, dtype=np.float32).copy(),
                "next_valid_mask": np.asarray(next_valid_mask, dtype=np.float32).copy(),
                "info": dict(info),
            }
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        rows = [self.buffer[i] for i in indices]
        return {
            "states": np.stack([row["state"] for row in rows]),
            "actions": np.asarray([row["action"] for row in rows], dtype=np.int64),
            "rewards": np.asarray([row["reward"] for row in rows], dtype=np.float32),
            "next_states": np.stack([row["next_state"] for row in rows]),
            "dones": np.asarray([row["done"] for row in rows], dtype=np.float32),
            "valid_masks": np.stack([row["valid_mask"] for row in rows]),
            "next_valid_masks": np.stack([row["next_valid_mask"] for row in rows]),
        }

    def __len__(self) -> int:
        return len(self.buffer)
