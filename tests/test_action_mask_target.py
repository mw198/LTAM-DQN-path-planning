from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.replay_buffer import ReplayBuffer
from src.agents.dqn_agent import compute_next_target


def test_replay_buffer_stores_valid_and_next_valid_masks():
    buffer = ReplayBuffer(capacity=8)
    buffer.add(
        state=np.zeros(60, dtype=np.float32),
        action=1,
        reward=0.5,
        next_state=np.ones(60, dtype=np.float32),
        done=False,
        valid_mask=np.array([1, 0, 1, 1, 1], dtype=np.float32),
        next_valid_mask=np.array([1, 1, 1, 0, 1], dtype=np.float32),
        info={"done_reason": "running"},
    )
    batch = buffer.sample(1)
    assert batch["valid_masks"].shape == (1, 5)
    assert batch["next_valid_masks"].shape == (1, 5)


def test_target_mask_excludes_invalid_next_actions():
    next_q = torch.tensor([[1.0, 9.0, 2.0, 3.0, 4.0]])
    next_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0]])
    target = compute_next_target(
        rewards=torch.tensor([[0.5]]),
        dones=torch.tensor([[0.0]]),
        next_q_values=next_q,
        next_valid_masks=next_mask,
        gamma=0.99,
        use_target_mask=True,
    )
    assert np.isclose(float(target.item()), 0.5 + 0.99 * 4.0)
