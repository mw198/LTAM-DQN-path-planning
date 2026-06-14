from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.dynamic_grid_env import DynamicGridEnv


def test_wait_action_is_always_valid_before_done():
    env = DynamicGridEnv(map_size=7, local_window=5, num_dynamic_obstacles=0, seed=1)
    _state, _info = env.reset(seed=1)
    mask = env.get_action_mask()
    assert mask.shape == (5,)
    assert mask[4] == 1.0


def test_out_of_bounds_action_is_masked():
    env = DynamicGridEnv(map_size=5, local_window=5, num_dynamic_obstacles=0, seed=2)
    env.reset(seed=2)
    env.agent_pos = np.array([0, 0], dtype=np.int32)
    mask = env.get_action_mask()
    assert mask[0] == 0.0
    assert mask[2] == 0.0


def test_dynamic_obstacle_can_collide_after_agent_move():
    env = DynamicGridEnv(map_size=5, local_window=5, num_dynamic_obstacles=1, dynamic_motion_mode="horizontal", seed=3)
    env.reset(seed=3)
    env.agent_pos = np.array([2, 2], dtype=np.int32)
    env.goal_pos = np.array([4, 4], dtype=np.int32)
    env.dynamic_obstacles = [
        {
            "pos": np.array([2, 3], dtype=np.int32),
            "dir": np.array([0, -1], dtype=np.int32),
        }
    ]
    _state, _reward, terminated, _truncated, info = env.step(4)
    assert terminated is True
    assert info["done_reason"] == "collision"
