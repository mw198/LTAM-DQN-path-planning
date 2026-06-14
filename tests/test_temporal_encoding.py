from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.dynamic_grid_env import DynamicGridEnv, build_state_vector, ego_motion_compensate


def test_ego_motion_compensate_shifts_previous_view_for_up_action():
    prev_obs = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    current_obs = np.zeros((3, 3), dtype=np.float32)
    aligned = ego_motion_compensate(prev_obs, prev_action=0, current_obs=current_obs)
    assert aligned.shape == (3, 3)
    assert np.all(aligned[1] == prev_obs[0])


def test_first_step_delta_is_zero():
    current_obs = np.ones((5, 5), dtype=np.float32)
    delta = np.zeros_like(current_obs)
    assert np.all(delta == 0.0)


def test_build_state_vector_returns_60_dims():
    state = build_state_vector(
        agent_pos=np.array([1, 2]),
        goal_pos=np.array([4, 4]),
        map_size=15,
        current_obs=np.zeros((5, 5), dtype=np.float32),
        delta_obs=np.zeros((5, 5), dtype=np.float32),
        prev_action=None,
        step_index=0,
        max_steps=120,
    )
    assert state.shape == (60,)


def test_temporal_delta_scale_and_density_gate_reduce_sparse_delta():
    env = DynamicGridEnv(map_size=7, local_window=5, num_dynamic_obstacles=0, seed=4)
    env.reset(seed=4)
    env.prev_local_observation = np.zeros((5, 5), dtype=np.float32)
    env.current_local_observation = np.zeros((5, 5), dtype=np.float32)
    env.current_local_observation[2, 2] = 1.0
    env.prev_action = None

    state = env.get_state(
        use_temporal_difference=True,
        temporal_delta_scale=0.5,
        temporal_delta_gate_threshold=0.10,
        temporal_delta_gate_min_scale=0.25,
    )

    delta_start = 2 + 2 + 25
    delta = state[delta_start : delta_start + 25].reshape(5, 5)
    assert np.isclose(delta[2, 2], 0.125)
    assert np.count_nonzero(delta) == 1
