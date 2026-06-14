from __future__ import annotations

from typing import Any

import numpy as np


ACTIONS: dict[int, np.ndarray] = {
    0: np.array([-1, 0], dtype=np.int32),
    1: np.array([1, 0], dtype=np.int32),
    2: np.array([0, -1], dtype=np.int32),
    3: np.array([0, 1], dtype=np.int32),
    4: np.array([0, 0], dtype=np.int32),
}


def ego_motion_compensate(prev_obs: np.ndarray, prev_action: int | None, current_obs: np.ndarray) -> np.ndarray:
    if prev_action is None or prev_action == 4:
        return prev_obs.copy()

    aligned = current_obs.copy()
    if prev_action == 0:
        aligned[1:, :] = prev_obs[:-1, :]
    elif prev_action == 1:
        aligned[:-1, :] = prev_obs[1:, :]
    elif prev_action == 2:
        aligned[:, 1:] = prev_obs[:, :-1]
    elif prev_action == 3:
        aligned[:, :-1] = prev_obs[:, 1:]
    else:
        raise ValueError(f"unsupported previous action: {prev_action}")
    return aligned


def build_state_vector(
    *,
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    map_size: int,
    current_obs: np.ndarray,
    delta_obs: np.ndarray,
    prev_action: int | None,
    step_index: int,
    max_steps: int,
) -> np.ndarray:
    scale = max(1, map_size - 1)
    agent_norm = agent_pos.astype(np.float32) / scale
    goal_rel = (goal_pos - agent_pos).astype(np.float32) / scale
    prev_action_onehot = np.zeros(5, dtype=np.float32)
    if prev_action is not None:
        prev_action_onehot[int(prev_action)] = 1.0
    step_ratio = np.array([step_index / max(1, max_steps)], dtype=np.float32)
    return np.concatenate(
        [
            agent_norm,
            goal_rel,
            current_obs.reshape(-1).astype(np.float32),
            delta_obs.reshape(-1).astype(np.float32),
            prev_action_onehot,
            step_ratio,
        ]
    ).astype(np.float32)


class DynamicGridEnv:
    def __init__(
        self,
        *,
        map_size: int = 15,
        local_window: int = 5,
        num_static_obstacles: int = 0,
        num_dynamic_obstacles: int = 5,
        dynamic_motion_mode: str = "mixed",
        obstacle_density: float = 0.0,
        max_steps: int = 120,
        start_goal_mode: str = "random",
        seed: int | None = None,
    ) -> None:
        if local_window % 2 == 0:
            raise ValueError("local_window must be odd")
        self.map_size = map_size
        self.local_window = local_window
        self.view_radius = local_window // 2
        self.num_static_obstacles = num_static_obstacles
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.dynamic_motion_mode = dynamic_motion_mode
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.start_goal_mode = start_goal_mode
        self.step_penalty = -0.05
        self.goal_reward = 10.0
        self.collision_penalty = -10.0
        self.timeout_penalty = -1.0
        self.progress_coef = 0.5
        self.risk_penalty = -0.2
        self.rng = np.random.default_rng(seed)

        self.static_obstacles: set[tuple[int, int]] = set()
        self.dynamic_obstacles: list[dict[str, np.ndarray]] = []
        self.agent_pos = np.zeros(2, dtype=np.int32)
        self.goal_pos = np.zeros(2, dtype=np.int32)
        self.start_pos = np.zeros(2, dtype=np.int32)
        self.time_step = 0
        self.done = False
        self.done_reason = "running"
        self.prev_action: int | None = None
        self.current_local_observation = np.zeros((self.local_window, self.local_window), dtype=np.float32)
        self.prev_local_observation = np.zeros_like(self.current_local_observation)
        self.action_attempts = 0
        self.invalid_action_count = 0
        self.path_history: list[list[int]] = []
        self.action_history: list[int] = []
        self.reward_history: list[float] = []
        self.dynamic_obstacle_history: list[list[list[int]]] = []
        self.cumulative_reward = 0.0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time_step = 0
        self.done = False
        self.done_reason = "running"
        self.prev_action = None
        self.action_attempts = 0
        self.invalid_action_count = 0
        self.path_history = []
        self.action_history = []
        self.reward_history = []
        self.dynamic_obstacle_history = []
        self.cumulative_reward = 0.0
        self.static_obstacles = self._sample_static_obstacles()
        self.start_pos, self.goal_pos = self._sample_start_goal()
        self.agent_pos = self.start_pos.copy()
        self.dynamic_obstacles = self._sample_dynamic_obstacles()
        self.current_local_observation = self.get_local_observation()
        self.prev_local_observation = np.zeros_like(self.current_local_observation)
        self.path_history.append(self.agent_pos.astype(int).tolist())
        self.dynamic_obstacle_history.append(self._snapshot_dynamic_obstacles())
        return self.get_state(), self._info()

    def step(self, action: int):
        if action not in ACTIONS:
            raise ValueError(f"invalid action: {action}")
        if self.done:
            raise RuntimeError("episode already terminated")

        self.time_step += 1
        self.action_attempts += 1
        reward = self.step_penalty
        invalid_action = False
        current_obs_before = self.current_local_observation.copy()
        old_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        candidate = self.agent_pos + ACTIONS[action]

        if not self._inside(candidate) or self._occupied(candidate):
            invalid_action = True
            self.invalid_action_count += 1
            self.done = True
            self.done_reason = "collision"
            reward += self.collision_penalty
        else:
            self.agent_pos = candidate
            self.path_history.append(self.agent_pos.astype(int).tolist())
            new_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
            reward += self.progress_coef * float(old_distance - new_distance)

        if (not self.done) and np.array_equal(self.agent_pos, self.goal_pos):
            self.done = True
            self.done_reason = "success"
            reward += self.goal_reward

        if not self.done:
            self._move_dynamic_obstacles()
            if self._occupied(self.agent_pos):
                self.done = True
                self.done_reason = "collision"
                reward += self.collision_penalty

        truncated = self.time_step >= self.max_steps
        if (not self.done) and truncated:
            self.done = True
            self.done_reason = "timeout"
            reward += self.timeout_penalty

        if not self.done:
            min_obstacle_distance = self._min_obstacle_distance(self.agent_pos)
            if min_obstacle_distance <= 1:
                reward += self.risk_penalty

        self.prev_local_observation = current_obs_before
        self.current_local_observation = self.get_local_observation()
        self.prev_action = action
        self.action_history.append(action)
        self.reward_history.append(float(reward))
        self.dynamic_obstacle_history.append(self._snapshot_dynamic_obstacles())
        self.cumulative_reward += float(reward)

        terminated = self.done and self.done_reason in {"success", "collision"}
        truncated = self.done and self.done_reason == "timeout"
        return self.get_state(), reward, terminated, truncated, self._info(invalid_action=invalid_action)

    def get_state(
        self,
        use_temporal_difference: bool = True,
        temporal_delta_scale: float = 1.0,
        temporal_delta_gate_threshold: float | None = None,
        temporal_delta_gate_min_scale: float = 1.0,
    ) -> np.ndarray:
        if use_temporal_difference:
            aligned_prev = ego_motion_compensate(
                self.prev_local_observation,
                self.prev_action,
                self.current_local_observation,
            )
            delta_obs = self.current_local_observation.astype(np.float32) - aligned_prev.astype(np.float32)
            delta_obs = delta_obs * float(temporal_delta_scale)
            if temporal_delta_gate_threshold is not None:
                local_density = float(self.current_local_observation.mean())
                if local_density < float(temporal_delta_gate_threshold):
                    delta_obs = delta_obs * float(temporal_delta_gate_min_scale)
        else:
            delta_obs = np.zeros_like(self.current_local_observation, dtype=np.float32)
        return build_state_vector(
            agent_pos=self.agent_pos,
            goal_pos=self.goal_pos,
            map_size=self.map_size,
            current_obs=self.current_local_observation,
            delta_obs=delta_obs,
            prev_action=self.prev_action,
            step_index=self.time_step,
            max_steps=self.max_steps,
        )

    def get_local_observation(self) -> np.ndarray:
        obs = np.zeros((self.local_window, self.local_window), dtype=np.float32)
        for dr in range(-self.view_radius, self.view_radius + 1):
            for dc in range(-self.view_radius, self.view_radius + 1):
                rr = dr + self.view_radius
                cc = dc + self.view_radius
                cell = self.agent_pos + np.array([dr, dc], dtype=np.int32)
                if not self._inside(cell):
                    obs[rr, cc] = 1.0
                elif self._occupied(cell):
                    obs[rr, cc] = 1.0
                else:
                    obs[rr, cc] = 0.0
        return obs

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(5, dtype=np.float32)
        for action, delta in ACTIONS.items():
            if action == 4:
                mask[action] = 1.0
                continue
            nxt = self.agent_pos + delta
            if self._inside(nxt) and not self._occupied(nxt):
                mask[action] = 1.0
        return mask

    def _sample_static_obstacles(self) -> set[tuple[int, int]]:
        obstacles: set[tuple[int, int]] = set()
        if self.obstacle_density <= 0.0 and self.num_static_obstacles <= 0:
            return obstacles
        target = self.num_static_obstacles
        if target <= 0:
            target = int(self.map_size * self.map_size * self.obstacle_density)
        while len(obstacles) < target:
            cell = tuple(self.rng.integers(0, self.map_size, size=2, dtype=np.int32).tolist())
            obstacles.add(cell)
        return obstacles

    def _sample_start_goal(self) -> tuple[np.ndarray, np.ndarray]:
        if self.start_goal_mode == "fixed":
            start = np.array([0, 0], dtype=np.int32)
            goal = np.array([self.map_size - 1, self.map_size - 1], dtype=np.int32)
            return start, goal
        start = self._sample_free_cell(set())
        occupied = {tuple(start)}
        goal = self._sample_free_cell(occupied)
        return start, goal

    def _sample_dynamic_obstacles(self) -> list[dict[str, np.ndarray]]:
        occupied = {tuple(self.start_pos), tuple(self.goal_pos), *self.static_obstacles}
        obstacles: list[dict[str, np.ndarray]] = []
        for _ in range(self.num_dynamic_obstacles):
            pos = self._sample_free_cell(occupied)
            direction = self._initial_direction()
            obstacles.append({"pos": pos, "dir": direction})
            occupied.add(tuple(pos))
        return obstacles

    def _sample_free_cell(self, occupied: set[tuple[int, int]]) -> np.ndarray:
        while True:
            cell = self.rng.integers(0, self.map_size, size=2, dtype=np.int32)
            cell_key = tuple(cell.tolist())
            if cell_key not in occupied:
                return cell

    def _initial_direction(self) -> np.ndarray:
        if self.dynamic_motion_mode == "horizontal":
            return self.rng.choice(
                [np.array([0, -1], dtype=np.int32), np.array([0, 1], dtype=np.int32)]
            )
        if self.dynamic_motion_mode == "vertical":
            return self.rng.choice(
                [np.array([-1, 0], dtype=np.int32), np.array([1, 0], dtype=np.int32)]
            )
        return ACTIONS[int(self.rng.integers(0, 4))].copy()

    def _move_dynamic_obstacles(self) -> None:
        occupied_after_move = {tuple(self.goal_pos.tolist()), *self.static_obstacles}
        new_obstacles: list[dict[str, np.ndarray]] = []
        for obstacle in self.dynamic_obstacles:
            pos = obstacle["pos"]
            direction = obstacle["dir"]
            if self.dynamic_motion_mode in {"random_walk", "mixed", "unseen_motion"}:
                direction = ACTIONS[int(self.rng.integers(0, 4))].copy()
            candidate = pos + direction
            if not self._inside(candidate) or tuple(candidate.tolist()) in occupied_after_move:
                direction = self._reverse_or_resample(direction)
                candidate = pos + direction
                if not self._inside(candidate) or tuple(candidate.tolist()) in occupied_after_move:
                    candidate = pos.copy()
            new_obstacles.append({"pos": candidate.copy(), "dir": direction.copy()})
            occupied_after_move.add(tuple(candidate.tolist()))
        self.dynamic_obstacles = new_obstacles

    def _reverse_or_resample(self, direction: np.ndarray) -> np.ndarray:
        if self.dynamic_motion_mode == "horizontal":
            return np.array([0, -direction[1]], dtype=np.int32)
        if self.dynamic_motion_mode == "vertical":
            return np.array([-direction[0], 0], dtype=np.int32)
        if self.dynamic_motion_mode in {"random_walk", "mixed", "unseen_motion"}:
            return ACTIONS[int(self.rng.integers(0, 4))].copy()
        return -direction

    def _occupied(self, pos: np.ndarray) -> bool:
        pos_key = tuple(pos.tolist())
        if pos_key in self.static_obstacles:
            return True
        return any(np.array_equal(pos, obstacle["pos"]) for obstacle in self.dynamic_obstacles)

    def _snapshot_dynamic_obstacles(self) -> list[list[int]]:
        return [obstacle["pos"].astype(int).tolist() for obstacle in self.dynamic_obstacles]

    def _manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        return int(np.abs(a - b).sum())

    def _min_obstacle_distance(self, pos: np.ndarray) -> int:
        if not self.dynamic_obstacles and not self.static_obstacles:
            return self.map_size * 2
        distances = [self._manhattan_distance(pos, np.array(cell, dtype=np.int32)) for cell in self.static_obstacles]
        distances.extend(self._manhattan_distance(pos, obstacle["pos"]) for obstacle in self.dynamic_obstacles)
        return min(distances)

    def _inside(self, pos: np.ndarray) -> bool:
        return 0 <= int(pos[0]) < self.map_size and 0 <= int(pos[1]) < self.map_size

    def _info(self, invalid_action: bool = False) -> dict[str, Any]:
        return {
            "success": self.done_reason == "success",
            "collision": self.done_reason == "collision",
            "timeout": self.done_reason == "timeout",
            "done_reason": self.done_reason,
            "time_step": self.time_step,
            "steps": self.time_step,
            "agent_pos": self.agent_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "path_length": max(0, len(self.path_history) - 1),
            "invalid_action": invalid_action,
            "total_invalid_actions": self.invalid_action_count,
            "invalid_action_ratio": self.invalid_action_count / max(1, self.action_attempts),
            "cumulative_reward": self.cumulative_reward,
            "action_mask": self.get_action_mask(),
            "path_history": list(self.path_history),
        }

    def get_trajectory(self) -> dict[str, Any]:
        collision_step = self.time_step if self.done_reason == "collision" else None
        return {
            "start": self.start_pos.astype(int).tolist(),
            "goal": self.goal_pos.astype(int).tolist(),
            "agent_positions": list(self.path_history),
            "dynamic_obstacle_positions": list(self.dynamic_obstacle_history),
            "actions": list(self.action_history),
            "rewards": list(self.reward_history),
            "done_reason": self.done_reason,
            "collision_step": collision_step,
        }
