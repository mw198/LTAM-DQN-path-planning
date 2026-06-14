from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


def shortest_path_length_static(
    *,
    map_size: int,
    start: list[int] | tuple[int, int],
    goal: list[int] | tuple[int, int],
    static_obstacles: Iterable[list[int] | tuple[int, int]],
) -> int | None:
    start_t = tuple(start)
    goal_t = tuple(goal)
    blocked = {tuple(item) for item in static_obstacles}
    if start_t in blocked or goal_t in blocked:
        return None

    queue: deque[tuple[tuple[int, int], int]] = deque([(start_t, 0)])
    visited = {start_t}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal_t:
            return dist
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if nr < 0 or nr >= map_size or nc < 0 or nc >= map_size:
                continue
            if nxt in blocked or nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def compute_path_efficiency(
    *,
    success: bool,
    reference_length: int | None,
    actual_path_length: int,
) -> float:
    if (not success) or reference_length is None or actual_path_length <= 0:
        return 0.0
    return float(reference_length / actual_path_length)


def compute_convergence_episode(success_series: list[float], threshold: float = 0.8, window: int = 100) -> int | None:
    if len(success_series) < window:
        return None
    rolling = deque(maxlen=window)
    for idx, value in enumerate(success_series, start=1):
        rolling.append(float(value))
        if len(rolling) == window and np.mean(rolling) >= threshold:
            return idx
    return None
