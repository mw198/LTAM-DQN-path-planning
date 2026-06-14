from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    map_size: int
    num_static_obstacles: int
    num_dynamic_obstacles: int
    dynamic_motion_mode: str
    obstacle_density: float
    max_steps: int
    start_goal_mode: str
    seed: int


def generate_fixed_scenarios(
    *,
    count: int,
    seed: int,
    map_size: int,
    num_dynamic_obstacles: int,
    dynamic_motion_mode: str,
    num_static_obstacles: int = 0,
    obstacle_density: float = 0.0,
    max_steps: int = 120,
    start_goal_mode: str = "random",
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    scenarios: list[dict[str, Any]] = []
    for idx in range(count):
        scenario_seed = int(rng.integers(0, 2**31 - 1))
        spec = ScenarioSpec(
            scenario_id=f"scenario-{idx:04d}",
            map_size=map_size,
            num_static_obstacles=num_static_obstacles,
            num_dynamic_obstacles=num_dynamic_obstacles,
            dynamic_motion_mode=dynamic_motion_mode,
            obstacle_density=obstacle_density,
            max_steps=max_steps,
            start_goal_mode=start_goal_mode,
            seed=scenario_seed,
        )
        scenarios.append(asdict(spec))
    return scenarios


def save_scenarios(path: str | Path, scenarios: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
