from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.scenario_generator import generate_fixed_scenarios


def test_generate_fixed_scenarios_is_deterministic():
    scenarios_a = generate_fixed_scenarios(
        count=3,
        seed=42,
        map_size=15,
        num_dynamic_obstacles=5,
        dynamic_motion_mode="mixed",
    )
    scenarios_b = generate_fixed_scenarios(
        count=3,
        seed=42,
        map_size=15,
        num_dynamic_obstacles=5,
        dynamic_motion_mode="mixed",
    )
    assert scenarios_a == scenarios_b
    assert len(scenarios_a) == 3
    assert scenarios_a[0]["scenario_id"].startswith("scenario-")
