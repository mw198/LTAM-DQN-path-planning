from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiments import (
    _build_scenario_groups,
    _build_train_scenario_mix,
    _select_validation_scenarios,
    run_formal_experiments,
    run_pilot_experiment,
)
from src.train import train_method
from src.utils import PROJECT_ROOT as APP_ROOT, METHOD_SPECS


def test_adjusted_ltam_method_uses_scaled_gated_temporal_delta():
    spec = METHOD_SPECS["LTAM-DQN-Adj"]
    assert spec.use_temporal_difference is True
    assert spec.use_behavior_mask is True
    assert spec.use_target_mask is True
    assert spec.temporal_delta_scale == 0.5
    assert spec.temporal_delta_gate_threshold == 0.10
    assert spec.temporal_delta_gate_min_scale == 0.25


def test_mixed_hard_curriculum_contains_dense_and_unseen_motion_cases():
    curriculum = _build_train_scenario_mix("mixed-hard")

    names = {entry["name"] for entry in curriculum}
    total_weight = sum(entry["weight"] for entry in curriculum)

    assert names == {"E0-main", "E2-dense", "E3-random", "E4-unseen-motion"}
    assert abs(total_weight - 1.0) < 1e-9
    assert any(entry["env_overrides"].get("obstacle_density") == 0.15 for entry in curriculum)
    assert any(entry["env_overrides"].get("dynamic_motion_mode") == "vertical" for entry in curriculum)


def test_main_dense_validation_profile_combines_main_and_dense_scenarios():
    groups = _build_scenario_groups()

    val_scenarios = _select_validation_scenarios(groups, "main-dense")

    assert len(val_scenarios) == 100
    assert sum(scenario["obstacle_density"] == 0.0 for scenario in val_scenarios) == 50
    assert sum(scenario["obstacle_density"] == 0.15 for scenario in val_scenarios) == 50


def test_train_method_records_curriculum_labels(tmp_path):
    output_root = tmp_path / "results"

    train_df = train_method(
        project_root=APP_ROOT,
        output_root=output_root,
        method_name="LTAM-DQN-Adj",
        train_seed=0,
        episodes=1,
        train_scenario_mix=[
            {
                "name": "dense-only",
                "weight": 1.0,
                "env_overrides": {"obstacle_density": 0.15},
            }
        ],
        val_scenarios=None,
        val_interval=10,
        device="cpu",
    )

    assert train_df.loc[0, "train_scenario"] == "dense-only"


def test_formal_runner_accepts_curriculum_and_validation_profile(tmp_path):
    result = run_formal_experiments(
        project_root=APP_ROOT,
        output_root=tmp_path / "formal",
        train_episodes=1,
        train_seeds=[0],
        methods=["LTAM-DQN-Adj"],
        train_curriculum="mixed-hard",
        validation_profile="main-dense",
        device="cpu",
    )

    train_df = result["train"]
    assert set(train_df["train_scenario"]) <= {"E0-main", "E2-dense", "E3-random", "E4-unseen-motion"}
    assert (tmp_path / "formal" / "configs" / "val_scenarios_seed42_main-dense.json").exists()
    assert (tmp_path / "formal" / "configs" / "train_curriculum_mixed-hard.json").exists()


def test_pilot_runner_does_not_require_curriculum_arguments(tmp_path):
    result = run_pilot_experiment(
        project_root=APP_ROOT,
        output_root=tmp_path / "pilot",
        episodes=1,
        seed=0,
        methods=["LTAM-DQN-Adj"],
        device="cpu",
    )

    assert set(result.keys()) == {"train", "eval"}
