from pathlib import Path
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.path_case_figures import build_path_case_artifacts, select_path_cases


def _write_eval(root: Path, scenario_group: str, method: str, seed: int, rows: list[dict]) -> None:
    path = root / "metrics_csv" / scenario_group / method / f"seed_{seed}" / "eval_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_traj(root: Path, scenario_group: str, method: str, seed: int, scenario_id: str, done_reason: str) -> None:
    path = root / "path_visualizations" / scenario_group / method / f"seed_{seed}" / f"{scenario_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "start": [0, 0],
        "goal": [4, 4],
        "agent_positions": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 4]],
        "dynamic_obstacle_positions": [[[0, 4]], [[1, 4]], [[2, 4]], [[3, 4]], [[4, 4]]],
        "actions": [1, 1, 1, 3],
        "rewards": [0.0, 0.0, 0.0, 10.0],
        "done_reason": done_reason,
        "collision_step": None,
        "scenario_id": scenario_id,
        "method": method,
        "seed": seed,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_select_path_cases_prefers_anchor_success_over_baseline_failure(tmp_path):
    root = tmp_path / "results"
    _write_eval(
        root,
        "E2-dense",
        "DQN",
        0,
        [
            {"scenario_group": "E2-dense", "scenario_id": "scenario-0001", "method": "DQN", "train_seed": 0, "success": 0, "collision": 1, "timeout": 0, "path_efficiency": 0.0, "path_length": 10},
            {"scenario_group": "E2-dense", "scenario_id": "scenario-0002", "method": "DQN", "train_seed": 0, "success": 1, "collision": 0, "timeout": 0, "path_efficiency": 0.5, "path_length": 12},
        ],
    )
    _write_eval(
        root,
        "E2-dense",
        "LTAM-DQN",
        0,
        [
            {"scenario_group": "E2-dense", "scenario_id": "scenario-0001", "method": "LTAM-DQN", "train_seed": 0, "success": 1, "collision": 0, "timeout": 0, "path_efficiency": 0.8, "path_length": 8},
            {"scenario_group": "E2-dense", "scenario_id": "scenario-0002", "method": "LTAM-DQN", "train_seed": 0, "success": 1, "collision": 0, "timeout": 0, "path_efficiency": 0.6, "path_length": 10},
        ],
    )

    selected = select_path_cases(root, scenario_group="E2-dense", anchor_method="LTAM-DQN", baseline_method="DQN", top_k=1)

    assert selected.iloc[0]["scenario_id"] == "scenario-0001"
    assert selected.iloc[0]["anchor_success"] == 1
    assert selected.iloc[0]["baseline_success"] == 0


def test_build_path_case_artifacts_writes_index_and_png(tmp_path):
    root = tmp_path / "results"
    scenario = {
        "scenario_id": "scenario-0001",
        "map_size": 5,
        "seed": 123,
        "num_static_obstacles": 0,
        "num_dynamic_obstacles": 1,
    }
    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "E2-dense.json").write_text(json.dumps([scenario]), encoding="utf-8")
    _write_eval(
        root,
        "E2-dense",
        "DQN",
        0,
        [{"scenario_group": "E2-dense", "scenario_id": "scenario-0001", "method": "DQN", "train_seed": 0, "success": 0, "collision": 1, "timeout": 0, "path_efficiency": 0.0, "path_length": 10}],
    )
    _write_eval(
        root,
        "E2-dense",
        "LTAM-DQN",
        0,
        [{"scenario_group": "E2-dense", "scenario_id": "scenario-0001", "method": "LTAM-DQN", "train_seed": 0, "success": 1, "collision": 0, "timeout": 0, "path_efficiency": 0.8, "path_length": 8}],
    )
    _write_traj(root, "E2-dense", "DQN", 0, "scenario-0001", "collision")
    _write_traj(root, "E2-dense", "LTAM-DQN", 0, "scenario-0001", "success")

    output_dir = build_path_case_artifacts(
        root,
        scenario_group="E2-dense",
        anchor_method="LTAM-DQN",
        baseline_method="DQN",
        top_k=1,
    )

    index_path = output_dir / "path_case_index.csv"
    assert index_path.exists()
    assert (output_dir / "E2-dense_seed0_scenario-0001_LTAM-DQN_vs_DQN.png").exists()
    assert pd.read_csv(index_path).iloc[0]["scenario_id"] == "scenario-0001"
