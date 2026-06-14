from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.artifact_completeness import summarize_method_artifacts


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")


def test_summarize_method_artifacts_marks_complete_when_counts_meet_thresholds(tmp_path):
    root = tmp_path / "results"
    for seed in range(2):
        _touch(root / "logs" / "DQN" / f"train_seed_{seed}.csv")
    for idx in range(3):
        _touch(root / "metrics_csv" / f"E{idx}" / "DQN" / "seed_0" / "eval_metrics.csv")
    for idx in range(2):
        _touch(root / "path_visualizations" / "E0-main" / "DQN" / "seed_0" / f"scenario-{idx:04d}.json")
    _touch(root / "summary_tables" / "main_results.csv")

    status = summarize_method_artifacts(
        root,
        "DQN",
        expected_train_logs=2,
        expected_eval_csv=3,
        min_path_json=2,
        min_summary_files=1,
    )

    assert status["complete"] is True
    assert status["train_logs"] == 2
    assert status["eval_csv"] == 3
    assert status["path_json"] == 2
    assert status["summary_files"] == 1


def test_summarize_method_artifacts_reports_missing_requirements(tmp_path):
    root = tmp_path / "results"
    _touch(root / "logs" / "LTAM-DQN" / "train_seed_0.csv")

    status = summarize_method_artifacts(
        root,
        "LTAM-DQN",
        expected_train_logs=2,
        expected_eval_csv=1,
        min_path_json=1,
        min_summary_files=1,
    )

    assert status["complete"] is False
    assert status["missing_train_logs"] == 1
    assert status["missing_eval_csv"] == 1
    assert status["missing_path_json"] == 1
    assert status["missing_summary_files"] == 1
