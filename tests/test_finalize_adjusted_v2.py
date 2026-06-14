from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.finalize_adjusted_v2 import ADJUSTED_V2_SHARDS, assert_all_complete, audit_adjusted_v2


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")


def test_assert_all_complete_reports_incomplete_methods():
    audit_df = pd.DataFrame(
        [
            {"method": "DQN", "complete": True},
            {"method": "DQN-T", "complete": False},
            {"method": "DQN-AM-noTarget", "complete": False},
        ]
    )

    with pytest.raises(RuntimeError) as excinfo:
        assert_all_complete(audit_df)

    message = str(excinfo.value)
    assert "DQN-T" in message
    assert "DQN-AM-noTarget" in message


def test_audit_adjusted_v2_uses_fixed_shard_mapping(tmp_path):
    base_root = tmp_path / "adjusted_v2"
    for root_name, method in ADJUSTED_V2_SHARDS:
        root = base_root / root_name
        _touch(root / "logs" / method / "train_seed_0.csv")
        _touch(root / "metrics_csv" / "E0-main" / method / "seed_0" / "eval_metrics.csv")
        _touch(root / "path_visualizations" / "E0-main" / method / "seed_0" / "scenario-0000.json")
        _touch(root / "summary_tables" / "main_results.csv")

    audit_df = audit_adjusted_v2(
        base_root,
        expected_train_logs=1,
        expected_eval_csv=1,
        min_path_json=1,
        min_summary_files=1,
    )

    assert set(audit_df["method"]) == {method for _, method in ADJUSTED_V2_SHARDS}
    assert audit_df["complete"].all()
