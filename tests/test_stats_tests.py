from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.stats_tests import write_stats


def test_write_stats_includes_adjusted_ltam_anchor_when_present(tmp_path):
    rows = []
    outcomes = {
        "DQN": [1, 0, 1, 0, 1, 0],
        "LTAM-DQN": [1, 1, 1, 0, 1, 0],
        "LTAM-DQN-Adj": [1, 1, 1, 1, 1, 0],
    }
    for method, successes in outcomes.items():
        for idx, success in enumerate(successes):
            rows.append(
                {
                    "method": method,
                    "scenario_group": "E0-main",
                    "scenario_id": f"case-{idx % 2}",
                    "train_seed": idx // 2,
                    "success": success,
                }
            )

    output_path = write_stats(pd.DataFrame(rows), tmp_path)
    stats_df = pd.read_csv(output_path)

    assert set(stats_df["anchor_method"]) == {"LTAM-DQN", "LTAM-DQN-Adj"}
    assert {
        ("LTAM-DQN", "DQN"),
        ("LTAM-DQN-Adj", "DQN"),
    }.issubset(set(zip(stats_df["anchor_method"], stats_df["compare_method"])))
