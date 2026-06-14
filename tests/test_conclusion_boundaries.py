from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.conclusion_boundaries import build_conclusion_boundaries, write_conclusion_boundaries


def test_conclusion_boundaries_warn_when_main_gap_is_not_significant(tmp_path):
    main_df = pd.DataFrame(
        [
            {"method": "DQN", "success_rate": 0.9324},
            {"method": "LTAM-DQN", "success_rate": 0.9428},
            {"method": "LTAM-DQN-Adj", "success_rate": 0.9380},
        ]
    )
    general_df = pd.DataFrame(
        [
            {"scenario_group": "E2-dense", "method": "DQN", "success_rate": 0.488},
            {"scenario_group": "E2-dense", "method": "LTAM-DQN", "success_rate": 0.618},
            {"scenario_group": "E2-dense", "method": "LTAM-DQN-Adj", "success_rate": 0.612},
        ]
    )
    stats_df = pd.DataFrame(
        [
            {
                "anchor_method": "LTAM-DQN",
                "compare_method": "DQN",
                "paired_t_success_p": 0.353089,
                "wilcoxon_success_p": 0.100097,
            },
            {
                "anchor_method": "LTAM-DQN-Adj",
                "compare_method": "DQN",
                "paired_t_success_p": 0.631346,
                "wilcoxon_success_p": 0.366157,
            },
        ]
    )

    text = build_conclusion_boundaries(main_df, general_df, stats_df)

    assert "LTAM-DQN vs DQN" in text
    assert "+1.04 pp" in text
    assert "+13.00 pp" in text
    assert "not statistically significant" in text
    assert "Do not write: statistically significant" in text


def test_write_conclusion_boundaries_reads_summary_tables(tmp_path):
    root = tmp_path / "merged"
    summary_dir = root / "summary_tables"
    summary_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"method": "DQN", "success_rate": 0.90},
            {"method": "LTAM-DQN", "success_rate": 0.95},
        ]
    ).to_csv(summary_dir / "main_results.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_group": "E2-dense", "method": "DQN", "success_rate": 0.50},
            {"scenario_group": "E2-dense", "method": "LTAM-DQN", "success_rate": 0.70},
        ]
    ).to_csv(summary_dir / "generalization_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "anchor_method": "LTAM-DQN",
                "compare_method": "DQN",
                "paired_t_success_p": 0.01,
                "wilcoxon_success_p": 0.02,
            }
        ]
    ).to_csv(summary_dir / "stat_tests.csv", index=False)

    output_path = write_conclusion_boundaries(root)

    assert output_path == root / "summary_tables" / "organized" / "paper_conclusion_boundaries.md"
    assert output_path.exists()
    assert "statistically significant" in output_path.read_text(encoding="utf-8")
