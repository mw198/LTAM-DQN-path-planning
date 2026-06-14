from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.organize_paper_tables import (
    build_ablation_table,
    build_main_table,
    build_stats_table,
    build_summary_markdown,
    organize_tables,
)


def _main_row(method: str, success_rate: float) -> dict:
    return {
        "method": method,
        "success_rate": success_rate,
        "collision_rate": 0.1,
        "timeout_rate": 0.0,
        "avg_path_length": 12.0,
        "path_efficiency": 0.8,
        "avg_reward": 1.0,
        "invalid_action_execution_rate": 0.0,
        "mask_intervention_rate": 0.1,
        "avg_inference_time_ms": 0.8,
        "num_parameters": 24965,
    }


def test_organized_tables_keep_adjusted_ltam_method():
    main_df = pd.DataFrame(
        [
            _main_row("LTAM-DQN-Adj", 0.97),
            _main_row("DQN", 0.95),
            _main_row("LTAM-DQN", 0.96),
        ]
    )

    main_table = build_main_table(main_df)
    ablation_table = build_ablation_table(main_df)

    assert "LTAM-DQN-Adj" in set(main_table.iloc[:, 0])
    adjusted_row = ablation_table[ablation_table.iloc[:, 0] == "LTAM-DQN-Adj"].iloc[0]
    assert adjusted_row.iloc[1] == "check"
    assert adjusted_row.iloc[2] == "check"
    assert adjusted_row.iloc[3] == "check"


def test_stats_table_uses_each_row_anchor_method_for_success_gap():
    main_df = pd.DataFrame(
        [
            _main_row("DQN", 0.95),
            _main_row("LTAM-DQN", 0.96),
            _main_row("LTAM-DQN-Adj", 0.98),
        ]
    )
    stats_df = pd.DataFrame(
        [
            {
                "anchor_method": "LTAM-DQN-Adj",
                "compare_method": "DQN",
                "paired_t_success_p": 0.1234567,
                "wilcoxon_success_p": 0.2345678,
            }
        ]
    )

    stats_table = build_stats_table(stats_df, main_df)
    row = stats_table.iloc[0]

    assert row.iloc[0] == "LTAM-DQN-Adj"
    assert row.iloc[1] == "DQN"
    assert row.iloc[2] == 3.0


def test_summary_markdown_reports_adjusted_ltam_stats_anchor():
    main_df = pd.DataFrame(
        [
            _main_row("DQN", 0.99),
            _main_row("LTAM-DQN", 0.96),
            _main_row("LTAM-DQN-Adj", 0.98),
        ]
    )
    general_df = pd.DataFrame(
        [
            {
                "scenario_group": "E2-dense",
                "method": "DQN",
                "success_rate": 0.50,
            },
            {
                "scenario_group": "E6-static",
                "method": "DQN",
                "success_rate": 0.90,
            },
        ]
    )
    complexity_df = pd.DataFrame(
        [
            {
                "method": "DQN",
                "avg_inference_time_ms": 0.8,
                "num_parameters": 24965,
            }
        ]
    )
    stats_df = pd.DataFrame(
        [
            {
                "anchor_method": "LTAM-DQN-Adj",
                "compare_method": "DQN",
                "paired_t_success_p": 0.123456,
                "wilcoxon_success_p": 0.234567,
            }
        ]
    )

    summary = build_summary_markdown(main_df, general_df, complexity_df, stats_df)

    assert "`LTAM-DQN-Adj`" in summary
    assert "3.00" in summary


def test_summary_markdown_uses_actual_main_scenario_best_methods():
    main_df = pd.DataFrame(
        [
            {**_main_row("DQN", 0.95), "path_efficiency": 0.70},
            {**_main_row("LTAM-DQN", 0.96), "path_efficiency": 0.81},
            {**_main_row("LTAM-DQN-Adj", 0.98), "path_efficiency": 0.82},
        ]
    )
    general_df = pd.DataFrame(
        [
            {
                "scenario_group": "E2-dense",
                "method": "DQN",
                "success_rate": 0.50,
            },
            {
                "scenario_group": "E6-static",
                "method": "DQN",
                "success_rate": 0.90,
            },
        ]
    )
    complexity_df = pd.DataFrame(
        [
            {
                "method": "DQN",
                "avg_inference_time_ms": 0.8,
                "num_parameters": 24965,
            }
        ]
    )
    stats_df = pd.DataFrame(
        [
            {
                "anchor_method": "LTAM-DQN-Adj",
                "compare_method": "DQN",
                "paired_t_success_p": 0.123456,
                "wilcoxon_success_p": 0.234567,
            }
        ]
    )

    summary = build_summary_markdown(main_df, general_df, complexity_df, stats_df)

    assert "未超过基础 `DQN`" not in summary
    assert "路径效率最高的方法是 `LTAM-DQN-Adj`" in summary
    assert "82.00%" in summary


def test_organize_tables_writes_conclusion_boundaries(tmp_path):
    root = tmp_path / "merged"
    summary_dir = root / "summary_tables"
    summary_dir.mkdir(parents=True)
    main_df = pd.DataFrame([_main_row("DQN", 0.90), _main_row("LTAM-DQN", 0.95)])
    main_df.to_csv(summary_dir / "main_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_group": "E2-dense",
                "method": "DQN",
                "success_rate": 0.50,
                "collision_rate": 0.1,
                "timeout_rate": 0.1,
                "path_efficiency": 0.4,
                "avg_reward": 0.0,
            },
            {
                "scenario_group": "E2-dense",
                "method": "LTAM-DQN",
                "success_rate": 0.70,
                "collision_rate": 0.1,
                "timeout_rate": 0.1,
                "path_efficiency": 0.5,
                "avg_reward": 1.0,
            },
            {
                "scenario_group": "E6-static",
                "method": "DQN",
                "success_rate": 0.90,
                "collision_rate": 0.0,
                "timeout_rate": 0.1,
                "path_efficiency": 0.8,
                "avg_reward": 2.0,
            },
        ]
    ).to_csv(summary_dir / "generalization_results.csv", index=False)
    pd.DataFrame([_main_row("DQN", 0.90), _main_row("LTAM-DQN", 0.95)]).to_csv(
        summary_dir / "complexity_results.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "anchor_method": "LTAM-DQN",
                "compare_method": "DQN",
                "paired_t_success_p": 0.20,
                "wilcoxon_success_p": 0.30,
            }
        ]
    ).to_csv(summary_dir / "stat_tests.csv", index=False)

    organized_dir = organize_tables(root)

    output_path = organized_dir / "paper_conclusion_boundaries.md"
    assert output_path.exists()
    assert "Do not write: statistically significant" in output_path.read_text(encoding="utf-8")
