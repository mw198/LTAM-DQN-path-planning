from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir


def _success_rate(df: pd.DataFrame, method: str, scenario_group: str | None = None) -> float | None:
    data = df[df["method"] == method]
    if scenario_group is not None and "scenario_group" in data.columns:
        data = data[data["scenario_group"] == scenario_group]
    if data.empty:
        return None
    return float(data.iloc[0]["success_rate"])


def _stats_row(stats_df: pd.DataFrame, anchor_method: str, baseline_method: str) -> pd.Series | None:
    rows = stats_df[(stats_df["anchor_method"] == anchor_method) & (stats_df["compare_method"] == baseline_method)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _format_pp(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:+.2f} pp"


def _comparison_line(
    *,
    main_df: pd.DataFrame,
    general_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    anchor_method: str,
    baseline_method: str,
) -> str:
    anchor_main = _success_rate(main_df, anchor_method)
    baseline_main = _success_rate(main_df, baseline_method)
    anchor_dense = _success_rate(general_df, anchor_method, "E2-dense")
    baseline_dense = _success_rate(general_df, baseline_method, "E2-dense")
    main_gap = None if anchor_main is None or baseline_main is None else (anchor_main - baseline_main) * 100.0
    dense_gap = None if anchor_dense is None or baseline_dense is None else (anchor_dense - baseline_dense) * 100.0
    row = _stats_row(stats_df, anchor_method, baseline_method)
    if row is None:
        significance = "main-scenario p values are missing; use descriptive wording only"
        p_text = "paired t p=missing, Wilcoxon p=missing"
        caution = "Do not write: statistically significant"
    else:
        paired_p = float(row["paired_t_success_p"])
        wilcoxon_p = float(row["wilcoxon_success_p"])
        is_significant = paired_p < 0.05 and wilcoxon_p < 0.05
        p_text = f"paired t p={paired_p:.6f}, Wilcoxon p={wilcoxon_p:.6f}"
        if is_significant:
            significance = "main-scenario advantage is statistically significant under both tests"
            caution = "Write significance only for this tested main-scenario comparison"
        else:
            significance = "main-scenario advantage is not statistically significant under the recorded tests"
            caution = "Do not write: statistically significant"
    return (
        f"- {anchor_method} vs {baseline_method}: E0-main success gap {_format_pp(main_gap)}; "
        f"E2-dense success gap {_format_pp(dense_gap)}; {p_text}; {significance}. {caution}."
    )


def build_conclusion_boundaries(
    main_df: pd.DataFrame,
    general_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    anchors: list[str] | None = None,
    baseline_method: str = "DQN",
) -> str:
    anchors = anchors or [method for method in ["LTAM-DQN", "LTAM-DQN-Adj"] if method in set(main_df["method"])]
    lines = [
        "# Conclusion Boundaries",
        "",
        "Use these statements to keep paper claims aligned with real experiment outputs.",
        "",
        "## Supported Comparisons",
    ]
    for anchor_method in anchors:
        if anchor_method == baseline_method:
            continue
        lines.append(
            _comparison_line(
                main_df=main_df,
                general_df=general_df,
                stats_df=stats_df,
                anchor_method=anchor_method,
                baseline_method=baseline_method,
            )
        )
    lines.extend(
        [
            "",
            "## Writing Guardrails",
            "- Do not claim universal optimality unless every reported scenario supports it.",
            "- Do not claim statistical significance unless the recorded p values support it.",
            "- Prefer wording such as observed improvement, higher success rate in this setting, or stronger robustness in dense dynamic scenarios.",
            "- Report adjusted variants explicitly as LTAM-DQN-Adj, not as the original LTAM-DQN.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_conclusion_boundaries(result_root: str | Path) -> Path:
    result_root = Path(result_root)
    summary_dir = result_root / "summary_tables"
    main_df = pd.read_csv(summary_dir / "main_results.csv")
    general_df = pd.read_csv(summary_dir / "generalization_results.csv")
    stats_df = pd.read_csv(summary_dir / "stat_tests.csv")
    organized_dir = ensure_dir(summary_dir / "organized")
    output_path = organized_dir / "paper_conclusion_boundaries.md"
    output_path.write_text(build_conclusion_boundaries(main_df, general_df, stats_df), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    args = parser.parse_args()
    print(write_conclusion_boundaries(args.input_root))


if __name__ == "__main__":
    main()
