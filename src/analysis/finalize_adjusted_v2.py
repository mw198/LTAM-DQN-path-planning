from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.artifact_completeness import summarize_method_artifacts
from src.analysis.organize_paper_tables import organize_tables
from src.analysis.path_case_figures import build_path_case_artifacts
from src.merge_results import merge_result_roots
from src.utils import ensure_dir


ADJUSTED_V2_SHARDS = [
    ("gpu0_DQN", "DQN"),
    ("gpu0_DQN_T", "DQN-T"),
    ("gpu5_DQN_AM_noTarget", "DQN-AM-noTarget"),
    ("gpu1_DQN_AM", "DQN-AM"),
    ("gpu5_LTAM_DQN", "LTAM-DQN"),
    ("gpu7_LTAM_DQN_Adj", "LTAM-DQN-Adj"),
]


def audit_adjusted_v2(
    base_root: str | Path,
    *,
    expected_train_logs: int = 5,
    expected_eval_csv: int = 35,
    min_path_json: int = 1,
    min_summary_files: int = 1,
) -> pd.DataFrame:
    base_root = Path(base_root)
    rows = []
    for root_name, method in ADJUSTED_V2_SHARDS:
        root = base_root / root_name
        row = summarize_method_artifacts(
            root,
            method,
            expected_train_logs=expected_train_logs,
            expected_eval_csv=expected_eval_csv,
            min_path_json=min_path_json,
            min_summary_files=min_summary_files,
        )
        row["result_root"] = str(root)
        rows.append(row)
    return pd.DataFrame(rows)


def assert_all_complete(audit_df: pd.DataFrame) -> None:
    incomplete = audit_df[~audit_df["complete"].astype(bool)]
    if incomplete.empty:
        return
    methods = ", ".join(incomplete["method"].astype(str).tolist())
    raise RuntimeError(f"artifact gate failed for incomplete methods: {methods}")


def finalize_adjusted_v2(
    *,
    base_root: str | Path,
    output_root: str | Path,
) -> Path:
    base_root = Path(base_root)
    output_root = Path(output_root)
    audit_df = audit_adjusted_v2(base_root)
    audit_dir = ensure_dir(output_root / "summary_tables")
    audit_df.to_csv(audit_dir / "artifact_completeness.csv", index=False)
    assert_all_complete(audit_df)

    source_roots = [base_root / root_name for root_name, _ in ADJUSTED_V2_SHARDS]
    merge_result_roots(source_roots=source_roots, output_root=output_root)
    organize_tables(output_root)
    build_path_case_artifacts(
        output_root,
        scenario_group="E2-dense",
        anchor_method="LTAM-DQN",
        baseline_method="DQN",
        top_k=3,
    )
    build_path_case_artifacts(
        output_root,
        scenario_group="E2-dense",
        anchor_method="LTAM-DQN-Adj",
        baseline_method="DQN",
        top_k=3,
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(finalize_adjusted_v2(base_root=args.base_root, output_root=args.output_root))


if __name__ == "__main__":
    main()
