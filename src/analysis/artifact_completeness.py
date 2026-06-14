from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir


def summarize_method_artifacts(
    result_root: str | Path,
    method: str,
    *,
    expected_train_logs: int = 5,
    expected_eval_csv: int = 35,
    min_path_json: int = 1,
    min_summary_files: int = 1,
) -> dict:
    result_root = Path(result_root)
    train_logs = len(list((result_root / "logs" / method).glob("train_seed_*.csv")))
    eval_csv = len(list((result_root / "metrics_csv").glob(f"*/{method}/seed_*/eval_metrics.csv")))
    path_json = len(list((result_root / "path_visualizations").glob(f"*/{method}/seed_*/*.json")))
    summary_files = len(list((result_root / "summary_tables").glob("*.csv")))
    row = {
        "method": method,
        "train_logs": train_logs,
        "eval_csv": eval_csv,
        "path_json": path_json,
        "summary_files": summary_files,
        "expected_train_logs": expected_train_logs,
        "expected_eval_csv": expected_eval_csv,
        "min_path_json": min_path_json,
        "min_summary_files": min_summary_files,
        "missing_train_logs": max(0, expected_train_logs - train_logs),
        "missing_eval_csv": max(0, expected_eval_csv - eval_csv),
        "missing_path_json": max(0, min_path_json - path_json),
        "missing_summary_files": max(0, min_summary_files - summary_files),
    }
    row["complete"] = (
        row["missing_train_logs"] == 0
        and row["missing_eval_csv"] == 0
        and row["missing_path_json"] == 0
        and row["missing_summary_files"] == 0
    )
    return row


def summarize_roots(
    roots: list[str | Path],
    methods: list[str],
    *,
    expected_train_logs: int = 5,
    expected_eval_csv: int = 35,
    min_path_json: int = 1,
    min_summary_files: int = 1,
) -> pd.DataFrame:
    rows = []
    for root in roots:
        root_path = Path(root)
        for method in methods:
            row = summarize_method_artifacts(
                root_path,
                method,
                expected_train_logs=expected_train_logs,
                expected_eval_csv=expected_eval_csv,
                min_path_json=min_path_json,
                min_summary_files=min_summary_files,
            )
            row["result_root"] = str(root_path)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", required=True, help="Comma-separated result roots")
    parser.add_argument("--methods", required=True, help="Comma-separated method names aligned with each root or shared")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--expected-train-logs", type=int, default=5)
    parser.add_argument("--expected-eval-csv", type=int, default=35)
    parser.add_argument("--min-path-json", type=int, default=1)
    parser.add_argument("--min-summary-files", type=int, default=1)
    args = parser.parse_args()

    roots = [item.strip() for item in args.roots.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if len(methods) == 1 and len(roots) > 1:
        methods = methods * len(roots)
    if len(roots) != len(methods):
        raise ValueError("--roots and --methods must have the same length unless one method is shared")

    frames = []
    for root, method in zip(roots, methods):
        frames.append(
            summarize_roots(
                [root],
                [method],
                expected_train_logs=args.expected_train_logs,
                expected_eval_csv=args.expected_eval_csv,
                min_path_json=args.min_path_json,
                min_summary_files=args.min_summary_files,
            )
        )
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(out.to_csv(index=False), end="")
    if args.output_root:
        output_dir = ensure_dir(Path(args.output_root) / "summary_tables")
        out.to_csv(output_dir / "artifact_completeness.csv", index=False)


if __name__ == "__main__":
    main()
