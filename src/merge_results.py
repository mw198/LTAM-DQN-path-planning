from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.analysis.plot_results import plot_training_metric
from src.analysis.stats_tests import write_stats
from src.analysis.summarize_results import summarize_main_results, summarize_train_logs, write_summary_csv
from src.utils import METHOD_SPECS, ensure_dir


def list_eval_metric_files(result_roots: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for root in result_roots:
        root_path = Path(root)
        files.extend(sorted((root_path / "metrics_csv").glob("*/*/seed_*/eval_metrics.csv")))
    return files


def list_train_log_files(result_roots: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for root in result_roots:
        root_path = Path(root)
        files.extend(sorted((root_path / "logs").glob("*/train_seed_*.csv")))
    return files


def _copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        relative = path.relative_to(src)
        target = dst / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if target.exists() and path.samefile(target):
                    continue
            except OSError:
                pass
            shutil.copy2(path, target)


def merge_result_roots(*, source_roots: list[str | Path], output_root: str | Path) -> Path:
    sources = [Path(root) for root in source_roots]
    output_root = Path(output_root)
    ensure_dir(output_root)

    for subdir in ["configs", "logs", "checkpoints", "metrics_csv", "path_visualizations"]:
        ensure_dir(output_root / subdir)
        for source in sources:
            _copy_tree_contents(source / subdir, output_root / subdir)

    train_files = list_train_log_files(sources)
    eval_files = list_eval_metric_files(sources)
    train_df = pd.concat([pd.read_csv(path) for path in train_files], ignore_index=True) if train_files else pd.DataFrame()
    eval_df = pd.concat([pd.read_csv(path) for path in eval_files], ignore_index=True) if eval_files else pd.DataFrame()

    ensure_dir(output_root / "summary_tables")
    ensure_dir(output_root / "figures_data")
    ensure_dir(output_root / "stats")

    if not train_df.empty:
        write_summary_csv(summarize_train_logs(train_df), output_root / "summary_tables" / "train_summary.csv")
        plot_training_metric(
            output_root=output_root,
            methods=list(METHOD_SPECS.keys()),
            metric="reward_ma_100",
            output_filename="training_reward_curve.png",
            title="Training Reward Curves",
        )
        plot_training_metric(
            output_root=output_root,
            methods=list(METHOD_SPECS.keys()),
            metric="success_rate_100",
            output_filename="training_success_curve.png",
            title="Training Success Curves",
        )
        plot_training_metric(
            output_root=output_root,
            methods=list(METHOD_SPECS.keys()),
            metric="collision_rate_100",
            output_filename="training_collision_curve.png",
            title="Training Collision Curves",
        )
        plot_training_metric(
            output_root=output_root,
            methods=list(METHOD_SPECS.keys()),
            metric="timeout_rate_100",
            output_filename="training_timeout_curve.png",
            title="Training Timeout Curves",
        )
        plot_training_metric(
            output_root=output_root,
            methods=list(METHOD_SPECS.keys()),
            metric="invalid_action_ratio_100",
            output_filename="invalid_action_curve.png",
            title="Invalid Action Curves",
        )
        plot_training_metric(
            output_root=output_root,
            methods=list(METHOD_SPECS.keys()),
            metric="mask_intervention_rate_100",
            output_filename="mask_intervention_curve.png",
            title="Mask Intervention Curves",
        )

    if not eval_df.empty:
        main_eval_df = eval_df[eval_df["scenario_group"] == "E0-main"].copy()
        generalization_df = eval_df[eval_df["scenario_group"] != "E0-main"].copy()
        complexity_df = (
            main_eval_df.groupby("method", as_index=False)
            .agg(
                avg_inference_time_ms=("avg_inference_time_ms", "mean"),
                num_parameters=("num_parameters", "mean"),
            )
        )
        write_summary_csv(summarize_main_results(main_eval_df), output_root / "summary_tables" / "main_results.csv")
        write_summary_csv(summarize_main_results(main_eval_df), output_root / "summary_tables" / "ablation_results.csv")
        write_summary_csv(
            generalization_df.groupby(["scenario_group", "method"], as_index=False)
            .agg(
                success_rate=("success", "mean"),
                collision_rate=("collision", "mean"),
                timeout_rate=("timeout", "mean"),
                path_efficiency=("path_efficiency", "mean"),
                avg_reward=("episode_reward", "mean"),
            ),
            output_root / "summary_tables" / "generalization_results.csv",
        )
        write_summary_csv(complexity_df, output_root / "summary_tables" / "complexity_results.csv")
        write_stats(main_eval_df, output_root)
        stats_csv = output_root / "stats" / "stat_tests_main.csv"
        if stats_csv.exists():
            write_summary_csv(pd.read_csv(stats_csv), output_root / "summary_tables" / "stat_tests.csv")

    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-roots", type=str, required=True, help="Comma-separated list of result roots")
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()
    source_roots = [item.strip() for item in args.source_roots.split(",") if item.strip()]
    merge_result_roots(source_roots=source_roots, output_root=args.output_root)


if __name__ == "__main__":
    main()
