from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from src.utils import ensure_dir


DEFAULT_STAT_ANCHORS = ("LTAM-DQN", "LTAM-DQN-Adj")
STATS_COLUMNS = [
    "anchor_method",
    "compare_method",
    "paired_t_success_stat",
    "paired_t_success_p",
    "wilcoxon_success_stat",
    "wilcoxon_success_p",
]


def compare_methods(eval_df: pd.DataFrame, anchor: str = "LTAM-DQN") -> pd.DataFrame:
    rows: list[dict] = []
    anchor_df = eval_df[eval_df["method"] == anchor]
    for method in sorted(eval_df["method"].unique()):
        if method == anchor:
            continue
        method_df = eval_df[eval_df["method"] == method]

        seed_anchor = anchor_df.groupby("train_seed")["success"].mean().sort_index()
        seed_method = method_df.groupby("train_seed")["success"].mean().sort_index()
        common_seeds = seed_anchor.index.intersection(seed_method.index)
        if len(common_seeds) > 1:
            t_stat, t_p = ttest_rel(seed_anchor.loc[common_seeds], seed_method.loc[common_seeds])
        else:
            t_stat, t_p = float("nan"), float("nan")

        pair_df = anchor_df.merge(
            method_df,
            on=["scenario_group", "scenario_id", "train_seed"],
            suffixes=("_anchor", "_method"),
        )
        if len(pair_df) > 0:
            try:
                w_stat, w_p = wilcoxon(pair_df["success_anchor"], pair_df["success_method"])
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
        else:
            w_stat, w_p = float("nan"), float("nan")

        rows.append(
            {
                "anchor_method": anchor,
                "compare_method": method,
                "paired_t_success_stat": t_stat,
                "paired_t_success_p": t_p,
                "wilcoxon_success_stat": w_stat,
                "wilcoxon_success_p": w_p,
            }
        )
    return pd.DataFrame(rows, columns=STATS_COLUMNS)


def write_stats(eval_df: pd.DataFrame, output_root: str | Path, anchors: list[str] | None = None) -> Path:
    output_root = Path(output_root)
    stats_dir = ensure_dir(output_root / "stats")
    output_path = stats_dir / "stat_tests_main.csv"
    methods = set(eval_df["method"].dropna().astype(str).tolist()) if "method" in eval_df.columns else set()
    selected_anchors = anchors if anchors is not None else list(DEFAULT_STAT_ANCHORS)
    selected_anchors = [anchor for anchor in selected_anchors if anchor in methods]
    frames = [compare_methods(eval_df, anchor=anchor) for anchor in selected_anchors]
    stats_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STATS_COLUMNS)
    stats_df.to_csv(output_path, index=False)
    return output_path
