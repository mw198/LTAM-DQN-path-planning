from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_dir


def summarize_main_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("method", as_index=False)
        .agg(
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            timeout_rate=("timeout", "mean"),
            avg_path_length=("path_length", "mean"),
            path_efficiency=("path_efficiency", "mean"),
            avg_reward=("episode_reward", "mean"),
            invalid_action_execution_rate=("invalid_action_execution_rate", "mean"),
            mask_intervention_rate=("mask_intervention_rate", "mean"),
            avg_inference_time_ms=("avg_inference_time_ms", "mean"),
            num_parameters=("num_parameters", "mean"),
        )
    )
    return grouped


def summarize_train_logs(train_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        train_df.groupby(["method", "train_seed"], as_index=False)
        .agg(
            final_reward_ma_100=("reward_ma_100", "last"),
            final_success_rate_100=("success_rate_100", "last"),
            final_collision_rate_100=("collision_rate_100", "last"),
            final_timeout_rate_100=("timeout_rate_100", "last"),
            final_invalid_action_ratio_100=("invalid_action_ratio_100", "last"),
            final_mask_intervention_rate_100=("mask_intervention_rate_100", "last"),
            convergence_episode=("convergence_episode", "last"),
        )
    )
    return grouped


def write_summary_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    return output_path
