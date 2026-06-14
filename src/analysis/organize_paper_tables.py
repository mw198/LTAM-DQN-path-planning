from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.conclusion_boundaries import build_conclusion_boundaries


METHOD_ORDER = ["DQN", "DQN-T", "DQN-AM-noTarget", "DQN-AM", "LTAM-DQN", "LTAM-DQN-Adj"]
SCENARIO_ORDER = ["E0-main", "E1-large", "E2-dense", "E3-random", "E4-unseen-motion", "E5-unseen-start-goal", "E6-static"]
SCENARIO_LABELS = {
    "E0-main": "主场景",
    "E1-large": "大地图泛化",
    "E2-dense": "高密度障碍",
    "E3-random": "高随机运动",
    "E4-unseen-motion": "未见运动模式",
    "E5-unseen-start-goal": "未见起终点",
    "E6-static": "静态环境",
}
ABLATION_FLAGS = {
    "DQN": ("x", "x", "x"),
    "DQN-T": ("check", "x", "x"),
    "DQN-AM-noTarget": ("x", "check", "x"),
    "DQN-AM": ("x", "check", "check"),
    "LTAM-DQN": ("check", "check", "check"),
    "LTAM-DQN-Adj": ("check", "check", "check"),
}


def _read_csv(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / "summary_tables" / name)


def _ordered_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    return out.sort_values("method").reset_index(drop=True)


def _scale_rate_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = (out[col] * 100.0).round(2)
    return out


def build_main_table(main_df: pd.DataFrame) -> pd.DataFrame:
    out = _ordered_methods(main_df)
    out = _scale_rate_columns(
        out,
        [
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "path_efficiency",
            "invalid_action_execution_rate",
            "mask_intervention_rate",
        ],
    )
    out["avg_path_length"] = out["avg_path_length"].round(3)
    out["avg_reward"] = out["avg_reward"].round(4)
    out["avg_inference_time_ms"] = out["avg_inference_time_ms"].round(4)
    out["num_parameters"] = out["num_parameters"].astype(int)
    out = out.rename(
        columns={
            "method": "方法",
            "success_rate": "成功率(%)",
            "collision_rate": "碰撞率(%)",
            "timeout_rate": "超时率(%)",
            "avg_path_length": "平均路径长度",
            "path_efficiency": "路径效率(%)",
            "avg_reward": "平均奖励",
            "invalid_action_execution_rate": "无效动作执行率(%)",
            "mask_intervention_rate": "掩码介入率(%)",
            "avg_inference_time_ms": "平均推理时间(ms)",
            "num_parameters": "参数量",
        }
    )
    return out


def build_ablation_table(main_df: pd.DataFrame) -> pd.DataFrame:
    out = _ordered_methods(main_df)
    out[["时序差分", "行为掩码", "目标掩码"]] = out["method"].astype(str).map(ABLATION_FLAGS).apply(pd.Series)
    out = _scale_rate_columns(out, ["success_rate", "collision_rate", "timeout_rate", "path_efficiency", "mask_intervention_rate"])
    out["avg_path_length"] = out["avg_path_length"].round(3)
    out = out[
        [
            "method",
            "时序差分",
            "行为掩码",
            "目标掩码",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "avg_path_length",
            "path_efficiency",
            "mask_intervention_rate",
        ]
    ].rename(
        columns={
            "method": "方法",
            "success_rate": "成功率(%)",
            "collision_rate": "碰撞率(%)",
            "timeout_rate": "超时率(%)",
            "avg_path_length": "平均路径长度",
            "path_efficiency": "路径效率(%)",
            "mask_intervention_rate": "掩码介入率(%)",
        }
    )
    return out


def build_generalization_tables(general_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = general_df.copy()
    out["scenario_group"] = pd.Categorical(out["scenario_group"], SCENARIO_ORDER, ordered=True)
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    out = out.sort_values(["scenario_group", "method"]).reset_index(drop=True)
    out = _scale_rate_columns(out, ["success_rate", "collision_rate", "timeout_rate", "path_efficiency"])
    out["avg_reward"] = out["avg_reward"].round(4)
    out["场景"] = out["scenario_group"].map(SCENARIO_LABELS)
    long_table = out[
        ["scenario_group", "场景", "method", "success_rate", "collision_rate", "timeout_rate", "path_efficiency", "avg_reward"]
    ].rename(
        columns={
            "scenario_group": "场景编号",
            "method": "方法",
            "success_rate": "成功率(%)",
            "collision_rate": "碰撞率(%)",
            "timeout_rate": "超时率(%)",
            "path_efficiency": "路径效率(%)",
            "avg_reward": "平均奖励",
        }
    )
    pivot = (
        out.pivot(index="scenario_group", columns="method", values="success_rate")
        .reindex([item for item in SCENARIO_ORDER if item in set(out["scenario_group"].astype(str).tolist())])
        .reset_index()
        .rename(columns={"scenario_group": "场景编号"})
    )
    pivot.insert(1, "场景", pivot["场景编号"].map(SCENARIO_LABELS))
    return long_table, pivot


def build_complexity_table(complexity_df: pd.DataFrame) -> pd.DataFrame:
    out = _ordered_methods(complexity_df)
    out["avg_inference_time_ms"] = out["avg_inference_time_ms"].round(4)
    out["num_parameters"] = out["num_parameters"].astype(int)
    return out.rename(
        columns={
            "method": "方法",
            "avg_inference_time_ms": "平均推理时间(ms)",
            "num_parameters": "参数量",
        }
    )


def build_stats_table(stats_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    main_lookup = main_df.set_index("method")["success_rate"]
    out = stats_df.copy()
    out["anchor_success_rate"] = out["anchor_method"].map(main_lookup)
    out["compare_success_rate"] = out["compare_method"].map(main_lookup)
    out["anchor_minus_compare_success_pp"] = ((out["anchor_success_rate"] - out["compare_success_rate"]) * 100.0).round(2)
    out["paired_t_success_p"] = out["paired_t_success_p"].round(6)
    out["wilcoxon_success_p"] = out["wilcoxon_success_p"].round(6)
    out["配对t显著"] = out["paired_t_success_p"] < 0.05
    out["Wilcoxon显著"] = out["wilcoxon_success_p"] < 0.05
    out = out[
        [
            "anchor_method",
            "compare_method",
            "anchor_minus_compare_success_pp",
            "paired_t_success_p",
            "配对t显著",
            "wilcoxon_success_p",
            "Wilcoxon显著",
        ]
    ].rename(
        columns={
            "anchor_method": "锚定方法",
            "compare_method": "对比方法",
            "anchor_minus_compare_success_pp": "锚定方法相对成功率差(百分点)",
            "paired_t_success_p": "配对t检验p值",
            "wilcoxon_success_p": "Wilcoxon检验p值",
        }
    )
    return out


def build_summary_markdown(
    main_df: pd.DataFrame,
    general_df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> str:
    main_best = main_df.sort_values("success_rate", ascending=False).iloc[0]
    dense_df = general_df[general_df["scenario_group"] == "E2-dense"].sort_values("success_rate", ascending=False)
    dense_best = dense_df.iloc[0]
    static_df = general_df[general_df["scenario_group"] == "E6-static"].sort_values("success_rate", ascending=False)
    static_best = static_df.iloc[0]
    fastest = complexity_df.sort_values("avg_inference_time_ms", ascending=True).iloc[0]
    main_path_best = main_df.sort_values("path_efficiency", ascending=False).iloc[0]
    main_success_lookup = main_df.set_index("method")["success_rate"]
    if "LTAM-DQN" in main_success_lookup.index and "DQN" in main_success_lookup.index:
        ltam_success = main_success_lookup.loc["LTAM-DQN"]
        dqn_success = main_success_lookup.loc["DQN"]
        ltam_gap = (ltam_success - dqn_success) * 100.0
        ltam_main_line = (
            f"- `LTAM-DQN` 在主场景成功率为 `{ltam_success*100:.2f}%`，"
            f"相对基础 `DQN` 为 `{ltam_gap:+.2f}` 个百分点。"
        )
    elif "LTAM-DQN" in main_success_lookup.index:
        ltam_success = main_success_lookup.loc["LTAM-DQN"]
        ltam_main_line = f"- `LTAM-DQN` 在主场景成功率为 `{ltam_success*100:.2f}%`。"
    else:
        ltam_main_line = "- 当前结果中未包含原始 `LTAM-DQN` 主场景成功率。"

    lines = [
        "# LTAM-DQN 正式实验结果整理",
        "",
        "## 主场景结论",
        f"- `E0-main` 主场景成功率最高的方法是 `{main_best['method']}`，成功率为 `{main_best['success_rate']*100:.2f}%`。",
        ltam_main_line,
        f"- 主场景路径效率最高的方法是 `{main_path_best['method']}`，为 `{main_path_best['path_efficiency']*100:.2f}%`。",
        "",
        "## 泛化场景结论",
        f"- 在高密度障碍场景 `E2-dense` 中，成功率最高的方法是 `{dense_best['method']}`，成功率 `{dense_best['success_rate']*100:.2f}%`。",
        f"- 在静态场景 `E6-static` 中，表现最好的方法是 `{static_best['method']}`，成功率 `{static_best['success_rate']*100:.2f}%`。",
        "",
        "## 复杂度结论",
        f"- 推理时间最短的方法是 `{fastest['method']}`，平均推理时间 `{fastest['avg_inference_time_ms']:.4f} ms`。",
        f"- 所有方法参数量相同，均为 `{int(complexity_df['num_parameters'].iloc[0])}`。",
        "",
        "## 统计检验结论",
    ]
    for _, row in stats_df.iterrows():
        anchor_method = row["anchor_method"]
        compare_method = row["compare_method"]
        anchor_rate = main_success_lookup.get(anchor_method, float("nan"))
        compare_rate = main_success_lookup.get(compare_method, float("nan"))
        success_gap = (anchor_rate - compare_rate) * 100.0
        lines.append(
            f"- `{anchor_method}` 相对 `{compare_method}` 的成功率差为 `{success_gap:.2f}` 个百分点，"
            f"配对 t 检验 p 值 `{row['paired_t_success_p']:.6f}`，Wilcoxon p 值 `{row['wilcoxon_success_p']:.6f}`。"
        )
    return "\n".join(lines) + "\n"


def organize_tables(input_root: Path) -> Path:
    main_df = _read_csv(input_root, "main_results.csv")
    general_df = _read_csv(input_root, "generalization_results.csv")
    complexity_df = _read_csv(input_root, "complexity_results.csv")
    stats_df = _read_csv(input_root, "stat_tests.csv")

    organized_dir = input_root / "summary_tables" / "organized"
    organized_dir.mkdir(parents=True, exist_ok=True)

    main_table = build_main_table(main_df)
    ablation_table = build_ablation_table(main_df)
    general_long, general_pivot = build_generalization_tables(general_df)
    complexity_table = build_complexity_table(complexity_df)
    stats_table = build_stats_table(stats_df, main_df)
    summary_md = build_summary_markdown(main_df, general_df, complexity_df, stats_df)
    conclusion_md = build_conclusion_boundaries(main_df, general_df, stats_df)

    main_table.to_csv(organized_dir / "paper_main_results.csv", index=False, encoding="utf-8-sig")
    ablation_table.to_csv(organized_dir / "paper_ablation_results.csv", index=False, encoding="utf-8-sig")
    general_long.to_csv(organized_dir / "paper_generalization_results.csv", index=False, encoding="utf-8-sig")
    general_pivot.to_csv(organized_dir / "paper_generalization_success_pivot.csv", index=False, encoding="utf-8-sig")
    complexity_table.to_csv(organized_dir / "paper_complexity_results.csv", index=False, encoding="utf-8-sig")
    stats_table.to_csv(organized_dir / "paper_stat_tests.csv", index=False, encoding="utf-8-sig")
    (organized_dir / "paper_results_summary.md").write_text(summary_md, encoding="utf-8")
    (organized_dir / "paper_conclusion_boundaries.md").write_text(conclusion_md, encoding="utf-8")
    return organized_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True)
    args = parser.parse_args()
    out = organize_tables(Path(args.input_root))
    print(out)


if __name__ == "__main__":
    main()
