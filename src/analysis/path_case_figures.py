from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir


def _load_eval_rows(result_root: Path, scenario_group: str, method: str) -> pd.DataFrame:
    files = sorted((result_root / "metrics_csv" / scenario_group / method).glob("seed_*/eval_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"no eval_metrics.csv files for {scenario_group}/{method}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trajectory_path(result_root: Path, scenario_group: str, method: str, seed: int, scenario_id: str) -> Path:
    return result_root / "path_visualizations" / scenario_group / method / f"seed_{seed}" / f"{scenario_id}.json"


def select_path_cases(
    result_root: str | Path,
    *,
    scenario_group: str,
    anchor_method: str = "LTAM-DQN",
    baseline_method: str = "DQN",
    top_k: int = 3,
) -> pd.DataFrame:
    result_root = Path(result_root)
    anchor_df = _load_eval_rows(result_root, scenario_group, anchor_method)
    baseline_df = _load_eval_rows(result_root, scenario_group, baseline_method)
    merged = anchor_df.merge(
        baseline_df,
        on=["scenario_group", "scenario_id", "train_seed"],
        suffixes=("_anchor", "_baseline"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["success_gain"] = merged["success_anchor"] - merged["success_baseline"]
    merged["efficiency_gain"] = merged["path_efficiency_anchor"] - merged["path_efficiency_baseline"]
    merged["length_gain"] = merged["path_length_baseline"] - merged["path_length_anchor"]
    merged["case_score"] = (
        merged["success_gain"] * 100.0
        + merged["efficiency_gain"] * 10.0
        + merged["length_gain"].clip(lower=0) * 0.1
    )
    merged = merged.sort_values(
        ["case_score", "success_gain", "efficiency_gain", "length_gain"],
        ascending=False,
    )
    selected = merged.head(top_k).copy()
    selected = selected.rename(
        columns={
            "train_seed": "seed",
            "success_anchor": "anchor_success",
            "success_baseline": "baseline_success",
            "collision_anchor": "anchor_collision",
            "collision_baseline": "baseline_collision",
            "timeout_anchor": "anchor_timeout",
            "timeout_baseline": "baseline_timeout",
            "path_efficiency_anchor": "anchor_path_efficiency",
            "path_efficiency_baseline": "baseline_path_efficiency",
            "path_length_anchor": "anchor_path_length",
            "path_length_baseline": "baseline_path_length",
        }
    )
    selected.insert(3, "anchor_method", anchor_method)
    selected.insert(4, "baseline_method", baseline_method)
    keep_columns = [
        "scenario_group",
        "scenario_id",
        "seed",
        "anchor_method",
        "baseline_method",
        "anchor_success",
        "baseline_success",
        "anchor_collision",
        "baseline_collision",
        "anchor_timeout",
        "baseline_timeout",
        "anchor_path_efficiency",
        "baseline_path_efficiency",
        "anchor_path_length",
        "baseline_path_length",
        "success_gain",
        "efficiency_gain",
        "length_gain",
        "case_score",
    ]
    return selected[keep_columns].reset_index(drop=True)


def _scenario_lookup(result_root: Path, scenario_group: str) -> dict[str, dict[str, Any]]:
    scenario_path = result_root / "configs" / f"{scenario_group}.json"
    if not scenario_path.exists():
        return {}
    scenarios = _load_json(scenario_path)
    return {item["scenario_id"]: item for item in scenarios}


def _draw_grid(ax: plt.Axes, map_size: int) -> None:
    ax.set_xlim(-0.5, map_size - 0.5)
    ax.set_ylim(map_size - 0.5, -0.5)
    ax.set_xticks(range(map_size))
    ax.set_yticks(range(map_size))
    ax.grid(True, color="#d6d6d6", linewidth=0.6)
    ax.set_aspect("equal")


def _plot_positions(ax: plt.Axes, positions: list[list[int]], *, color: str, label: str) -> None:
    if not positions:
        return
    xs = [pos[1] for pos in positions]
    ys = [pos[0] for pos in positions]
    ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=3.0, label=label)


def _plot_dynamic_obstacles(ax: plt.Axes, snapshots: list[list[list[int]]]) -> None:
    if not snapshots:
        return
    first = snapshots[0]
    last = snapshots[-1]
    if first:
        ax.scatter([pos[1] for pos in first], [pos[0] for pos in first], marker="s", s=70, c="#c9a227", label="dynamic start")
    if last:
        ax.scatter([pos[1] for pos in last], [pos[0] for pos in last], marker="X", s=70, c="#7f5f00", label="dynamic end")


def render_path_case(
    *,
    result_root: str | Path,
    case: pd.Series,
    output_dir: str | Path,
) -> Path:
    result_root = Path(result_root)
    output_dir = ensure_dir(output_dir)
    scenario_group = str(case["scenario_group"])
    scenario_id = str(case["scenario_id"])
    seed = int(case["seed"])
    anchor_method = str(case["anchor_method"])
    baseline_method = str(case["baseline_method"])
    scenario = _scenario_lookup(result_root, scenario_group).get(scenario_id, {})
    anchor = _load_json(_trajectory_path(result_root, scenario_group, anchor_method, seed, scenario_id))
    baseline = _load_json(_trajectory_path(result_root, scenario_group, baseline_method, seed, scenario_id))
    map_size = int(scenario.get("map_size", max(anchor["goal"] + anchor["start"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
    for ax, method, trajectory, color, success_col, efficiency_col in [
        (axes[0], baseline_method, baseline, "#4f6f8f", "baseline_success", "baseline_path_efficiency"),
        (axes[1], anchor_method, anchor, "#c84b31", "anchor_success", "anchor_path_efficiency"),
    ]:
        _draw_grid(ax, map_size)
        _plot_dynamic_obstacles(ax, trajectory.get("dynamic_obstacle_positions", []))
        start = trajectory["start"]
        goal = trajectory["goal"]
        ax.scatter(start[1], start[0], marker="o", s=100, c="#2b8c4b", label="start", zorder=4)
        ax.scatter(goal[1], goal[0], marker="*", s=160, c="#d1495b", label="goal", zorder=4)
        _plot_positions(ax, trajectory.get("agent_positions", []), color=color, label="agent path")
        ax.set_title(
            f"{method}: {trajectory.get('done_reason')}\n"
            f"success={int(case[success_col])}, eff={float(case[efficiency_col]):.3f}"
        )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    safe_name = f"{scenario_group}_seed{seed}_{scenario_id}_{anchor_method}_vs_{baseline_method}".replace("/", "_")
    output_path = output_dir / f"{safe_name}.png"
    fig.suptitle(f"{scenario_group} {scenario_id}: {anchor_method} vs {baseline_method}", fontsize=11)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def build_path_case_artifacts(
    result_root: str | Path,
    *,
    scenario_group: str = "E2-dense",
    anchor_method: str = "LTAM-DQN",
    baseline_method: str = "DQN",
    top_k: int = 3,
) -> Path:
    result_root = Path(result_root)
    output_dir = ensure_dir(result_root / "figures_data" / "path_cases" / f"{scenario_group}_{anchor_method}_vs_{baseline_method}")
    selected = select_path_cases(
        result_root,
        scenario_group=scenario_group,
        anchor_method=anchor_method,
        baseline_method=baseline_method,
        top_k=top_k,
    )
    selected.to_csv(output_dir / "path_case_index.csv", index=False)
    for _, row in selected.iterrows():
        render_path_case(result_root=result_root, case=row, output_dir=output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--scenario-group", default="E2-dense")
    parser.add_argument("--anchor-method", default="LTAM-DQN")
    parser.add_argument("--baseline-method", default="DQN")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    out = build_path_case_artifacts(
        args.input_root,
        scenario_group=args.scenario_group,
        anchor_method=args.anchor_method,
        baseline_method=args.baseline_method,
        top_k=args.top_k,
    )
    print(out)


if __name__ == "__main__":
    main()
