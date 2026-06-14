from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.plot_results import plot_training_metric
from src.analysis.stats_tests import write_stats
from src.analysis.summarize_results import summarize_main_results, summarize_train_logs, write_summary_csv
from src.envs.scenario_generator import generate_fixed_scenarios, save_scenarios
from src.evaluate import evaluate_method
from src.train import train_method
from src.utils import DEFAULT_METHOD_ORDER, METHOD_SPECS, PROJECT_ROOT, ensure_dir, parse_seed_list, save_json


def resolve_methods(methods_arg: str | None) -> list[str]:
    if not methods_arg:
        return list(DEFAULT_METHOD_ORDER)
    requested = [item.strip() for item in methods_arg.split(",") if item.strip()]
    invalid = [name for name in requested if name not in METHOD_SPECS]
    if invalid:
        raise ValueError(f"unknown methods requested: {invalid}")
    return requested


def _prepare_output_root(output_root: str | Path | None) -> Path:
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "results"
    ensure_dir(root)
    ensure_dir(root / "configs")
    ensure_dir(root / "logs")
    ensure_dir(root / "checkpoints")
    ensure_dir(root / "metrics_csv")
    ensure_dir(root / "summary_tables")
    ensure_dir(root / "figures_data")
    ensure_dir(root / "path_visualizations")
    ensure_dir(root / "stats")
    return root


def _build_scenario_groups() -> dict[str, list[dict]]:
    return {
        "E0-main": generate_fixed_scenarios(
            count=500,
            seed=42,
            map_size=15,
            num_dynamic_obstacles=5,
            dynamic_motion_mode="mixed",
        ),
        "E1-large": generate_fixed_scenarios(
            count=200,
            seed=43,
            map_size=20,
            num_dynamic_obstacles=5,
            dynamic_motion_mode="mixed",
        ),
        "E2-dense": generate_fixed_scenarios(
            count=200,
            seed=44,
            map_size=15,
            num_dynamic_obstacles=5,
            dynamic_motion_mode="mixed",
            obstacle_density=0.15,
        ),
        "E3-random": generate_fixed_scenarios(
            count=200,
            seed=45,
            map_size=15,
            num_dynamic_obstacles=5,
            dynamic_motion_mode="random_walk",
        ),
        "E4-unseen-motion": generate_fixed_scenarios(
            count=200,
            seed=46,
            map_size=15,
            num_dynamic_obstacles=5,
            dynamic_motion_mode="vertical",
        ),
        "E5-unseen-start-goal": generate_fixed_scenarios(
            count=200,
            seed=47,
            map_size=15,
            num_dynamic_obstacles=5,
            dynamic_motion_mode="mixed",
            start_goal_mode="unseen",
        ),
        "E6-static": generate_fixed_scenarios(
            count=200,
            seed=48,
            map_size=15,
            num_dynamic_obstacles=0,
            dynamic_motion_mode="mixed",
        ),
    }


def _build_train_scenario_mix(profile: str) -> list[dict]:
    if profile == "none":
        return []
    if profile != "mixed-hard":
        raise ValueError(f"unknown train curriculum profile: {profile}")
    return [
        {"name": "E0-main", "weight": 0.50, "env_overrides": {}},
        {"name": "E2-dense", "weight": 0.30, "env_overrides": {"obstacle_density": 0.15}},
        {"name": "E3-random", "weight": 0.10, "env_overrides": {"dynamic_motion_mode": "random_walk"}},
        {"name": "E4-unseen-motion", "weight": 0.10, "env_overrides": {"dynamic_motion_mode": "vertical"}},
    ]


def _select_validation_scenarios(
    scenario_groups: dict[str, list[dict]],
    validation_profile: str,
) -> list[dict]:
    if validation_profile == "main":
        return scenario_groups["E0-main"][:100]
    if validation_profile == "main-dense":
        return scenario_groups["E0-main"][:50] + scenario_groups["E2-dense"][:50]
    raise ValueError(f"unknown validation profile: {validation_profile}")


def run_sanity_experiment(
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_root: str | Path | None = None,
    episodes: int = 100,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, pd.DataFrame]:
    project_root = Path(project_root)
    output_root = _prepare_output_root(output_root)
    val_scenarios = generate_fixed_scenarios(
        count=10,
        seed=seed + 900,
        map_size=15,
        num_dynamic_obstacles=5,
        dynamic_motion_mode="mixed",
    )
    save_scenarios(output_root / "configs" / "sanity_val_scenarios.json", val_scenarios)
    train_df = train_method(
        project_root=project_root,
        output_root=output_root,
        method_name="LTAM-DQN",
        train_seed=seed,
        episodes=episodes,
        val_scenarios=val_scenarios,
        val_interval=10,
        device=device,
    )
    scenarios = generate_fixed_scenarios(
        count=5,
        seed=seed + 1000,
        map_size=15,
        num_dynamic_obstacles=5,
        dynamic_motion_mode="mixed",
    )
    save_scenarios(output_root / "configs" / "sanity_scenarios_seed42.json", scenarios)
    eval_df = evaluate_method(
        project_root=project_root,
        output_root=output_root,
        method_name="LTAM-DQN",
        train_seed=seed,
        scenarios=scenarios,
        scenario_group="sanity",
        device=device,
    )
    summary_df = summarize_main_results(eval_df)
    write_summary_csv(summary_df, output_root / "summary_tables" / "sanity_results.csv")
    return {"train": train_df, "eval": eval_df, "summary": summary_df}


def run_pilot_experiment(
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_root: str | Path | None = None,
    episodes: int = 1000,
    seed: int = 0,
    device: str | None = None,
    methods: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    project_root = Path(project_root)
    output_root = _prepare_output_root(output_root)
    methods = methods or ["DQN", "DQN-T", "DQN-AM", "LTAM-DQN"]
    train_frames: list[pd.DataFrame] = []
    eval_frames: list[pd.DataFrame] = []
    scenarios = generate_fixed_scenarios(
        count=20,
        seed=seed + 2000,
        map_size=15,
        num_dynamic_obstacles=5,
        dynamic_motion_mode="mixed",
    )
    val_scenarios = generate_fixed_scenarios(
        count=20,
        seed=seed + 1500,
        map_size=15,
        num_dynamic_obstacles=5,
        dynamic_motion_mode="mixed",
    )
    save_scenarios(output_root / "configs" / "pilot_scenarios_seed42.json", scenarios)
    save_scenarios(output_root / "configs" / "pilot_val_scenarios_seed42.json", val_scenarios)
    for method in methods:
        train_frames.append(
            train_method(
                project_root=project_root,
                output_root=output_root,
                method_name=method,
                train_seed=seed,
                episodes=episodes,
                val_scenarios=val_scenarios,
                val_interval=25,
                device=device,
            )
        )
        eval_frames.append(
            evaluate_method(
                project_root=project_root,
                output_root=output_root,
                method_name=method,
                train_seed=seed,
                scenarios=scenarios,
                scenario_group="pilot",
                device=device,
            )
        )
    train_df = pd.concat(train_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True)
    write_summary_csv(summarize_train_logs(train_df), output_root / "summary_tables" / "pilot_train_summary.csv")
    write_summary_csv(summarize_main_results(eval_df), output_root / "summary_tables" / "pilot_eval_summary.csv")
    return {"train": train_df, "eval": eval_df}


def run_formal_experiments(
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_root: str | Path | None = None,
    train_episodes: int = 3000,
    train_seeds: list[int] | None = None,
    device: str | None = None,
    methods: list[str] | None = None,
    train_curriculum: str = "none",
    validation_profile: str = "main",
) -> dict[str, pd.DataFrame]:
    project_root = Path(project_root)
    output_root = _prepare_output_root(output_root)
    methods = methods or list(DEFAULT_METHOD_ORDER)
    seeds = train_seeds or [0, 1, 2, 3, 4]
    scenario_groups = _build_scenario_groups()
    train_scenario_mix = _build_train_scenario_mix(train_curriculum)
    val_scenarios = _select_validation_scenarios(scenario_groups, validation_profile)
    val_suffix = "" if validation_profile == "main" else f"_{validation_profile}"
    save_scenarios(output_root / "configs" / f"val_scenarios_seed42{val_suffix}.json", val_scenarios)
    if train_scenario_mix:
        save_json(output_root / "configs" / f"train_curriculum_{train_curriculum}.json", train_scenario_mix)
    save_scenarios(output_root / "configs" / "test_scenarios_seed42.json", scenario_groups["E0-main"])
    for group_name, scenarios in scenario_groups.items():
        save_scenarios(output_root / "configs" / f"{group_name}.json", scenarios)

    train_frames: list[pd.DataFrame] = []
    eval_frames: list[pd.DataFrame] = []
    for method in methods:
        for seed in seeds:
            train_frames.append(
                train_method(
                    project_root=project_root,
                    output_root=output_root,
                    method_name=method,
                    train_seed=seed,
                    episodes=train_episodes,
                    train_scenario_mix=train_scenario_mix,
                    val_scenarios=val_scenarios,
                    val_interval=100,
                    device=device,
                )
            )
            for scenario_group, scenarios in scenario_groups.items():
                eval_frames.append(
                    evaluate_method(
                        project_root=project_root,
                        output_root=output_root,
                        method_name=method,
                        train_seed=seed,
                        scenarios=scenarios,
                        scenario_group=scenario_group,
                        device=device,
                    )
                )

    train_df = pd.concat(train_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True)
    main_eval_df = eval_df[eval_df["scenario_group"] == "E0-main"].copy()
    generalization_df = eval_df[eval_df["scenario_group"] != "E0-main"].copy()
    complexity_df = (
        main_eval_df.groupby("method", as_index=False)
        .agg(
            avg_inference_time_ms=("avg_inference_time_ms", "mean"),
            num_parameters=("num_parameters", "mean"),
        )
    )
    write_summary_csv(summarize_train_logs(train_df), output_root / "summary_tables" / "train_summary.csv")
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
    plot_training_metric(
        output_root=output_root,
        methods=methods,
        metric="reward_ma_100",
        output_filename="training_reward_curve.png",
        title="Training Reward Curves",
    )
    plot_training_metric(
        output_root=output_root,
        methods=methods,
        metric="success_rate_100",
        output_filename="training_success_curve.png",
        title="Training Success Curves",
    )
    plot_training_metric(
        output_root=output_root,
        methods=methods,
        metric="collision_rate_100",
        output_filename="training_collision_curve.png",
        title="Training Collision Curves",
    )
    plot_training_metric(
        output_root=output_root,
        methods=methods,
        metric="timeout_rate_100",
        output_filename="training_timeout_curve.png",
        title="Training Timeout Curves",
    )
    plot_training_metric(
        output_root=output_root,
        methods=methods,
        metric="invalid_action_ratio_100",
        output_filename="invalid_action_curve.png",
        title="Invalid Action Curves",
    )
    plot_training_metric(
        output_root=output_root,
        methods=methods,
        metric="mask_intervention_rate_100",
        output_filename="mask_intervention_curve.png",
        title="Mask Intervention Curves",
    )
    return {"train": train_df, "eval": eval_df, "main_eval": main_eval_df, "generalization": generalization_df}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sanity", "pilot", "formal"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--train-curriculum", choices=["none", "mixed-hard"], default="none")
    parser.add_argument("--validation-profile", choices=["main", "main-dense"], default="main")
    args = parser.parse_args()
    selected_methods = resolve_methods(args.methods)

    if args.stage == "sanity":
        run_sanity_experiment(output_root=args.output_root, episodes=args.episodes, seed=args.seed, device=args.device)
    elif args.stage == "pilot":
        run_pilot_experiment(
            output_root=args.output_root,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            methods=selected_methods,
        )
    else:
        run_formal_experiments(
            output_root=args.output_root,
            train_episodes=args.train_episodes,
            train_seeds=parse_seed_list(args.train_seeds),
            device=args.device,
            methods=selected_methods,
            train_curriculum=args.train_curriculum,
            validation_profile=args.validation_profile,
        )


if __name__ == "__main__":
    main()
