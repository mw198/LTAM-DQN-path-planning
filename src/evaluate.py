from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import torch

from src.agents.dqn_agent import DQNAgent, DQNConfig
from src.analysis.metrics import compute_path_efficiency, shortest_path_length_static
from src.envs.dynamic_grid_env import DynamicGridEnv
from src.utils import METHOD_SPECS, count_parameters, ensure_dir, load_yaml, save_json, set_global_seed


def evaluate_method(
    *,
    project_root: str | Path,
    output_root: str | Path,
    method_name: str,
    train_seed: int,
    scenarios: list[dict],
    scenario_group: str,
    device: str | None = None,
) -> pd.DataFrame:
    project_root = Path(project_root)
    output_root = Path(output_root)
    set_global_seed(train_seed)

    base_config = load_yaml(project_root / "configs" / "default.yaml")
    method_spec = METHOD_SPECS[method_name]
    state_kwargs = method_spec.state_kwargs()
    agent = DQNAgent(
        config=DQNConfig(
            gamma=base_config["train"]["gamma"],
            lr=base_config["train"]["lr"],
            batch_size=base_config["train"]["batch_size"],
            buffer_size=base_config["train"]["buffer_size"],
            min_buffer_size=base_config["train"]["min_buffer_size"],
            target_update_freq=base_config["train"]["target_update_freq"],
            epsilon_start=base_config["train"]["epsilon_start"],
            epsilon_end=base_config["train"]["epsilon_end"],
            epsilon_decay_steps=base_config["train"]["epsilon_decay_steps"],
            use_temporal_difference=method_spec.use_temporal_difference,
            use_behavior_mask=method_spec.use_behavior_mask,
            use_target_mask=method_spec.use_target_mask,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            seed=train_seed,
        )
    )
    checkpoint_path = output_root / "checkpoints" / method_name / f"seed_{train_seed}" / "best.pt"
    agent.load(str(checkpoint_path))

    metrics_dir = ensure_dir(output_root / "metrics_csv" / scenario_group / method_name / f"seed_{train_seed}")
    traj_dir = ensure_dir(output_root / "path_visualizations" / scenario_group / method_name / f"seed_{train_seed}")

    rows: list[dict] = []
    for scenario in scenarios:
        env = DynamicGridEnv(
            map_size=scenario["map_size"],
            local_window=base_config["env"]["local_window"],
            num_static_obstacles=scenario["num_static_obstacles"],
            num_dynamic_obstacles=scenario["num_dynamic_obstacles"],
            dynamic_motion_mode=scenario["dynamic_motion_mode"],
            obstacle_density=scenario["obstacle_density"],
            max_steps=scenario["max_steps"],
            start_goal_mode=scenario["start_goal_mode"],
            seed=scenario["seed"],
        )
        env.reset(seed=scenario["seed"])
        state = env.get_state(**state_kwargs)
        terminated = False
        truncated = False
        decision_time_sum = 0.0
        mask_interventions = 0
        episode_reward = 0.0

        while not (terminated or truncated):
            valid_mask = env.get_action_mask()
            start_time = time.perf_counter()
            action, intervention = agent.act_with_info(state, valid_mask=valid_mask, greedy=True)
            decision_time_sum += time.perf_counter() - start_time
            _next_state_raw, reward, terminated, truncated, info = env.step(action)
            state = env.get_state(**state_kwargs)
            mask_interventions += int(intervention)
            episode_reward += float(reward)

        trajectory = env.get_trajectory()
        trajectory.update(
            {
                "scenario_id": scenario["scenario_id"],
                "method": method_name,
                "seed": train_seed,
            }
        )
        save_json(traj_dir / f"{scenario['scenario_id']}.json", trajectory)

        reference_length = shortest_path_length_static(
            map_size=scenario["map_size"],
            start=trajectory["start"],
            goal=trajectory["goal"],
            static_obstacles=list(env.static_obstacles),
        )
        actual_path_length = max(0, len(trajectory["agent_positions"]) - 1)
        rows.append(
            {
                "scenario_group": scenario_group,
                "scenario_id": scenario["scenario_id"],
                "method": method_name,
                "train_seed": train_seed,
                "success": int(info["success"]),
                "collision": int(info["collision"]),
                "timeout": int(info["timeout"]),
                "path_length": actual_path_length,
                "episode_reward": episode_reward,
                "invalid_action_execution_rate": float(info["invalid_action_ratio"]),
                "mask_intervention_rate": mask_interventions / max(1, int(info["steps"])),
                "avg_inference_time_ms": (decision_time_sum / max(1, int(info["steps"]))) * 1000.0,
                "num_parameters": count_parameters(agent.q_net),
                "path_efficiency": compute_path_efficiency(
                    success=bool(info["success"]),
                    reference_length=reference_length,
                    actual_path_length=actual_path_length,
                ),
            }
        )

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(metrics_dir / "eval_metrics.csv", index=False)
    return eval_df
