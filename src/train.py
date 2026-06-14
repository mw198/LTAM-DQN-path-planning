from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.agents.dqn_agent import DQNAgent, DQNConfig
from src.analysis.metrics import compute_convergence_episode
from src.envs.dynamic_grid_env import DynamicGridEnv
from src.utils import METHOD_SPECS, ensure_dir, load_yaml, set_global_seed


def _artifact_paths(output_root: Path, method_name: str, train_seed: int) -> dict[str, Path]:
    logs_dir = ensure_dir(output_root / "logs" / method_name)
    checkpoint_dir = ensure_dir(output_root / "checkpoints" / method_name / f"seed_{train_seed}")
    return {
        "train_log": logs_dir / f"train_seed_{train_seed}.csv",
        "best_ckpt": checkpoint_dir / "best.pt",
        "last_ckpt": checkpoint_dir / "last.pt",
    }


def _evaluate_on_validation_scenarios(
    *,
    agent: DQNAgent,
    method_name: str,
    scenarios: list[dict],
) -> float:
    method_spec = METHOD_SPECS[method_name]
    state_kwargs = method_spec.state_kwargs()
    successes = 0
    for scenario in scenarios:
        env = DynamicGridEnv(
            map_size=scenario["map_size"],
            local_window=5,
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
        while not (terminated or truncated):
            valid_mask = env.get_action_mask()
            action = agent.act(state, valid_mask=valid_mask, greedy=True)
            _next_state_raw, _reward, terminated, truncated, info = env.step(action)
            state = env.get_state(**state_kwargs)
        successes += int(info["success"])
    return successes / max(1, len(scenarios))


def _normalise_train_scenario_mix(train_scenario_mix: list[dict] | None) -> list[dict]:
    if not train_scenario_mix:
        return []
    total_weight = float(sum(float(entry["weight"]) for entry in train_scenario_mix))
    if total_weight <= 0.0:
        raise ValueError("train_scenario_mix must have a positive total weight")
    normalised = []
    for entry in train_scenario_mix:
        normalised.append(
            {
                "name": entry["name"],
                "weight": float(entry["weight"]) / total_weight,
                "env_overrides": dict(entry.get("env_overrides", {})),
            }
        )
    return normalised


def _sample_train_scenario(mix: list[dict], rng: np.random.Generator) -> dict:
    probabilities = np.array([entry["weight"] for entry in mix], dtype=np.float64)
    index = int(rng.choice(len(mix), p=probabilities))
    return mix[index]


def train_method(
    *,
    project_root: str | Path,
    output_root: str | Path,
    method_name: str,
    train_seed: int,
    episodes: int,
    env_overrides: dict | None = None,
    train_scenario_mix: list[dict] | None = None,
    val_scenarios: list[dict] | None = None,
    val_interval: int = 50,
    device: str | None = None,
) -> pd.DataFrame:
    project_root = Path(project_root)
    output_root = Path(output_root)
    set_global_seed(train_seed)

    base_config = load_yaml(project_root / "configs" / "default.yaml")
    env_config = dict(base_config["env"])
    env_config.update(env_overrides or {})
    method_spec = METHOD_SPECS[method_name]
    state_kwargs = method_spec.state_kwargs()
    scenario_mix = _normalise_train_scenario_mix(train_scenario_mix)
    scenario_rng = np.random.default_rng(train_seed + 7919)

    env = DynamicGridEnv(seed=train_seed, **env_config)
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
    paths = _artifact_paths(output_root, method_name, train_seed)

    rows: list[dict] = []
    success_history: list[float] = []
    best_score = -1.0

    for episode in range(episodes):
        train_scenario_name = "default"
        if scenario_mix:
            scenario_entry = _sample_train_scenario(scenario_mix, scenario_rng)
            episode_env_config = dict(env_config)
            episode_env_config.update(scenario_entry["env_overrides"])
            env = DynamicGridEnv(seed=train_seed, **episode_env_config)
            train_scenario_name = str(scenario_entry["name"])
        env.reset(seed=train_seed * 100000 + episode)
        state = env.get_state(**state_kwargs)
        terminated = False
        truncated = False
        episode_reward = 0.0
        mask_interventions = 0
        update_losses: list[float] = []

        while not (terminated or truncated):
            valid_mask = env.get_action_mask()
            action, intervention = agent.act_with_info(
                state,
                valid_mask=valid_mask,
                greedy=False,
            )
            _next_state_raw, reward, terminated, truncated, info = env.step(action)
            next_state = env.get_state(**state_kwargs)
            next_valid_mask = env.get_action_mask()

            agent.store(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated or truncated,
                valid_mask=valid_mask,
                next_valid_mask=next_valid_mask,
                info=info,
            )
            agent.global_step += 1
            update_info = agent.update()
            if update_info is not None:
                update_losses.append(update_info["loss"])

            state = next_state
            episode_reward += float(reward)
            mask_interventions += int(intervention)

        success = int(info["success"])
        success_history.append(success)
        trailing_success = float(sum(success_history[-100:]) / max(1, len(success_history[-100:])))
        if val_scenarios and (episode + 1) % max(1, val_interval) == 0:
            val_score = _evaluate_on_validation_scenarios(
                agent=agent,
                method_name=method_name,
                scenarios=val_scenarios,
            )
            if val_score > best_score:
                best_score = val_score
                agent.save(str(paths["best_ckpt"]))
        elif episode >= 50 and trailing_success > best_score:
            best_score = trailing_success
            agent.save(str(paths["best_ckpt"]))

        rows.append(
            {
                "method": method_name,
                "train_seed": train_seed,
                "episode": episode + 1,
                "train_scenario": train_scenario_name,
                "reward": episode_reward,
                "success": success,
                "collision": int(info["collision"]),
                "timeout": int(info["timeout"]),
                "steps": int(info["steps"]),
                "path_length": int(info["path_length"]),
                "invalid_action_ratio": float(info["invalid_action_ratio"]),
                "mask_intervention_rate": mask_interventions / max(1, int(info["steps"])),
                "epsilon": agent.get_epsilon(),
                "avg_loss": float(sum(update_losses) / len(update_losses)) if update_losses else 0.0,
            }
        )

    if not paths["best_ckpt"].exists():
        agent.save(str(paths["best_ckpt"]))
    agent.save(str(paths["last_ckpt"]))

    train_df = pd.DataFrame(rows)
    train_df["reward_ma_100"] = train_df["reward"].rolling(100, min_periods=1).mean()
    train_df["success_rate_100"] = train_df["success"].rolling(100, min_periods=1).mean()
    train_df["collision_rate_100"] = train_df["collision"].rolling(100, min_periods=1).mean()
    train_df["timeout_rate_100"] = train_df["timeout"].rolling(100, min_periods=1).mean()
    train_df["invalid_action_ratio_100"] = train_df["invalid_action_ratio"].rolling(100, min_periods=1).mean()
    train_df["mask_intervention_rate_100"] = train_df["mask_intervention_rate"].rolling(100, min_periods=1).mean()
    convergence_episode = compute_convergence_episode(train_df["success"].tolist())
    train_df["convergence_episode"] = convergence_episode if convergence_episode is not None else -1
    train_df.to_csv(paths["train_log"], index=False)
    return train_df
