from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    use_temporal_difference: bool
    use_behavior_mask: bool
    use_target_mask: bool
    temporal_delta_scale: float = 1.0
    temporal_delta_gate_threshold: float | None = None
    temporal_delta_gate_min_scale: float = 1.0

    def state_kwargs(self) -> dict[str, Any]:
        return {
            "use_temporal_difference": self.use_temporal_difference,
            "temporal_delta_scale": self.temporal_delta_scale,
            "temporal_delta_gate_threshold": self.temporal_delta_gate_threshold,
            "temporal_delta_gate_min_scale": self.temporal_delta_gate_min_scale,
        }


METHOD_SPECS: dict[str, MethodSpec] = {
    "DQN": MethodSpec("DQN", False, False, False),
    "DQN-T": MethodSpec("DQN-T", True, False, False),
    "DQN-AM-noTarget": MethodSpec("DQN-AM-noTarget", False, True, False),
    "DQN-AM": MethodSpec("DQN-AM", False, True, True),
    "LTAM-DQN": MethodSpec("LTAM-DQN", True, True, True),
    "LTAM-DQN-Adj": MethodSpec(
        "LTAM-DQN-Adj",
        True,
        True,
        True,
        temporal_delta_scale=0.5,
        temporal_delta_gate_threshold=0.10,
        temporal_delta_gate_min_scale=0.25,
    ),
}

DEFAULT_METHOD_ORDER = ["DQN", "DQN-T", "DQN-AM-noTarget", "DQN-AM", "LTAM-DQN"]


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))
