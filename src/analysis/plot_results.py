from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir


def plot_training_metric(
    *,
    output_root: str | Path,
    methods: list[str],
    metric: str,
    output_filename: str,
    title: str,
) -> Path:
    output_root = Path(output_root)
    plt.figure(figsize=(8, 5))
    plotted = False
    for method in methods:
        method_dir = output_root / "logs" / method
        if not method_dir.exists():
            continue
        frames: list[pd.DataFrame] = []
        for csv_path in sorted(method_dir.glob("train_seed_*.csv")):
            frames.append(pd.read_csv(csv_path))
        if not frames:
            continue
        plotted = True
        merged = pd.concat(frames, ignore_index=True)
        grouped = merged.groupby("episode", as_index=False)[metric].mean()
        plt.plot(grouped["episode"], grouped[metric], label=method)
    if not plotted:
        raise FileNotFoundError("no training logs available for plotting")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    figures_dir = ensure_dir(output_root / "figures_data")
    output_path = figures_dir / output_filename
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
