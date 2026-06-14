# LTAM-DQN Path Planning

## Overview

This project is a fully independent implementation of the LTAM-DQN dynamic-grid path-planning pipeline. It does not import the legacy `rl_path_planning/` project at runtime.

The default formal methods are:

- `DQN`
- `DQN-T`
- `DQN-AM-noTarget`
- `DQN-AM`
- `LTAM-DQN`

## Project layout

```text
LTAM-DQN-path-planning/
  configs/
  src/
  tests/
  results/
  README.md
```

## Run tests

```bash
cd LTAM-DQN-path-planning
pytest tests -v
```

## Run sanity experiment

```bash
python -m src.run_experiments --stage sanity --episodes 100 --seed 0
```

## Run pilot experiment

```bash
python -m src.run_experiments --stage pilot --episodes 1000 --seed 0
```

## Run formal experiments

```bash
python -m src.run_experiments --stage formal --train-episodes 3000 --train-seeds 0,1,2,3,4
```

## Run adjusted LTAM quick-check experiments

`LTAM-DQN-Adj` keeps the behavior mask and target mask of `LTAM-DQN`, but scales sparse temporal-difference observations to reduce noise in easy scenes. The mixed-hard curriculum adds dense, random-motion, and unseen-motion episodes during training, while `main-dense` selects checkpoints on a 50/50 mix of E0-main and E2-dense validation scenes.

```bash
python -m src.run_experiments --stage formal --methods DQN,DQN-AM,LTAM-DQN,LTAM-DQN-Adj --train-curriculum mixed-hard --validation-profile main-dense --train-episodes 3000 --train-seeds 0,1,2,3,4 --output-root results_ltam_adjusted
```

## Main outputs

The pipeline writes outputs under `results/`:

- `configs/`: saved scenario splits
- `logs/`: training CSV logs
- `checkpoints/`: best and last model checkpoints
- `metrics_csv/`: evaluation CSV files
- `summary_tables/`: main, ablation, generalization, complexity, and stats tables
- `figures_data/`: generated training curves
- `path_visualizations/`: trajectory JSON files
- `stats/`: statistical test CSV

## Notes

- `wait` is part of the action space and is always available before termination.
- `next_valid_mask` is stored in replay transitions and used by `DQN-AM` and `LTAM-DQN` target computation.
- The current implementation focuses on reproducible dynamic-grid experiments for the LTAM-DQN paper workflow.
