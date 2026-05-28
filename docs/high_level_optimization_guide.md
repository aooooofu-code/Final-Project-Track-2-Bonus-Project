# High-Level Optimization Guide

The starter high-level planner is weak on purpose.

## Starter Search Command

```bash
python train_highlevel_starter.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --output-dir artifacts/highlevel_train \
  --iterations 8 \
  --population 12
```

This is black-box search over starter planner parameters. It optimizes
`scores.composite_score` from `run_track_bonus.py`.

## Possible Directions

- Tune `speed_mps`, `k_heading`, `k_lateral`, and command scaling inside the
  starter planner.
- Replace `track_bonus/planner.py` with an MLP or RL controller.
- Train the low-level policy to track nonzero `vy` and `yaw_rate`.
- Use staged evaluation: straight, turn entry, turn middle, turn exit, full lap.

## Useful Metrics

- `lap_completion`, `valid_distance_m`, `finish_time`
- `fall`, `boundary_violation`
- lateral error, slip, energy

## Loop

1. Run starter eval.
2. Inspect `results.json` and `race.mp4`.
3. Improve low-level tracking or high-level planner.
4. Re-evaluate.
