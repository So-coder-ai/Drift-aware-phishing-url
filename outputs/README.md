# Experiment outputs

The final exported metrics, error stream, and sampled embedding artifacts are
included here for reproducibility. The corresponding Kaggle Datasets remain
the backup location for the full run outputs.

- `results_summary.md` reports the validation, frozen-test, MMD, ADWIN, and RL
  outcomes.
- `training/` contains validation metrics, the confusion matrix, and epoch log.
- `frozen_test/` contains the chronological test metrics, error stream, and
  monthly embedding samples.
- `drift_rl/` contains MMD, ADWIN, and RL-policy outputs.
