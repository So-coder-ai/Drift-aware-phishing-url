# Figure plan

Use the saved Kaggle CSV/JSON outputs to create these figures for the report:

1. **Training curves:** `epoch_metrics.csv`, with training loss, validation
   loss, and validation macro-F1 by epoch.
2. **Temporal performance:** monthly frozen-test macro-F1 and phishing recall
   for December 2022 through February 2023.
3. **MMD drift timeline:** monthly MMD² values with the 0.00104 alert line.
4. **ADWIN alert timeline:** alert counts grouped by day or week, rather than
   plotting all 484 individual alerts.
5. **Baseline vs RL comparison:** macro-F1, phishing precision, and phishing
   recall for the fixed threshold and RL-adjusted threshold policies.

Figures should be generated from the private Kaggle result datasets; the large
raw arrays and prediction streams are intentionally not committed to Git.
