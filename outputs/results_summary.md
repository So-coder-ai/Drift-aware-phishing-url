# Final experiment summary

## Model selection

The CharCNN was trained on 13,327,517 URLs over eight epochs. The validation
checkpoint was selected from November 2022 by macro-F1, with a phishing
threshold calibrated to 0.36.

| Stage | Macro-F1 | Phishing precision | Phishing recall |
| --- | ---: | ---: | ---: |
| November validation | 0.9096 | 0.8930 | 0.8206 |
| Frozen Dec-Feb baseline | 0.8846 | 0.8504 | 0.7734 |
| ADWIN-aware RL threshold policy | 0.8464 | 0.6703 | 0.8732 |

## Drift monitoring

MMD compared each test month's frozen CNN embeddings against the November
reference. Every month exceeded the MMD alert threshold of 0.00104.

| Month | MMD² | Drift alert |
| --- | ---: | --- |
| December 2022 | 0.03105 | Yes |
| January 2023 | 0.04378 | Yes |
| February 2023 | 0.04417 | Yes |

ADWIN monitored the chronological binary prediction-error stream and generated
484 raw change alerts. These are fine-grained online alerts, not 484 independent
long-term concept-drift incidents.

## Interpretation

The frozen model's macro-F1 fell from 0.9096 in validation to 0.8846 in the
future test stream, while MMD and ADWIN both signalled distribution/performance
changes. RL threshold adaptation improved phishing recall but lowered precision
and macro-F1, so the fixed-threshold frozen baseline remains the stronger model
under macro-F1 evaluation.
