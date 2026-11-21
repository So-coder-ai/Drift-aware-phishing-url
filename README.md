# Drift-aware phishing URL detection

This project implements the instructor-specified temporal experiment using
DeepURLBench's `urls_without_dns` configuration. Only `url`, `first_seen`
(renamed to `date`), and `label` are used. `date` controls chronological
splits and online ordering only; it is never a CharCNN input.

## Fixed experimental protocol

| Split | Date range | Use |
| --- | --- | --- |
| Train | 1 Jul 2021 - 31 Aug 2022 | Train CharCNN on raw URL characters |
| Validation | 1 Nov 2022 - 30 Nov 2022 | Select checkpoint, calibrate threshold, MMD reference |
| Test | 1 Dec 2022 - 28 Feb 2023 | Frozen chronological online test |

September-October 2022 are deliberately skipped because the audit found no
benign URLs. The task retains `benign` and `phishing` only; `mal` is removed.
The data stage fails rather than altering these dates if either binary class is
absent from a required split. URLs are globally deduplicated before label
filtering and splitting, retaining the earliest observation, so no URL can
appear in multiple splits.

## Repository layout

```text
configs/config.yaml        # fixed dates, paths, seeds, hyperparameters
src/charcnn/               # data, vocab, model, train, evaluate, drift, RL
notebooks/                 # 01_data_audit, 02_train, 03_frozen_test, 04_drift_rl
data_manifest/             # audit, manifests, monthly counts (Kaggle outputs)
outputs/                   # metrics, matrices, drift logs (Kaggle outputs)
figures/                   # loss and drift figures (Kaggle outputs)
models/                    # small metadata only; model weights live in Kaggle output
```

## Kaggle workflow

1. Attach this repository as a Kaggle Dataset, enable Internet, and open
   `notebooks/01_data_audit_and_split.ipynb`.
2. Set `PROJECT_INPUT` to the attached dataset directory. The notebook copies
   code to `/kaggle/working/charcnn` and writes all generated data below that
   directory.
3. If the audit succeeds, publish `/kaggle/working/charcnn/data` as a Kaggle
   Dataset. Later notebooks will read that prepared data; they do not download
   or re-split DeepURLBench.

4. Attach that prepared Dataset and the updated project Dataset to
   `notebooks/02_train_charcnn.ipynb`, enable a T4 GPU, and run all cells.
   Its output directory is `/kaggle/working/charcnn/artifacts`. Publish that
   directory as another private Kaggle Dataset for the frozen test stage.

5. Attach the project, prepared-data, and training-artifacts Datasets to
   `notebooks/03_frozen_test.ipynb`. Keep Internet and GPU off. It writes
   frozen-test baseline metrics, a chronological error stream, and sampled
   monthly CNN embeddings for the drift/RL stage.

The data loader uses `davanstrien/DeepURLBench`, `urls_without_dns`. Do not
resolve or visit dataset URLs.
