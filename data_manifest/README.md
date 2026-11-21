# Prepared temporal split manifest

The raw DeepURLBench data is not committed because of its size. It was prepared
once in Kaggle and saved as a private Dataset. The preparation retained only
`url`, `first_seen` (renamed to `date`), and `label`, globally deduplicated URLs
by their earliest observation, and excluded the `mal` class.

| Split | Date range | Rows | Benign | Phishing |
| --- | --- | ---: | ---: | ---: |
| Train | 2021-07-01 to 2022-08-31 | 13,327,517 | 7,836,493 | 5,491,024 |
| Validation | 2022-11-01 to 2022-11-30 | 585,068 | 463,567 | 121,501 |
| Test | 2022-12-01 to 2023-02-28 | 1,428,259 | 1,162,990 | 265,269 |

September and October 2022 were deliberately skipped because the data audit
found no benign URLs in those months. `date` is used only to create the
chronological partitions and order the test stream; it is never passed to the
CharCNN.

The private prepared-data Dataset contains `train.parquet`,
`validation.parquet`, `test.parquet`, class-count CSV files, `audit.json`,
`manifest.json`, and `label_map.json`.

This folder includes the small audit, manifest, label map, and monthly count
files. The three large Parquet split files remain in the private Kaggle Dataset
and local backup, rather than GitHub.
