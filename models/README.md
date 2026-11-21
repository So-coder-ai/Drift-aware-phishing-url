# Model artifacts

Large model files are stored in the private Kaggle Dataset
`charcnn-training-artifacts`, not in Git.

Required artifacts for reproducible frozen testing are:

- `best_model.pt` - selected CharCNN weights from validation.
- `vocab.json` - character vocabulary built from training URLs only.
- `label_map.json` - `benign: 0`, `phishing: 1`.
- `config.yaml` - architecture, seed, and training configuration.
- `chosen_threshold.json` - fixed phishing decision threshold of `0.36`.
- `validation_mmd_reference_embeddings.npz` - November reference embeddings.

The selected checkpoint was epoch 7 of the full eight-epoch run. It must be
loaded in `eval()` mode and never updated in the frozen online test.
