"""Frozen, chronological CPU test of the selected CharCNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .model import CharCNN
from .vocab import URLParquetDataset


def make_model(config: dict, vocab_size: int) -> CharCNN:
    model_cfg = config["model"]
    return CharCNN(vocab_size, model_cfg["char_embedding_dim"], model_cfg["channels"], model_cfg["kernel_sizes"], model_cfg["dropout"], model_cfg["embedding_dim"])


def save_report(y_true: np.ndarray, y_pred: np.ndarray, output: Path, prefix: str) -> dict:
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["benign", "phishing"], output_dict=True, zero_division=0)
    report["macro_f1"] = float(report["macro avg"]["f1-score"])
    (output / f"{prefix}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with (output / f"{prefix}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", "benign", "phishing"])
        writer.writerow(["benign", *matrix[0]])
        writer.writerow(["phishing", *matrix[1]])
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/charcnn/frozen_test")
    args = parser.parse_args(argv)
    data_dir, artifact_dir, output = Path(args.data_dir), Path(args.artifact_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((artifact_dir / "config.yaml").read_text(encoding="utf-8"))
    label_map = json.loads((artifact_dir / "label_map.json").read_text(encoding="utf-8"))
    vocab = json.loads((artifact_dir / "vocab.json").read_text(encoding="utf-8"))
    threshold = json.loads((artifact_dir / "chosen_threshold.json").read_text(encoding="utf-8"))["threshold"]
    checkpoint = torch.load(artifact_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model = make_model(config, len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    test_path = data_dir / "test.parquet"
    meta = pq.read_table(test_path, columns=["date"])
    dates = np.asarray(meta.column("date").to_pylist(), dtype="datetime64[ns]")
    if np.any(dates[1:] < dates[:-1]):
        raise RuntimeError("test.parquet is not date-sorted; refusing to run online simulation.")
    dataset = URLParquetDataset(test_path, vocab, label_map, config["model"]["max_length"])
    workers = min(2, int(config["runtime"]["num_workers"]))
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=workers, pin_memory=False, persistent_workers=workers > 0)
    print(f"Frozen CPU test: {len(dataset):,} date-ordered URLs, threshold={threshold:.2f}", flush=True)

    probabilities, labels = [], []
    reservoir_limit = int(config["training"]["mmd_reference_max_rows"])
    reservoirs: dict[str, list[np.ndarray]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    rng = np.random.default_rng(int(config["project"]["seed"]))
    offset = 0
    with torch.inference_mode():
        for batch_index, (urls, targets) in enumerate(loader, start=1):
            logits, embeddings = model(urls, return_embedding=True)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            embedding_array = embeddings.numpy().astype(np.float32, copy=False)
            batch_dates = dates[offset : offset + len(targets)]
            for row, item_date in enumerate(batch_dates):
                month = str(item_date)[:7]
                seen[month] += 1
                bucket = reservoirs[month]
                if len(bucket) < reservoir_limit:
                    bucket.append(embedding_array[row].copy())
                else:
                    replacement = int(rng.integers(seen[month]))
                    if replacement < reservoir_limit:
                        bucket[replacement] = embedding_array[row].copy()
            offset += len(targets)
            probabilities.append(probs)
            labels.append(targets.numpy())
            if batch_index % 500 == 0:
                print(f"  processed {offset:,}/{len(dataset):,} URLs", flush=True)

    y_true, phishing_prob = np.concatenate(labels), np.concatenate(probabilities)
    y_pred = (phishing_prob >= threshold).astype(np.int64)
    overall = save_report(y_true, y_pred, output, "frozen_test")
    monthly_rows = []
    for month in sorted(reservoirs):
        mask = np.asarray([str(item)[:7] == month for item in dates])
        report = save_report(y_true[mask], y_pred[mask], output, f"frozen_test_{month.replace('-', '_')}")
        monthly_rows.append({"month": month, "rows": int(mask.sum()), "macro_f1": report["macro_f1"], "phishing_recall": report["phishing"]["recall"]})
    with (output / "frozen_test_monthly_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["month", "rows", "macro_f1", "phishing_recall"])
        writer.writeheader(); writer.writerows(monthly_rows)
    pq.write_table(pa.table({"date": meta.column("date"), "true_label": y_true, "phishing_probability": phishing_prob, "predicted_label": y_pred, "error": (y_true != y_pred).astype(np.int8)}), output / "frozen_test_error_stream.parquet", compression="zstd")
    np.savez_compressed(output / "frozen_test_monthly_embeddings.npz", **{f"month_{month.replace('-', '_')}": np.stack(values) for month, values in reservoirs.items()})
    (output / "frozen_test_run.json").write_text(json.dumps({"mode": "model.eval + torch.inference_mode", "device": "cpu", "threshold": threshold, "rows": int(len(dataset)), "macro_f1": overall["macro_f1"], "date_order_verified": True}, indent=2), encoding="utf-8")
    print(json.dumps({"macro_f1": overall["macro_f1"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
