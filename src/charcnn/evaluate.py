"""Binary metrics, threshold selection, and frozen-embedding export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


@torch.inference_mode()
def predict(model, loader, device: torch.device, collect_embeddings: bool = False):
    model.eval()
    labels, probabilities, embeddings = [], [], []
    for urls, targets in loader:
        urls = urls.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            result = model(urls, return_embedding=collect_embeddings)
        logits, embedding = result if collect_embeddings else (result, None)
        probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(targets.numpy())
        if embedding is not None:
            embeddings.append(embedding.float().cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities), (np.concatenate(embeddings) if embeddings else None)


def choose_threshold(y_true: np.ndarray, phishing_probabilities: np.ndarray, start: float, end: float, step: float) -> tuple[float, float]:
    choices = np.arange(start, end + step / 2, step)
    scores = [f1_score(y_true, phishing_probabilities >= threshold, average="macro", zero_division=0) for threshold in choices]
    best_index = int(np.argmax(scores))
    return float(choices[best_index]), float(scores[best_index])


def save_metrics(y_true: np.ndarray, phishing_probabilities: np.ndarray, threshold: float, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = (phishing_probabilities >= threshold).astype(np.int64)
    report = classification_report(y_true, predictions, labels=[0, 1], target_names=["benign", "phishing"], output_dict=True, zero_division=0)
    report["chosen_threshold"] = float(threshold)
    report["macro_f1"] = float(report["macro avg"]["f1-score"])
    (output / "validation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    with (output / "validation_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", "benign", "phishing"])
        writer.writerow(["benign", *matrix[0]])
        writer.writerow(["phishing", *matrix[1]])
    (output / "chosen_threshold.json").write_text(json.dumps({"threshold": float(threshold)}, indent=2), encoding="utf-8")
    return report
