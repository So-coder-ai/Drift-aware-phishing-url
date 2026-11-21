"""Resumable Kaggle CharCNN training command."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .evaluate import choose_threshold, predict, save_metrics
from .model import CharCNN
from .vocab import URLParquetDataset, build_vocab, save_vocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_f1: float, config: dict) -> None:
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "best_validation_macro_f1": best_f1, "config": config}, path)


@torch.inference_mode()
def compute_loss(model, data_loader, criterion, device: torch.device) -> float:
    model.eval()
    total, rows = 0.0, 0
    for urls, labels in data_loader:
        urls, labels = urls.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(urls), labels)
        total += float(loss) * labels.size(0)
        rows += labels.size(0)
    return total / rows


def save_mmd_reference(embeddings: np.ndarray, labels: np.ndarray, max_rows: int, seed: int, artifact_dir: Path) -> None:
    """Save a deterministic, balanced September embedding reference sample."""
    rng = np.random.default_rng(seed)
    per_class = max_rows // 2
    indices = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        indices.extend(rng.choice(candidates, size=min(per_class, len(candidates)), replace=False).tolist())
    indices = np.asarray(sorted(indices))
    np.savez_compressed(artifact_dir / "validation_mmd_reference_embeddings.npz", embeddings=embeddings[indices], labels=labels[indices])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True, help="Attached prepared Kaggle Dataset directory")
    parser.add_argument("--resume", default=None, help="Optional epoch checkpoint path")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config["project"]["seed"]))
    data_dir, artifact_dir = Path(args.data_dir), Path(config["paths"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, artifact_dir / "config.yaml")
    label_map = json.loads((data_dir / "label_map.json").read_text(encoding="utf-8"))
    (artifact_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    train_file, validation_file = data_dir / "train.parquet", data_dir / "validation.parquet"
    vocab_path = artifact_dir / "vocab.json"
    vocab_exists = vocab_path.exists()
    vocab = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_exists else build_vocab(train_file)
    if vocab_exists:
        print(f"Using saved training vocabulary ({len(vocab)} characters).", flush=True)
    save_vocab(vocab, vocab_path)
    small = config["runtime"]["small_run"]
    limit = int(config["runtime"]["small_run_rows_per_split"]) if small else None
    model_cfg, train_cfg = config["model"], config["training"]
    train_set = URLParquetDataset(train_file, vocab, label_map, model_cfg["max_length"], limit)
    validation_set = URLParquetDataset(validation_file, vocab, label_map, model_cfg["max_length"], limit)
    workers = int(config["runtime"]["num_workers"])
    train_loader = loader(train_set, train_cfg["batch_size"], workers, True)
    validation_loader = loader(validation_set, train_cfg["batch_size"], workers, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}; train rows: {len(train_set):,}; validation rows: {len(validation_set):,}", flush=True)
    model = CharCNN(len(vocab), model_cfg["char_embedding_dim"], model_cfg["channels"], model_cfg["kernel_sizes"], model_cfg["dropout"], model_cfg["embedding_dim"]).to(device)
    counts = np.bincount([label_map[label] for label in train_set.labels], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["runtime"]["mixed_precision"])
    start_epoch, best_f1, stale_epochs = 1, -1.0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch, best_f1 = checkpoint["epoch"] + 1, checkpoint["best_validation_macro_f1"]
    log_path = artifact_dir / "epoch_metrics.csv"
    write_header = not log_path.exists() or start_epoch == 1
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "validation_loss", "validation_macro_f1", "threshold"])
        if write_header: writer.writeheader()
        for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
            print(f"Epoch {epoch}/{train_cfg['epochs']} started.", flush=True)
            epoch_started = time.perf_counter()
            model.train(); total_loss = 0.0
            for batch_index, (urls, labels) in enumerate(train_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                urls, labels = urls.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and config["runtime"]["mixed_precision"]):
                    loss = criterion(model(urls), labels)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                total_loss += float(loss.detach()) * labels.size(0)
                if batch_index % 500 == 0:
                    print(f"  epoch {epoch}: batch {batch_index:,}/{len(train_loader):,}, loss={float(loss):.4f}, elapsed={time.perf_counter() - epoch_started:.0f}s", flush=True)
            targets, probabilities, _ = predict(model, validation_loader, device)
            validation_loss = compute_loss(model, validation_loader, criterion, device)
            threshold, validation_f1 = choose_threshold(targets, probabilities, train_cfg["threshold_grid_start"], train_cfg["threshold_grid_end"], train_cfg["threshold_grid_step"])
            save_checkpoint(artifact_dir / f"checkpoint_epoch_{epoch}.pt", model, optimizer, epoch, best_f1, config)
            writer.writerow({"epoch": epoch, "train_loss": total_loss / len(train_set), "validation_loss": validation_loss, "validation_macro_f1": validation_f1, "threshold": threshold}); log_file.flush()
            print(f"Epoch {epoch} complete: train_loss={total_loss / len(train_set):.4f}, validation_loss={validation_loss:.4f}, validation_macro_f1={validation_f1:.4f}, threshold={threshold:.2f}. Checkpoint saved.", flush=True)
            if validation_f1 > best_f1:
                best_f1, stale_epochs = validation_f1, 0
                torch.save({"model_state": model.state_dict(), "config": config}, artifact_dir / "best_model.pt")
                save_metrics(targets, probabilities, threshold, artifact_dir)
                print("  New best checkpoint saved as best_model.pt.", flush=True)
            else:
                stale_epochs += 1
                if stale_epochs >= int(train_cfg["early_stopping_patience"]): break
    print("Exporting validation MMD reference embeddings from the selected checkpoint...", flush=True)
    best = torch.load(artifact_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    targets, probabilities, embeddings = predict(model, validation_loader, device, collect_embeddings=True)
    chosen = json.loads((artifact_dir / "chosen_threshold.json").read_text(encoding="utf-8"))["threshold"]
    save_metrics(targets, probabilities, chosen, artifact_dir)
    save_mmd_reference(embeddings, targets, int(train_cfg["mmd_reference_max_rows"]), int(config["project"]["seed"]), artifact_dir)
    print("Training stage complete.", flush=True)
    print(json.dumps({"best_validation_macro_f1": best_f1, "artifacts": str(artifact_dir)}, indent=2))


if __name__ == "__main__":
    main()
