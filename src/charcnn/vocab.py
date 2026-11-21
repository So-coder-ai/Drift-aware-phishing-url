"""Train-only character vocabulary and PyTorch URL dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterator
import time

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def parquet_url_batches(path: str | Path, batch_size: int = 100_000) -> Iterator[list[str]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["url"]):
        yield [str(url) for url in batch.column(0).to_pylist()]


def build_vocab(train_path: str | Path) -> dict[str, int]:
    """Build a character vocabulary from training URLs only."""
    print("Building character vocabulary from training URLs...", flush=True)
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    rows = 0
    for batch_number, urls in enumerate(parquet_url_batches(train_path), start=1):
        for url in urls:
            counts.update(url)
        rows += len(urls)
        if batch_number % 10 == 0:
            print(f"  vocabulary: {rows:,} URLs scanned ({time.perf_counter() - started:.0f}s)", flush=True)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for character in sorted(counts):
        vocab[character] = len(vocab)
    print(f"Vocabulary complete: {len(vocab)} characters from {rows:,} URLs.", flush=True)
    return vocab


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    Path(path).write_text(json.dumps(vocab, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_vocab(path: str | Path) -> dict[str, int]:
    return {key: int(value) for key, value in json.loads(Path(path).read_text(encoding="utf-8")).items()}


class URLParquetDataset(Dataset):
    """Memory-mapped Parquet URL dataset; `date` is intentionally unused."""

    def __init__(self, path: str | Path, vocab: dict[str, int], label_map: dict[str, int], max_length: int, limit: int | None = None):
        self.path = str(path)
        self.vocab = vocab
        self.label_map = label_map
        self.max_length = max_length
        self.pad_id = vocab[PAD_TOKEN]
        table = pq.read_table(self.path, columns=["url", "label"])
        if limit is not None:
            table = table.slice(0, min(limit, table.num_rows))
        self.urls = table.column("url").to_pylist()
        self.labels = table.column("label").to_pylist()
        print(f"Loaded {len(self.urls):,} rows from {Path(path).name}.", flush=True)

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        url = str(self.urls[index])[: self.max_length]
        encoded = [self.vocab.get(character, self.vocab[UNK_TOKEN]) for character in url]
        encoded.extend([self.pad_id] * (self.max_length - len(encoded)))
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.label_map[self.labels[index]], dtype=torch.long)
