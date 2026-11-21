"""Character-level CNN for binary benign/phishing classification."""

from __future__ import annotations

import torch
from torch import nn


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        char_embedding_dim: int,
        channels: list[int],
        kernel_sizes: list[int],
        dropout: float,
        embedding_dim: int,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must have equal lengths")
        self.character_embedding = nn.Embedding(vocab_size, char_embedding_dim, padding_idx=pad_id)
        blocks: list[nn.Module] = []
        in_channels = char_embedding_dim
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            blocks.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.embedding_head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(in_channels, embedding_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, character_ids: torch.Tensor, return_embedding: bool = False):
        x = self.character_embedding(character_ids).transpose(1, 2)
        embedding = self.embedding_head(self.features(x))
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits
