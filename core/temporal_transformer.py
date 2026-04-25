from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class TemporalTransformerConfig:
    input_dim: int = 5
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TemporalTransformer(nn.Module):
    """
    Lightweight Transformer encoder over long-term behavioral / temporal patterns.

    Produces a temporal anomaly score in [0, 1].
    """

    def __init__(self, config: TemporalTransformerConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            temporal_anomaly: Tensor of shape (batch_size,) in [0, 1]
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        score = self.head(pooled).squeeze(-1)
        return score

