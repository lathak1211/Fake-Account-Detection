from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class BehaviorLSTMConfig:
    input_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2


class BehaviorLSTM(nn.Module):
    """
    Simple LSTM over sequential behavioral features, e.g.:
    - posting frequency
    - login timing
    - session gaps

    Exposes a single `forward` method that returns a risk score in [0, 1].
    """

    def __init__(self, config: BehaviorLSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            risk_score: Tensor of shape (batch_size,) in [0, 1]
        """
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # (batch_size, hidden_dim)
        risk = self.head(h_last).squeeze(-1)
        return risk


def demo_behavior_scores(
    batch_size: int = 4, seq_len: int = 50, input_dim: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility for generating demo risk scores on random behavioral sequences.
    Used by the Streamlit app when no trained model is available.
    """
    config = BehaviorLSTMConfig(input_dim=input_dim)
    model = BehaviorLSTM(config)
    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, input_dim)
        scores = model(x)
    return x, scores

