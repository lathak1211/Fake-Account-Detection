from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch


def build_behavior_sequence(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    user_col: str = "user_id",
    event_type_col: str = "event_type",
    max_seq_len: int = 200,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Turn raw event logs into per-user sequences of behavioral features.

    Expected columns:
    - user_id
    - timestamp (datetime-like or numeric)
    - event_type (e.g. 'post', 'login', 'like')

    Returns:
        sequences: Tensor (num_users, seq_len, num_features)
        user_ids: list of user ids in the same order
    """
    # Handle datetime columns first to avoid np.issubdtype issues
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = df[time_col].astype("int64") // 10**9
    elif not np.issubdtype(df[time_col].dtype, np.number):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col]).astype("int64") // 10**9

    feature_vectors: List[np.ndarray] = []
    user_ids: List[str] = []

    for user_id, group in df.groupby(user_col):
        group = group.sort_values(time_col)
        times = group[time_col].values.astype(float)
        if len(times) < 2:
            inter_arrivals = np.array([0.0])
        else:
            inter_arrivals = np.diff(times, prepend=times[0])

        # Simple handcrafted features:
        # - inter-arrival time
        # - normalized hour of day
        # - binary flags per event_type (up to 3 main types)
        hours = (pd.to_datetime(group[time_col], unit="s").dt.hour.values.astype(float)) / 23.0

        event_types = group[event_type_col].astype(str).values
        unique_types = ["post", "login", "like"]
        type_features = np.zeros((len(event_types), len(unique_types)))
        for i, t in enumerate(event_types):
            if t in unique_types:
                type_features[i, unique_types.index(t)] = 1.0

        feats = np.stack([inter_arrivals, hours], axis=1)
        feats = np.concatenate([feats, type_features], axis=1)

        if len(feats) > max_seq_len:
            feats = feats[-max_seq_len:]
        else:
            pad_len = max_seq_len - len(feats)
            if pad_len > 0:
                feats = np.pad(feats, ((pad_len, 0), (0, 0)), mode="constant")

        feature_vectors.append(feats)
        user_ids.append(str(user_id))

    if not feature_vectors:
        return torch.zeros(0, max_seq_len, 5, dtype=torch.float16), []

    arr = np.stack(feature_vectors, axis=0)
    return torch.tensor(arr, dtype=torch.float16), user_ids

