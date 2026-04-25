from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from difflib import SequenceMatcher


@dataclass
class CrossPlatformConfig:
    username_weight: float = 0.4
    behavior_weight: float = 0.3
    content_weight: float = 0.3


def username_similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(None, a.lower(), b.lower()).ratio())


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float((a_norm * b_norm).sum(axis=1).mean())


def cross_platform_similarity(
    username_a: str,
    username_b: str,
    behavior_vec_a: Iterable[float],
    behavior_vec_b: Iterable[float],
    content_vec_a: Iterable[float],
    content_vec_b: Iterable[float],
    config: CrossPlatformConfig | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a composite similarity between accounts on different platforms.

    Inputs:
    - username_*: raw handles
    - behavior_vec_*: aggregated behavioral features (e.g., posting rate buckets)
    - content_vec_*: aggregated content style features (e.g., topic or embedding means)
    """
    cfg = config or CrossPlatformConfig()
    u_sim = username_similarity(username_a, username_b)
    b_sim = cosine_sim(np.array(behavior_vec_a, dtype=float), np.array(behavior_vec_b, dtype=float))
    c_sim = cosine_sim(np.array(content_vec_a, dtype=float), np.array(content_vec_b, dtype=float))

    overall = (
        cfg.username_weight * u_sim
        + cfg.behavior_weight * b_sim
        + cfg.content_weight * c_sim
    )
    overall = float(max(0.0, min(1.0, overall)))

    details = {
        "username_similarity": u_sim,
        "behavior_similarity": b_sim,
        "content_similarity": c_sim,
    }
    return overall, details

