from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


LifecycleStage = Literal["creation", "warm-up", "attack", "dormant"]


@dataclass
class LifecycleConfig:
    warmup_days: int = 7
    attack_post_threshold: int = 20
    dormant_days: int = 14


def infer_lifecycle_stage(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    user_col: str = "user_id",
    event_type_col: str = "event_type",
    config: LifecycleConfig | None = None,
) -> pd.DataFrame:
    """
    Simple heuristic lifecycle estimator per user.

    Stages:
    - creation: account age < warmup_days and few posts
    - warm-up: age >= warmup_days but cumulative posts < attack_post_threshold
    - attack: short time window with burst of posts / interactions
    - dormant: no activity for > dormant_days
    """
    cfg = config or LifecycleConfig()
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    rows = []
    for user_id, group in df.groupby(user_col):
        group = group.sort_values(time_col)
        first = group[time_col].iloc[0]
        last = group[time_col].iloc[-1]
        age_days = (last - first).days + 1

        posts = group[group[event_type_col] == "post"]
        total_posts = len(posts)

        # dormant heuristic
        now = df[time_col].max()
        inactive_days = (now - last).days

        # burstiness: max posts per day
        if not posts.empty:
            daily_counts = posts.groupby(posts[time_col].dt.date).size()
            max_daily_posts = int(daily_counts.max())
        else:
            max_daily_posts = 0

        if inactive_days >= cfg.dormant_days:
            stage: LifecycleStage = "dormant"
        elif age_days <= cfg.warmup_days and total_posts < cfg.attack_post_threshold // 4:
            stage = "creation"
        elif max_daily_posts >= cfg.attack_post_threshold:
            stage = "attack"
        elif total_posts < cfg.attack_post_threshold:
            stage = "warm-up"
        else:
            stage = "attack"

        rows.append(
            {
                "user_id": user_id,
                "age_days": age_days,
                "total_posts": int(total_posts),
                "inactive_days": int(inactive_days),
                "max_daily_posts": int(max_daily_posts),
                "stage": stage,
            }
        )

    return pd.DataFrame(rows)

