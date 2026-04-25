from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class FusionConfig:
    use_meta_classifier: bool = False


class ScoreFusion:
    """
    Combine outputs from:
    - LSTM (behavioral risk)
    - Transformer (temporal anomaly)
    - GNN (graph / cluster risk)
    - NLP (content risk)

    into a final:
    - fake / real label
    - risk score in [0, 100]
    """

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()
        self.meta_clf: LogisticRegression | None = None
        if self.config.use_meta_classifier:
            self._fit_demo_meta_classifier()

    def _fit_demo_meta_classifier(self) -> None:
        # Small synthetic training set for demonstration
        X = np.array(
            [
                [0.9, 0.8, 0.7, 0.9, 0.8],  # obvious fake
                [0.8, 0.7, 0.6, 0.8, 0.7],
                [0.1, 0.2, 0.1, 0.1, 0.2],  # obvious real
                [0.2, 0.1, 0.2, 0.2, 0.1],
                [0.6, 0.6, 0.4, 0.7, 0.5],  # borderline
                [0.3, 0.3, 0.4, 0.2, 0.3],
            ]
        )
        y = np.array([1, 1, 0, 0, 1, 0])
        self.meta_clf = LogisticRegression()
        self.meta_clf.fit(X, y)

    def fuse_scores(
        self,
        lstm: float,
        transformer: float,
        gnn: float,
        xgb: float,
        content: float = 0.0,  # optional for backward compatibility
    ) -> Dict[str, float | str]:
        """
        All inputs are assumed to be in [0, 1].
        """
        features = np.array([[lstm, transformer, gnn, xgb, content]])

        if self.meta_clf is not None:
            prob_fake = float(self.meta_clf.predict_proba(features)[0, 1])
        else:
            # Simple weighted ensemble
            weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # equal weights for lstm, transformer, gnn, xgb, content
            prob_fake = float(np.clip((features @ weights.reshape(-1, 1))[0, 0], 0.0, 1.0))

        risk_score_0_100 = round(prob_fake * 100, 1)
        label = "Fake" if prob_fake >= 0.5 else "Real"

        return {
            "label": label,
            "prob_fake": prob_fake,
            "risk_score": risk_score_0_100,
            "lstm": lstm,
            "transformer": transformer,
            "gnn": gnn,
            "xgb": xgb,
            "content": content,
        }

