from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None


@dataclass
class NLPConfig:
    use_embeddings: bool = False  # BERT/Sentence-Transformer (optional)
    model_name: str = "all-MiniLM-L6-v2"


class ContentNLPEngine:
    """
    Lightweight NLP module for content credibility / spam detection.

    Default behavior:
    - Fit a TF-IDF + Logistic Regression on synthetic labels (for demo only)
    - Optionally use SentenceTransformer embeddings if installed
    """

    def __init__(self, config: NLPConfig | None = None):
        self.config = config or NLPConfig()
        self.vectorizer = None
        self.clf = None
        self.embedder = None

        if self.config.use_embeddings and HAS_SENTENCE_TRANSFORMERS:
            self.embedder = SentenceTransformer(self.config.model_name)

        # Train a tiny demo model on synthetic samples so we can output scores
        self._fit_demo_model()

    def _fit_demo_model(self) -> None:
        spam_examples = [
            "Win money now!!! click here",
            "Limited time offer, buy followers fast",
            "Free crypto airdrop visit our site",
            "Get rich quick with this trading bot",
        ]
        real_examples = [
            "Had a great day with friends today",
            "New blog post about deep learning and graphs",
            "Sharing some photos from my vacation",
            "Interesting article on social media safety",
        ]
        texts = spam_examples + real_examples
        y = np.array([1] * len(spam_examples) + [0] * len(real_examples))

        if self.embedder is None:
            self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X = self.vectorizer.fit_transform(texts)
        else:
            X = self.embedder.encode(texts)

        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X, y)

    def _encode(self, texts: List[str]):
        if self.embedder is not None:
            return self.embedder.encode(texts)
        return self.vectorizer.transform(texts)

    def content_risk_scores(self, texts: Iterable[str]) -> np.ndarray:
        """
        Args:
            texts: iterable of strings

        Returns:
            np.ndarray of shape (n_samples,) with scores in [0, 1]
        """
        texts = list(texts)
        if not texts:
            return np.array([])
        X = self._encode(texts)
        probs = self.clf.predict_proba(X)[:, 1]
        return probs

    def summarize_batch(self, texts: Iterable[str]) -> Tuple[float, float]:
        """
        Returns:
            mean_risk, max_risk
        """
        scores = self.content_risk_scores(texts)
        if scores.size == 0:
            return 0.0, 0.0
        return float(scores.mean()), float(scores.max())

