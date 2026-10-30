"""Verified model loading and framework-independent prediction service."""

import hashlib
import json
from pathlib import Path

import joblib

from ethereum_fraud.schema import FEATURES, LABELS, validate_records


class PredictionService:
    """Load one trusted deployment artifact and reuse it across requests."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.report = json.loads((model_dir / "metrics.json").read_text())
        path = model_dir / "pipeline.joblib"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != self.report["artifact_sha256"]
        ):
            raise ValueError("Model artifact integrity check failed.")
        self.model = joblib.load(path)  # Only trusted, reviewed repository artifacts.
        if list(self.model.feature_names_in_) != list(FEATURES):
            raise ValueError("Model feature contract does not match the application.")
        self.sample = json.loads((model_dir / "sample.json").read_text())

    def predict(self, records, threshold=None):
        """Return source categories and uncalibrated model class scores."""
        if threshold is None:
            threshold = self.report["threshold"]
        try:
            if isinstance(threshold, bool):
                raise ValueError
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Threshold must be a number between 0.01 and 0.99."
            ) from exc
        if not 0.01 <= threshold <= 0.99:
            raise ValueError("Threshold must be between 0.01 and 0.99.")
        frame = validate_records(records)
        scores = self.model.predict_proba(frame)
        positive = list(self.model.classes_).index("1")
        labels = ["1" if row[positive] >= threshold else "0" for row in scores]
        return [
            {
                "category": str(label),
                "threshold": threshold,
                "description": LABELS[str(label)],
                "scores": {
                    str(key): round(float(value), 6)
                    for key, value in zip(self.model.classes_, row)
                },
            }
            for label, row in zip(labels, scores)
        ]
