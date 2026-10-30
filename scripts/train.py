"""Reproducible historical fraud modeling with separated selection and test data."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/project.json").read_text())
SCHEMA = importlib.import_module(CONFIG["package"] + ".schema")
FEATURES = SCHEMA.FEATURES


def measure(model, features, labels, threshold):
    """Report minority-sensitive ranking and threshold-dependent metrics."""
    scores = model.predict_proba(features)[:, list(model.classes_).index("1")]
    predicted = np.where(scores >= threshold, "1", "0")
    return {
        "rows": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro"),
        "weighted_f1": f1_score(labels, predicted, average="weighted"),
        "average_precision": average_precision_score(labels == "1", scores),
        "roc_auc": roc_auc_score(labels == "1", scores),
        "classification_report": classification_report(
            labels, predicted, labels=["0", "1"], output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=["0", "1"]
        ).tolist(),
    }


def load_partitions(data_dir):
    """Use time boundaries for card data, feature groups for Ethereum data."""
    frame = pd.read_csv(data_dir / CONFIG["file"])
    original_rows = len(frame)
    if CONFIG["target"] == "Class":
        frame = frame.sort_values("Time", kind="stable")
        frame = frame.drop_duplicates(subset=list(FEATURES), keep="first").reset_index(
            drop=True
        )
        first, second = frame["Time"].quantile([0.6, 0.8])
        train = np.flatnonzero(frame["Time"] <= first)
        valid = np.flatnonzero((frame["Time"] > first) & (frame["Time"] <= second))
        test = np.flatnonzero(frame["Time"] > second)
        strategy = (
            "chronological 60/20/20 by Time; duplicate selected profiles keep earliest"
        )
        detail = {
            "time_boundaries_seconds": [float(first), float(second)],
            "duplicate_profiles_removed": original_rows - len(frame),
        }
    else:
        # Addresses, identifiers, token text and outcome are never model inputs.
        groups = pd.util.hash_pandas_object(
            frame[list(FEATURES)], index=False
        ).to_numpy()
        develop, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
                frame, groups=groups
            )
        )
        train_rel, valid_rel = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43).split(
                frame.iloc[develop], groups=groups[develop]
            )
        )
        train, valid = develop[train_rel], develop[valid_rel]
        strategy = (
            "grouped random 60/20/20; identical selected input vectors stay together"
        )
        detail = {}
    features = frame[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
    labels = frame[CONFIG["target"]].astype(str)
    partitions = {"train": train, "validation": valid, "test": test}
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    for name, indexes in partitions.items():
        if set(labels.iloc[indexes]) != {"0", "1"}:
            raise ValueError(f"Both classes must appear in {name}")
    for left, right in [(train, valid), (train, test), (valid, test)]:
        if not set(groups[left]).isdisjoint(groups[right]):
            raise ValueError("Input-profile overlap between partitions")
    return features, labels, partitions, strategy, detail, original_rows


def train_models(data_dir, output_dir):
    """Select by validation AP and choose threshold on validation fraud F1."""
    features, labels, partitions, strategy, detail, original_rows = load_partitions(
        data_dir
    )
    training, validation, test = (
        partitions[key] for key in ["train", "validation", "test"]
    )
    candidates = {
        "majority_baseline": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            DummyClassifier(strategy="most_frequent"),
        ),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=2,
                random_state=42,
            ),
        ),
    }
    comparison = {}
    for name, model in candidates.items():
        print("Fitting", name, flush=True)
        model.fit(features.iloc[training], labels.iloc[training])
        comparison[name] = measure(
            model, features.iloc[validation], labels.iloc[validation], 0.5
        )
    winner = max(comparison, key=lambda key: comparison[key]["average_precision"])
    model = candidates[winner]
    valid_scores = model.predict_proba(features.iloc[validation])[
        :, list(model.classes_).index("1")
    ]
    thresholds = np.arange(0.01, 1, 0.01)
    threshold = float(
        thresholds[
            np.argmax(
                [
                    f1_score(labels.iloc[validation] == "1", valid_scores >= value)
                    for value in thresholds
                ]
            )
        ]
    )
    print("Selected", winner, "threshold", threshold, flush=True)
    # Importance is a validation explanation; test remains for final reporting only.
    importance = permutation_importance(
        model,
        features.iloc[validation],
        labels.iloc[validation],
        scoring=make_scorer(
            average_precision_score, response_method="predict_proba", pos_label="1"
        ),
        n_repeats=3,
        random_state=42,
        n_jobs=2,
        max_samples=min(3000, len(validation)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pipeline.joblib"
    joblib.dump(model, path, compress=3)
    report = {
        "version": "1.0.0",
        "selected_model": winner,
        "threshold": threshold,
        "features": list(FEATURES),
        "classes": ["0", "1"],
        "original_rows": original_rows,
        "source_sha256": hashlib.sha256(
            (data_dir / CONFIG["file"]).read_bytes()
        ).hexdigest(),
        "source_url": CONFIG["url"],
        "split_strategy": strategy,
        "split_detail": detail,
        "split": {
            name: {
                "rows": len(idx),
                "class_counts": labels.iloc[idx].value_counts().to_dict(),
            }
            for name, idx in partitions.items()
        },
        "validation_candidates": comparison,
        "validation_selected_threshold": measure(
            model, features.iloc[validation], labels.iloc[validation], threshold
        ),
        "test": measure(model, features.iloc[test], labels.iloc[test], threshold),
        "permutation_importance": dict(
            zip(FEATURES, importance.importances_mean.tolist())
        ),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    sample = features.iloc[training].median().to_dict()
    sample = {
        key: int(value) if key in CONFIG["integer"] else value
        for key, value in sample.items()
    }
    (output_dir / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    print(json.dumps({"winner": winner, "test": report["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    train_models(args.data_dir, args.output_dir)
