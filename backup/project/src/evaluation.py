"""Metrics shared by initial training and (later) adaptive-update evaluation."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from src.data_loader import is_benign


def binary_metrics(y_true, y_pred) -> dict:
    """Benign-vs-attack view. 'Positive' = attack."""
    t = ~is_benign(y_true)
    p = ~is_benign(y_pred)
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    tn, fn = int((~t & ~p).sum()), int((t & ~p).sum())
    return {
        "detection_rate": tp / (tp + fn) if tp + fn else 0.0,      # recall on attacks
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,  # benign flagged as attack
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "attack_f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def evaluate(model, X, y, classes=None) -> dict:
    y_pred = model.predict(X)
    out = {
        "accuracy": accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, y_pred, average="macro", zero_division=0),
        **binary_metrics(y, y_pred),
    }
    return out, y_pred


def per_class_report(y, y_pred) -> pd.DataFrame:
    rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T
    return df.drop(index=["accuracy"], errors="ignore")


def confusion_df(y, y_pred, labels) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y, y_pred, labels=labels), index=labels, columns=labels)


def use_single_thread(model) -> None:
    """Forests parallelise across cores, which is slower than one thread for a single row."""
    if "clf__n_jobs" in model.get_params():
        model.set_params(clf__n_jobs=1)


def single_row_latency_ms(model, X: pd.DataFrame, n: int = 200) -> float:
    """Average time to classify ONE flow in the deployed (single-threaded) configuration."""
    prev = model.get_params().get("clf__n_jobs", "absent")
    use_single_thread(model)
    try:
        rows = X.sample(min(n, len(X)), random_state=0)
        start = time.perf_counter()
        for i in range(len(rows)):
            model.predict(rows.iloc[[i]])
        return (time.perf_counter() - start) / len(rows) * 1000
    finally:
        if prev != "absent":
            model.set_params(clf__n_jobs=prev)
