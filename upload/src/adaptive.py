"""Adaptive learning: controlled, evaluated, versioned model updates from admin feedback.

Procedure (run_adaptation):
  1. Split NEW validated feedback into a training part and a small validation part.
  2. Fit a candidate = clone(parent) on  base training data (replay)
                                       + earlier feedback
                                       + new training feedback   (feedback weighted > 1)
  3. Evaluate parent and candidate on identical data:
        - gate hold-out   : an unbiased set the candidate never saw -> regression check
        - feedback-val    : new labelled cases the candidate never trained on -> learning check
  4. Promote only if there is NO regression AND at least one measurable improvement.
Improvement is therefore *tested*, never assumed. Rejected candidates are still saved
(for audit) but never become the active model.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import config
from src import data_loader as dl
from src import evaluation as ev
from src import model_registry as reg
from src import repository as repo
from src import training


# ----------------------------------------------------------------------------- pure logic
def _plain(d: dict) -> dict:
    return {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v)
            for k, v in d.items()}


def split_feedback(X: pd.DataFrame, y: pd.Series, val_fraction: float, seed: int):
    """-> X_train, y_train, X_val, y_val. Tiny batches skip validation (everything trains)."""
    if len(X) < 8 or val_fraction <= 0:
        return X, y, X.iloc[0:0], y.iloc[0:0]
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_fraction, stratify=y, random_state=seed)
    except ValueError:                       # a class too rare / too many classes for the val size
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_fraction, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def decide(parent_gate: dict, cand_gate: dict, parent_val: float | None, cand_val: float | None,
           gate: dict | None = None) -> tuple[bool, list[str]]:
    """Promotion rule. Returns (promote?, human-readable reasons)."""
    g = gate or config.GATE
    problems, gains = [], []

    d_f1 = cand_gate["macro_f1"] - parent_gate["macro_f1"]
    d_dr = cand_gate["detection_rate"] - parent_gate["detection_rate"]
    d_fpr = cand_gate["false_positive_rate"] - parent_gate["false_positive_rate"]

    # --- no-regression checks
    if d_f1 < -g["max_macro_f1_drop"]:
        problems.append(f"macro-F1 fell {d_f1:+.4f} (allowed -{g['max_macro_f1_drop']})")
    if d_dr < -g["max_detection_drop"]:
        problems.append(f"detection rate fell {d_dr:+.4f} (allowed -{g['max_detection_drop']})")
    if d_fpr > g["max_fpr_increase"]:
        problems.append(f"false-positive rate rose {d_fpr:+.5f} (allowed +{g['max_fpr_increase']})")

    # --- improvement checks (need at least one)
    if parent_val is not None and cand_val is not None and cand_val > parent_val + 1e-12:
        gains.append(f"feedback-validation accuracy {parent_val:.3f} -> {cand_val:.3f}")
    if d_f1 >= g["min_macro_f1_gain"]:
        gains.append(f"gate macro-F1 {parent_gate['macro_f1']:.4f} -> {cand_gate['macro_f1']:.4f}")
    if -d_fpr >= g["min_fpr_gain"]:
        gains.append(f"false-positive rate {parent_gate['false_positive_rate']:.5f} -> "
                     f"{cand_gate['false_positive_rate']:.5f}")

    if problems:
        return False, problems
    if not gains:
        return False, ["no measurable improvement over the current model"]
    return True, gains


def run_adaptation(parent, base: pd.DataFrame, gate: pd.DataFrame, X_new: pd.DataFrame, y_new: pd.Series,
                   X_prev: pd.DataFrame | None = None, y_prev: pd.Series | None = None,
                   weight: float | None = None, val_fraction: float | None = None,
                   gate_cfg: dict | None = None, seed: int = config.RANDOM_STATE) -> dict:
    """DB-free core, also used by the offline experiment script."""
    weight = config.FEEDBACK_WEIGHT if weight is None else weight
    val_fraction = config.FEEDBACK_VAL_FRACTION if val_fraction is None else val_fraction
    feats = dl.feature_columns(base)
    lab = config.LABEL_COL

    X_tr_new, y_tr_new, X_val, y_val = split_feedback(X_new[feats], y_new, val_fraction, seed)

    parts_X, parts_y, parts_w = [base[feats]], [base[lab]], [np.ones(len(base))]
    if X_prev is not None and len(X_prev):
        parts_X.append(X_prev[feats]); parts_y.append(y_prev); parts_w.append(np.full(len(X_prev), weight))
    parts_X.append(X_tr_new); parts_y.append(y_tr_new); parts_w.append(np.full(len(X_tr_new), weight))
    X_fit = pd.concat(parts_X, ignore_index=True)
    y_fit = pd.concat(parts_y, ignore_index=True)
    w_fit = np.concatenate(parts_w)

    t0 = time.perf_counter()
    cand = clone(parent)
    if "clf__n_jobs" in cand.get_params():
        cand.set_params(clf__n_jobs=-1)               # parallel for training
    cand.fit(X_fit, y_fit, clf__sample_weight=w_fit)
    fit_s = time.perf_counter() - t0
    if "clf__n_jobs" in cand.get_params():
        cand.set_params(clf__n_jobs=1)                # fast single-flow inference

    Xg, yg = gate[feats], gate[lab]
    m_parent, _ = ev.evaluate(parent, Xg, yg)
    m_cand, _ = ev.evaluate(cand, Xg, yg)

    p_val = c_val = None
    if len(X_val):
        p_val = float((parent.predict(X_val) == y_val.to_numpy()).mean())
        c_val = float((cand.predict(X_val) == y_val.to_numpy()).mean())

    promote, reasons = decide(m_parent, m_cand, p_val, c_val, gate_cfg)
    return {
        "candidate": cand, "promote": promote, "reasons": reasons,
        "metrics_parent": _plain(m_parent), "metrics_candidate": _plain(m_cand),
        "val_accuracy_parent": p_val, "val_accuracy_candidate": c_val,
        "n_train_new": int(len(X_tr_new)), "n_val": int(len(X_val)),
        "train_ids": list(X_tr_new.index), "val_ids": list(X_val.index),
        "n_fit_rows": int(len(X_fit)), "fit_seconds": round(fit_s, 1),
    }


# ----------------------------------------------------------------------------- DB-backed job
_cache: dict = {}


def clear_cache() -> None:
    _cache.clear()


def _base_and_gate(meta: dict):
    if "base" not in _cache:
        _cache["base"] = training.get_base_train(meta)
    if "gate" not in _cache:
        _cache["gate"], _ = dl.split_holdout(dl.load_holdout())
    return _cache["base"], _cache["gate"]


def _frame(rows: list[dict], feats: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    data = [json.loads(r["features_json"]) for r in rows]
    X = pd.DataFrame(data, columns=feats).astype("float32")        # None -> NaN
    X.index = [r["id"] for r in rows]
    y = pd.Series([r["final_label"] for r in rows], index=X.index, name=config.LABEL_COL)
    return X, y


def retrain_from_db(triggered_by: str = "admin") -> dict:
    """Collect feedback -> retrain -> evaluate -> register -> (maybe) promote. Returns a summary."""
    parent_version = reg.active_version()
    parent, meta = reg.load_version(parent_version)
    feats = meta["features"]

    new_rows = repo.fetch_feedback_rows(("new",))
    prev_rows = repo.fetch_feedback_rows(("trained", "validation"))
    run_id = repo.create_run(triggered_by, parent_version, len(new_rows))
    try:
        if not new_rows:
            raise RuntimeError("No new validated feedback to learn from")
        X_new, y_new = _frame(new_rows, feats)
        X_prev, y_prev = _frame(prev_rows, feats) if prev_rows else (None, None)
        base, gate = _base_and_gate(meta)

        res = run_adaptation(parent, base, gate, X_new, y_new, X_prev, y_prev)
        cand = res["candidate"]

        cand_meta = {
            "algorithm": meta["algorithm"], "features": feats,
            "classes": [str(c) for c in cand.classes_],
            "train_rows": res["n_fit_rows"],
            "metrics": res["metrics_candidate"],
            "metrics_basis": "gate hold-out (50% of original test set)",
            "parent_gate_metrics": res["metrics_parent"],
            "decision": "promoted" if res["promote"] else "rejected",
            "reasons": res["reasons"],
            "n_feedback_new": len(new_rows),
            "feedback_weight": config.FEEDBACK_WEIGHT,
        }
        version = reg.register_model(cand, cand_meta, parent=parent_version, trigger="feedback")
        full_meta = reg.get_metadata(version)

        if res["promote"]:
            reg.promote(version)
            repo.upsert_model_version(full_meta, "retired")
            repo.set_active_in_db(version)
            repo.mark_feedback(res["train_ids"], "trained", version)
            repo.mark_feedback(res["val_ids"], "validation", version)
        else:
            repo.upsert_model_version(full_meta, "rejected")      # feedback stays 'new' for the next attempt

        repo.finish_run(run_id, candidate_version=version, n_train_new=res["n_train_new"], n_val=res["n_val"],
                        status="completed", decision=cand_meta["decision"], reasons=res["reasons"],
                        metrics_parent={**res["metrics_parent"], "val_accuracy": res["val_accuracy_parent"]},
                        metrics_candidate={**res["metrics_candidate"], "val_accuracy": res["val_accuracy_candidate"]})
        return {"run_id": run_id, "candidate_version": version, "decision": cand_meta["decision"],
                "reasons": res["reasons"], "promoted": res["promote"], "parent_version": parent_version,
                "fit_seconds": res["fit_seconds"]}
    except Exception as e:                                         # never leave a run 'running'
        repo.finish_run(run_id, status="failed", error=str(e))
        raise


def sync_registry_to_db() -> None:
    """Make sure every registry version has a row in model_versions (used at app start-up)."""
    active = reg.active_version()
    for meta in reg.list_versions():
        known = repo.get_model_status(meta["version"])
        if known is None:
            status = ("active" if meta["version"] == active
                      else "rejected" if meta.get("decision") == "rejected" else "retired")
            repo.upsert_model_version(meta, status)
    if active:
        repo.set_active_in_db(active)
