"""All database reads/writes for predictions, alerts, feedback, model versions and retrain runs."""
from __future__ import annotations

import json

import config
from src import db
from src.data_loader import is_benign


def severity_for(label: str) -> str:
    return config.SEVERITY.get(label, config.DEFAULT_SEVERITY)


def _int(x) -> int:
    return int(x or 0)


# ============================================================================ predictions
def log_predictions(items: list[dict]) -> list[tuple[int, int | None]]:
    """items: dicts with keys result, features_json, source, sim_true. One transaction.
    Returns [(prediction_id, alert_id_or_None), ...]."""
    out = []
    with db.transaction() as tx:
        for it in items:
            r = it["result"]
            ts = db.now_iso()
            pid = tx.execute(
                "INSERT INTO predictions (ts, model_version, predicted_label, is_attack, confidence,"
                " latency_ms, source, sim_true_label, top3_json, features_json, review_state)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (ts, r["model_version"], r["label"], int(r["is_attack"]), r["confidence"],
                 r["latency_ms"], it.get("source", "api"), it.get("sim_true"),
                 json.dumps(r["top3"]), it["features_json"], "unreviewed"))
            aid = None
            if r["is_attack"] and r["confidence"] >= config.ALERT_MIN_CONFIDENCE:
                aid = tx.execute(
                    "INSERT INTO alerts (prediction_id, severity, status, created_at, updated_at)"
                    " VALUES (?,?,?,?,?)", (pid, severity_for(r["label"]), "open", ts, ts))
            out.append((pid, aid))
    return out


_PRED_COLS = ("p.id, p.ts, p.model_version, p.predicted_label, p.is_attack, p.confidence, p.latency_ms,"
              " p.source, p.review_state, a.id AS alert_id, a.status AS alert_status, a.severity")


def _clean_pred(r: dict) -> dict:
    r["is_attack"] = bool(r["is_attack"])
    r["confidence"] = float(r["confidence"]) if r["confidence"] is not None else None
    r["latency_ms"] = float(r["latency_ms"]) if r["latency_ms"] is not None else None
    return r


def list_predictions(view: str = "all", limit: int = 50, label: str | None = None) -> list[dict]:
    where, params = [], []
    if view == "queue":
        where.append("p.review_state = 'unreviewed' AND (p.is_attack = 1 OR p.confidence < ?)")
        params.append(config.REVIEW_CONFIDENCE_THRESHOLD)
    elif view == "alerts":
        where.append("a.status IN ('open','acknowledged')")
    if label:
        where.append("p.predicted_label = ?")
        params.append(label)
    order = "p.confidence ASC, p.id DESC" if view == "queue" else "p.id DESC"
    sql = (f"SELECT {_PRED_COLS} FROM predictions p LEFT JOIN alerts a ON a.prediction_id = p.id"
           + (" WHERE " + " AND ".join(where) if where else "")
           + f" ORDER BY {order} LIMIT ?")
    return [_clean_pred(r) for r in db.query(sql, (*params, int(limit)))]


def get_prediction(pid: int) -> dict | None:
    rows = db.query(
        f"SELECT {_PRED_COLS}, p.sim_true_label, p.top3_json FROM predictions p"
        " LEFT JOIN alerts a ON a.prediction_id = p.id WHERE p.id = ?", (pid,))
    if not rows:
        return None
    r = _clean_pred(rows[0])
    r["top3"] = json.loads(r.pop("top3_json") or "[]")
    fb = db.query("SELECT verdict, final_label, notes, reviewer, state FROM feedback WHERE prediction_id = ?", (pid,))
    r["feedback"] = fb[0] if fb else None
    return r


# ============================================================================ stats
def stats(minutes: int = 30) -> dict:
    tot = db.query("SELECT COUNT(*) AS n, SUM(is_attack) AS a, AVG(latency_ms) AS lat FROM predictions")[0]
    open_alerts = db.query("SELECT COUNT(*) AS n FROM alerts WHERE status IN ('open','acknowledged')")[0]["n"]
    reviewed = db.query("SELECT COUNT(*) AS n FROM predictions WHERE review_state = 'reviewed'")[0]["n"]
    by_label = db.query("SELECT predicted_label AS label, COUNT(*) AS n FROM predictions"
                        " GROUP BY predicted_label ORDER BY n DESC")
    from datetime import datetime, timedelta, timezone
    since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S")
    tl = db.query("SELECT SUBSTR(ts,1,16) AS m, is_attack, COUNT(*) AS n FROM predictions"
                  " WHERE ts >= ? GROUP BY SUBSTR(ts,1,16), is_attack ORDER BY m", (since,))
    series: dict[str, dict] = {}
    for r in tl:
        s = series.setdefault(r["m"], {"minute": r["m"], "benign": 0, "attack": 0})
        s["attack" if r["is_attack"] else "benign"] += _int(r["n"])
    return {
        "totals": {"flows": _int(tot["n"]), "attacks": _int(tot["a"]),
                   "avg_latency_ms": round(float(tot["lat"] or 0), 2),
                   "open_alerts": _int(open_alerts), "reviewed": _int(reviewed),
                   "pending_feedback": count_feedback("new")},
        "by_label": [{"label": r["label"], "count": _int(r["n"])} for r in by_label],
        "timeline": list(series.values()),
    }


# ============================================================================ alerts
ALERT_STATES = {"open", "acknowledged", "confirmed", "false_positive", "resolved"}


def list_alerts(status: str | None = None, limit: int = 100) -> list[dict]:
    sql = (f"SELECT {_PRED_COLS}, a.created_at FROM alerts a JOIN predictions p ON p.id = a.prediction_id"
           + (" WHERE a.status = ?" if status else "") + " ORDER BY a.id DESC LIMIT ?")
    params = (status, int(limit)) if status else (int(limit),)
    return [_clean_pred(r) for r in db.query(sql, params)]


def update_alert(alert_id: int, status: str, user: str) -> bool:
    if status not in ALERT_STATES:
        raise ValueError(f"status must be one of {sorted(ALERT_STATES)}")
    with db.transaction() as tx:
        if not tx.query("SELECT id FROM alerts WHERE id = ?", (alert_id,)):
            return False
        tx.execute("UPDATE alerts SET status = ?, updated_at = ?, handled_by = ? WHERE id = ?",
                   (status, db.now_iso(), user, alert_id))
    return True


# ============================================================================ feedback
class FeedbackError(Exception):
    def __init__(self, msg: str, code: int = 422):
        super().__init__(msg)
        self.code = code


def submit_feedback(prediction_id: int, reviewer: str, verdict: str, final_label: str | None,
                    notes: str, classes: list[str]) -> dict:
    with db.transaction() as tx:
        rows = tx.query("SELECT id, predicted_label, is_attack FROM predictions WHERE id = ?", (prediction_id,))
        if not rows:
            raise FeedbackError("Prediction not found", 404)
        pred = rows[0]
        if verdict == "correct":
            final_label = pred["predicted_label"]
        elif verdict == "incorrect":
            if final_label not in classes:
                raise FeedbackError(f"final_label must be one of the model's classes: {classes}")
            if final_label == pred["predicted_label"]:
                raise FeedbackError("Verdict 'incorrect' but final_label equals the prediction")
        else:
            raise FeedbackError("verdict must be 'correct' or 'incorrect'")

        existing = tx.query("SELECT id, state FROM feedback WHERE prediction_id = ?", (prediction_id,))
        now = db.now_iso()
        if existing:
            if existing[0]["state"] != "new":
                raise FeedbackError("This feedback was already used in a retraining run", 409)
            tx.execute("UPDATE feedback SET reviewer=?, verdict=?, final_label=?, notes=?, created_at=? WHERE id=?",
                       (reviewer, verdict, final_label, notes, now, existing[0]["id"]))
            fid = existing[0]["id"]
        else:
            fid = tx.execute(
                "INSERT INTO feedback (prediction_id, reviewer, verdict, original_label, final_label, notes,"
                " created_at, state) VALUES (?,?,?,?,?,?,?,?)",
                (prediction_id, reviewer, verdict, pred["predicted_label"], final_label, notes, now, "new"))
        tx.execute("UPDATE predictions SET review_state = 'reviewed' WHERE id = ?", (prediction_id,))

        # keep the alert in step with the admin's decision
        if tx.query("SELECT id FROM alerts WHERE prediction_id = ?", (prediction_id,)):
            new_status = "false_positive" if bool(is_benign([final_label])[0]) else "confirmed"
            tx.execute("UPDATE alerts SET status=?, updated_at=?, handled_by=? WHERE prediction_id=?",
                       (new_status, now, reviewer, prediction_id))
    return {"feedback_id": fid, "final_label": final_label}


def count_feedback(state: str | None = None) -> int:
    if state:
        return _int(db.query("SELECT COUNT(*) AS n FROM feedback WHERE state = ?", (state,))[0]["n"])
    return _int(db.query("SELECT COUNT(*) AS n FROM feedback")[0]["n"])


def feedback_summary() -> dict:
    rows = db.query("SELECT state, verdict, COUNT(*) AS n FROM feedback GROUP BY state, verdict")
    return {"rows": [{"state": r["state"], "verdict": r["verdict"], "count": _int(r["n"])} for r in rows],
            "pending": count_feedback("new"), "total": count_feedback()}


def fetch_feedback_rows(states: tuple[str, ...]) -> list[dict]:
    ph = ",".join("?" * len(states))
    return db.query(
        f"SELECT f.id, f.final_label, f.state, p.features_json FROM feedback f"
        f" JOIN predictions p ON p.id = f.prediction_id WHERE f.state IN ({ph}) ORDER BY f.id", tuple(states))


def mark_feedback(ids: list[int], state: str, version: str) -> None:
    if not ids:
        return
    with db.transaction() as tx:
        for i in ids:
            tx.execute("UPDATE feedback SET state = ?, used_in_version = ? WHERE id = ?", (state, version, int(i)))


# ============================================================================ model versions
def upsert_model_version(meta: dict, status: str) -> None:
    metrics = {"basis": meta.get("metrics_basis", "full hold-out test set"), **meta.get("metrics", {})}
    notes = json.dumps({k: meta[k] for k in ("decision", "reasons", "n_feedback_new") if k in meta})
    exists = db.query("SELECT version FROM model_versions WHERE version = ?", (meta["version"],))
    if exists:
        db.execute("UPDATE model_versions SET status = ?, metrics_json = ?, notes = ? WHERE version = ?",
                   (status, json.dumps(metrics), notes, meta["version"]))
    else:
        db.execute("INSERT INTO model_versions (version, algorithm, trigger_type, parent_version, status,"
                   " created_at, train_rows, metrics_json, notes) VALUES (?,?,?,?,?,?,?,?,?)",
                   (meta["version"], meta["algorithm"], meta.get("trigger"), meta.get("parent_version"),
                    status, meta.get("created_at"), meta.get("train_rows"), json.dumps(metrics), notes))


def set_active_in_db(version: str) -> None:
    with db.transaction() as tx:
        tx.execute("UPDATE model_versions SET status = 'retired' WHERE status = 'active'")
        tx.execute("UPDATE model_versions SET status = 'active' WHERE version = ?", (version,))


def list_model_versions() -> list[dict]:
    rows = db.query("SELECT * FROM model_versions ORDER BY created_at DESC, version DESC")
    for r in rows:
        r["metrics"] = json.loads(r.pop("metrics_json") or "{}")
        r["notes"] = json.loads(r["notes"]) if r.get("notes") else {}
    return rows


def get_model_status(version: str) -> str | None:
    rows = db.query("SELECT status FROM model_versions WHERE version = ?", (version,))
    return rows[0]["status"] if rows else None


# ============================================================================ retrain runs
def create_run(triggered_by: str, parent: str, n_new: int) -> int:
    return db.execute("INSERT INTO retrain_runs (started_at, triggered_by, parent_version, n_feedback_new, status)"
                      " VALUES (?,?,?,?,?)", (db.now_iso(), triggered_by, parent, n_new, "running"))


def finish_run(run_id: int, **f) -> None:
    cols = {"candidate_version": "candidate_version", "n_train_new": "n_train_new", "n_val": "n_val",
            "status": "status", "decision": "decision", "error": "error"}
    sets, params = ["finished_at = ?"], [db.now_iso()]
    for k, col in cols.items():
        if k in f:
            sets.append(f"{col} = ?"); params.append(f[k])
    for k in ("reasons", "metrics_parent", "metrics_candidate"):
        if k in f:
            sets.append(f"{k}_json = ?"); params.append(json.dumps(f[k]))
    db.execute(f"UPDATE retrain_runs SET {', '.join(sets)} WHERE id = ?", (*params, run_id))


def list_runs(limit: int = 20) -> list[dict]:
    rows = db.query("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (int(limit),))
    for r in rows:
        for k in ("reasons", "metrics_parent", "metrics_candidate"):
            raw = r.pop(f"{k}_json", None)
            r[k] = json.loads(raw) if raw else None
    return rows
