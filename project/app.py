"""Flask application: prediction API + alerts + feedback + adaptive retraining + dashboard.

    python scripts/04_init_db.py      # once (creates tables)
    python app.py                     # http://127.0.0.1:5000   (login: see config / .env)
"""
from __future__ import annotations
from src import repository as repo
from src.live_monitor import LiveMonitor

import hmac
import json
import random
import threading
import time
import traceback
from functools import wraps

import numpy as np
import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request, session, url_for)

import config
from src import adaptive
from src import data_loader as dl
from src import db
from src import evaluation as ev
from src import model_registry as reg
from src import repository as repo

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
MAX_BATCH = 1000

_lock = threading.Lock()
_state: dict = {"model": None, "meta": None, "sim": None}
_retrain = {"running": False, "last": None}
_retrain_lock = threading.Lock()

live_monitor = LiveMonitor()


# ============================================================================ model handling
def load_model() -> None:
    model, meta = reg.load_active()
    ev.use_single_thread(model)
    with _lock:
        _state.update(model=model, meta=meta)


def _to_frame(flows: list[dict]) -> tuple[pd.DataFrame | None, str | None]:
    """Build a DataFrame in the exact training column order; unknown keys are ignored,
    missing/inf values become NaN (the model's imputer fills them)."""
    feats = _state["meta"]["features"]
    df = pd.DataFrame(flows)
    df.columns = [str(c).strip() for c in df.columns]
    present = [f for f in feats if f in df.columns]
    if len(present) < 0.5 * len(feats):
        return None, f"Only {len(present)}/{len(feats)} required features supplied"
    df = df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).astype("float32"), None


def _predict(df: pd.DataFrame) -> list[dict]:
    with _lock:
        model, meta = _state["model"], _state["meta"]
    t0 = time.perf_counter()
    proba = model.predict_proba(df)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    classes = model.classes_
    out = []
    for p in proba:
        order = np.argsort(p)[::-1]
        label = str(classes[order[0]])
        out.append({
            "label": label,
            "is_attack": not bool(dl.is_benign([label])[0]),
            "confidence": round(float(p[order[0]]), 4),
            "top3": [{"label": str(classes[i]), "probability": round(float(p[i]), 4)} for i in order[:3]],
            "model_version": meta["version"],
            "latency_ms": round(elapsed_ms / len(df), 3),
            "needs_review": bool(p[order[0]] < config.REVIEW_CONFIDENCE_THRESHOLD),
        })
    return out


def _features_json(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return [json.dumps({c: (None if v != v else round(float(v), 6)) for c, v in zip(cols, row)})
            for row in df.to_numpy()]


def _log(results: list[dict], df: pd.DataFrame, source: str, sim_true: list | None = None):
    """Persist predictions (+alerts). A DB outage must not break detection, so failures are
    reported in the response instead of raising."""
    try:
        fj = _features_json(df)
        ids = repo.log_predictions([
            {"result": r, "features_json": fj[i], "source": source,
             "sim_true": sim_true[i] if sim_true else None} for i, r in enumerate(results)])
        for r, (pid, aid) in zip(results, ids):
            r["prediction_id"], r["alert_id"] = pid, aid
        return True
    except Exception:
        traceback.print_exc()
        return False


def process_live_flows(flows: list[dict]) -> dict:
    """Classify and log live network flows."""

    if not flows:
        return {"processed": 0, "rejected": 0, "logged": False}

    from src.live_features import COLUMN_MAP

    features = _state["meta"]["features"]

    df = pd.DataFrame(flows)
    df.columns = [str(c).strip() for c in df.columns]

    # Map PyCICFlowMeter column names to model feature names.
    df = df.rename(columns={
        source: target
        for target, source in COLUMN_MAP.items()
        if source in df.columns
    })

    missing = [f for f in features if f not in df.columns]

    if missing:
        raise ValueError(f"Missing live features: {missing}")

    df = df.reindex(columns=features).apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    valid = df.notna().all(axis=1)

    for column in ("Flow Bytes/s", "Flow Packets/s"):
        valid &= df[column] >= 0

    rejected = int((~valid).sum())
    df = df.loc[valid].astype("float32")

    if df.empty:
        return {
            "processed": 0,
            "rejected": rejected,
            "logged": False
        }

    results = _predict(df)
    logged = _log(results, df, "live")

    return {
        "processed": len(results),
        "rejected": rejected,
        "logged": logged
    }



def process_live_flows(flows: list[dict]) -> dict:
    """Classify and log live network flows."""

    if not flows:
        return {"processed": 0, "rejected": 0, "logged": False}

    from src.live_features import COLUMN_MAP

    features = _state["meta"]["features"]

    df = pd.DataFrame(flows)
    df.columns = [str(c).strip() for c in df.columns]

    # Map PyCICFlowMeter column names to model feature names.
    df = df.rename(columns={
        source: target
        for target, source in COLUMN_MAP.items()
        if source in df.columns
    })

    missing = [f for f in features if f not in df.columns]

    if missing:
        raise ValueError(f"Missing live features: {missing}")

    df = df.reindex(columns=features).apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    valid = df.notna().all(axis=1)

    for column in ("Flow Bytes/s", "Flow Packets/s"):
        valid &= df[column] >= 0

    rejected = int((~valid).sum())
    df = df.loc[valid].astype("float32")

    if df.empty:
        return {
            "processed": 0,
            "rejected": rejected,
            "logged": False
        }

    results = _predict(df)
    logged = _log(results, df, "live")

    return {
        "processed": len(results),
        "rejected": rejected,
        "logged": logged
    }


def process_live_flows(flows: list[dict]) -> dict:
    """Classify and log live network flows."""

    if not flows:
        return {"processed": 0, "rejected": 0, "logged": False}

    from src.live_features import COLUMN_MAP

    features = _state["meta"]["features"]

    df = pd.DataFrame(flows)
    df.columns = [str(c).strip() for c in df.columns]

    # Map PyCICFlowMeter column names to model feature names.
    df = df.rename(columns={
        source: target
        for target, source in COLUMN_MAP.items()
        if source in df.columns
    })

    missing = [f for f in features if f not in df.columns]

    if missing:
        raise ValueError(f"Missing live features: {missing}")

    df = df.reindex(columns=features).apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    valid = df.notna().all(axis=1)

    for column in ("Flow Bytes/s", "Flow Packets/s"):
        valid &= df[column] >= 0

    rejected = int((~valid).sum())
    df = df.loc[valid].astype("float32")

    if df.empty:
        return {
            "processed": 0,
            "rejected": rejected,
            "logged": False
        }

    results = _predict(df)
    logged = _log(results, df, "live")

    return {
        "processed": len(results),
        "rejected": rejected,
        "logged": logged
    }

def process_live_flows(flows: list[dict]) -> dict:
    """Classify and log live network flows."""

    if not flows:
        return {"processed": 0, "rejected": 0, "logged": False}

    from src.live_features import COLUMN_MAP

    features = _state["meta"]["features"]

    df = pd.DataFrame(flows)
    df.columns = [str(c).strip() for c in df.columns]

    # Map PyCICFlowMeter column names to model feature names.
    df = df.rename(columns={
        source: target
        for target, source in COLUMN_MAP.items()
        if source in df.columns
    })

    missing = [f for f in features if f not in df.columns]

    if missing:
        raise ValueError(f"Missing live features: {missing}")

    df = df.reindex(columns=features).apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    valid = df.notna().all(axis=1)

    for column in ("Flow Bytes/s", "Flow Packets/s"):
        valid &= df[column] >= 0

    rejected = int((~valid).sum())
    df = df.loc[valid].astype("float32")

    if df.empty:
        return {
            "processed": 0,
            "rejected": rejected,
            "logged": False
        }

    results = _predict(df)
    logged = _log(results, df, "live")

    return {
        "processed": len(results),
        "rejected": rejected,
        "logged": logged
    }
live_monitor.on_flows = process_live_flows



# ============================================================================ auth
def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if not session.get("user"):
            if request.path.startswith("/api/"):
                return jsonify(error="Authentication required"), 401
            return redirect(url_for("login", next=request.path))
        return f(*a, **k)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u, p = request.form.get("username", ""), request.form.get("password", "")
        if (hmac.compare_digest(u, config.ADMIN_USER) and hmac.compare_digest(p, config.ADMIN_PASSWORD)):
            session["user"] = u
            return redirect(request.args.get("next") or url_for("dashboard"))
        error = "Invalid username or password"
    return render_template("login.html", error=error), (401 if error else 200)


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session["user"])

# ============================================================================ live monitoring

LIVE_INTERFACE = r"\Device\NPF_{3A0D4B56-DCC4-4F3A-87F3-BD7FCF5A26A8}"


@app.get("/api/live/status")
@login_required
def live_status():
    return jsonify(live_monitor.status())


@app.post("/api/live/start")
@login_required
def live_start():
    body = request.get_json(silent=True) or {}
    interface = body.get("interface", LIVE_INTERFACE)

    if interface != LIVE_INTERFACE:
        return jsonify(error="Unapproved network interface"), 403

    ok, message = live_monitor.start(interface)

    return jsonify(status=message, **live_monitor.status()), (
        200 if ok else 409
    )


@app.post("/api/live/stop")
@login_required
def live_stop():
    ok, message = live_monitor.stop()

    return jsonify(status=message, **live_monitor.status()), (
        200 if ok else 409
    )

# ============================================================================ open API (sensors / tests)
@app.get("/api/health")
def health():
    ok = _state["model"] is not None
    return jsonify(status="ok" if ok else "no_model", db=config.DB_BACKEND,
                   model_version=_state["meta"]["version"] if ok else None), (200 if ok else 503)


@app.get("/api/model")
def model_info():
    return jsonify(_state["meta"])


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    flow = body.get("features")
    if not isinstance(flow, dict):
        return jsonify(error='Body must be {"features": {name: value, ...}}'), 400
    df, err = _to_frame([flow])
    if err:
        return jsonify(error=err), 422
    res = _predict(df)
    res[0]["logged"] = _log(res, df, "api") if body.get("log", True) else False
    return jsonify(res[0])


@app.post("/api/predict/batch")
def predict_batch():
    body = request.get_json(silent=True) or {}
    flows = body.get("flows")
    if not isinstance(flows, list) or not flows:
        return jsonify(error='Body must be {"flows": [{...}, ...]}'), 400
    if len(flows) > MAX_BATCH:
        return jsonify(error=f"Max {MAX_BATCH} flows per request"), 413
    df, err = _to_frame(flows)
    if err:
        return jsonify(error=err), 422
    results = _predict(df)
    logged = _log(results, df, "api") if body.get("log", True) else False
    return jsonify(count=len(results), attacks=sum(r["is_attack"] for r in results),
                   logged=logged, results=results)


# ============================================================================ simulator
def _sim_pools():
    """Benign/attack pools drawn from the STREAM half of the hold-out (disjoint from the gate)."""
    if _state["sim"] is None:
        _, stream = dl.split_holdout(dl.load_holdout())
        mask = dl.is_benign(stream[config.LABEL_COL])
        _state["sim"] = (stream[mask], stream[~mask])
    return _state["sim"]


@app.get("/api/simulate")
def simulate_preview():
    """Random held-out flow + the model's verdict (nothing is logged)."""
    try:
        benign, attack = _sim_pools()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 404
    use_attack = request.args.get("attack") == "1" or (request.args.get("attack") != "0" and random.random() < 0.5)
    pool = attack if (use_attack and len(attack)) else benign
    row = pool.sample(1).iloc[0]
    flow = {f: (None if pd.isna(row[f]) else float(row[f])) for f in _state["meta"]["features"]}
    df, _ = _to_frame([flow])
    return jsonify(features=flow, true_label=str(row[config.LABEL_COL]), prediction=_predict(df)[0])


@app.post("/api/simulate/tick")
@login_required
def simulate_tick():
    """Generate one simulated flow, classify it and log it (called repeatedly by the dashboard)."""
    body = request.get_json(silent=True) or {}
    ratio = min(max(float(body.get("attack_ratio", 0.3)), 0.0), 1.0)
    try:
        benign, attack = _sim_pools()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 404
    pool = attack if (random.random() < ratio and len(attack)) else benign
    row = pool.sample(1).iloc[0]
    flow = {f: (None if pd.isna(row[f]) else float(row[f])) for f in _state["meta"]["features"]}
    df, _ = _to_frame([flow])
    res = _predict(df)
    ok = _log(res, df, "simulator", [str(row[config.LABEL_COL])])
    return jsonify({**res[0], "logged": ok})


# ============================================================================ dashboard data
@app.get("/api/classes")
@login_required
def classes():
    return jsonify(_state["meta"]["classes"])


@app.get("/api/stats")
@login_required
def stats():
    s = repo.stats()
    s["model"] = {"version": _state["meta"]["version"], "algorithm": _state["meta"]["algorithm"]}
    return jsonify(s)


@app.get("/api/predictions")
@login_required
def predictions():
    view = request.args.get("view", "all")
    if view not in ("all", "queue", "alerts"):
        return jsonify(error="view must be all|queue|alerts"), 400
    return jsonify(repo.list_predictions(view, min(int(request.args.get("limit", 50)), 500),
                                         request.args.get("label")))


@app.get("/api/predictions/<int:pid>")
@login_required
def prediction_detail(pid):
    p = repo.get_prediction(pid)
    return (jsonify(p), 200) if p else (jsonify(error="Not found"), 404)


@app.get("/api/alerts")
@login_required
def alerts():
    return jsonify(repo.list_alerts(request.args.get("status"), min(int(request.args.get("limit", 100)), 500)))


@app.post("/api/alerts/<int:aid>")
@login_required
def alert_update(aid):
    status = (request.get_json(silent=True) or {}).get("status", "")
    try:
        ok = repo.update_alert(aid, status, session["user"])
    except ValueError as e:
        return jsonify(error=str(e)), 422
    return (jsonify(status="updated"), 200) if ok else (jsonify(error="Alert not found"), 404)


# ============================================================================ feedback
@app.post("/api/feedback")
@login_required
def feedback():
    b = request.get_json(silent=True) or {}
    try:
        out = repo.submit_feedback(int(b.get("prediction_id", 0)), session["user"], b.get("verdict", ""),
                                   b.get("final_label"), (b.get("notes") or "")[:1000],
                                   _state["meta"]["classes"])
    except repo.FeedbackError as e:
        return jsonify(error=str(e)), e.code
    out["pending_feedback"] = repo.count_feedback("new")
    return jsonify(out)


@app.get("/api/feedback/summary")
@login_required
def feedback_summary():
    return jsonify({**repo.feedback_summary(), "min_required": config.MIN_FEEDBACK_TO_RETRAIN})


# ============================================================================ models + adaptation
@app.get("/api/models")
@login_required
def models():
    return jsonify(repo.list_model_versions())


@app.post("/api/models/<version>/promote")
@login_required
def promote(version):
    status = repo.get_model_status(version)
    if status is None:
        return jsonify(error="Unknown version"), 404
    if status == "rejected" and not (request.get_json(silent=True) or {}).get("force"):
        return jsonify(error="This version failed the promotion gate; pass force=true to override"), 409
    reg.promote(version)
    repo.set_active_in_db(version)
    load_model()
    return jsonify(status="activated", model_version=version)


def _retrain_job(user: str):
    try:
        result = adaptive.retrain_from_db(user)
        if result["promoted"]:
            load_model()
        _retrain["last"] = {"ok": True, **result}
    except Exception as e:
        traceback.print_exc()
        _retrain["last"] = {"ok": False, "error": str(e)}
    finally:
        _retrain["running"] = False
        _retrain_lock.release()


@app.post("/api/adaptive/retrain")
@login_required
def retrain():
    force = bool((request.get_json(silent=True) or {}).get("force"))
    pending = repo.count_feedback("new")
    if pending == 0:
        return jsonify(error="No new validated feedback yet", pending=pending), 409
    if pending < config.MIN_FEEDBACK_TO_RETRAIN and not force:
        return jsonify(error=f"Need at least {config.MIN_FEEDBACK_TO_RETRAIN} new feedback labels "
                             f"(have {pending}); use force to override", pending=pending), 409
    if not _retrain_lock.acquire(blocking=False):
        return jsonify(error="A retraining run is already in progress"), 409
    _retrain.update(running=True, last=None)
    threading.Thread(target=_retrain_job, args=(session["user"],), daemon=True).start()
    return jsonify(status="started", pending=pending), 202


@app.get("/api/adaptive/status")
@login_required
def retrain_status():
    return jsonify(running=_retrain["running"], last=_retrain["last"],
                   pending=repo.count_feedback("new"), min_required=config.MIN_FEEDBACK_TO_RETRAIN,
                   gate=config.GATE, feedback_weight=config.FEEDBACK_WEIGHT)


@app.get("/api/adaptive/runs")
@login_required
def retrain_runs():
    return jsonify(repo.list_runs())


@app.post("/api/model/reload")
@login_required
def reload_model():
    load_model()
    return jsonify(status="reloaded", model_version=_state["meta"]["version"])


# ============================================================================ startup
def init_app() -> None:
    db.init_db()
    load_model()
    adaptive.sync_registry_to_db()


if __name__ == "__main__":
    init_app()
    print(f"Loaded model {_state['meta']['version']} ({_state['meta']['algorithm']}) | DB: {config.DB_BACKEND}")
    if config.ADMIN_PASSWORD == "admin123":
        print("WARNING: default admin password in use - set AIDS_ADMIN_PASSWORD in .env")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
