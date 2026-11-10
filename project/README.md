# Adaptive AI Cybersecurity Monitoring System

Detects malicious network flows (CIC-IDS2017), shows them on an admin dashboard, and **learns from
validated administrator feedback** - updating the model in a controlled, evaluated, versioned way.

## Quick start (Windows PowerShell)
```powershell
cd project
python -m venv venv ; venv\Scripts\activate
pip install -r requirements.txt

# data: the 8 .parquet files go in data\
python scripts/01_explore_data.py                                   # Step 1
python scripts/02_train_models.py --max-train-per-class 100000      # Step 2 -> models\v1
python app.py                                                       # Steps 3-6 -> http://127.0.0.1:5000
```
Sign in with `admin` / `admin123` (change it: copy `.env.example` to `.env`, edit `AIDS_ADMIN_PASSWORD`).

## Using the dashboard
1. **Overview** - press *Start* on the traffic simulator. It replays *unseen* flows from the hold-out set
   through the live model; every flow is logged, attacks raise alerts.
2. **Alerts & Review** - open the review queue (unreviewed attacks + low-confidence flows, least
   confident first). Click *Review*, say whether the prediction is right, or pick the correct label.
   In simulation the true label is shown as a hint, standing in for the analyst's investigation.
3. **Models & Adaptation** - when enough validated labels exist, *Retrain now*. A candidate model is
   trained, evaluated against the current one, and either **promoted** or **rejected**. Any version can
   be re-activated (rollback).

## How adaptation works (`src/adaptive.py`)
1. New validated labels are split: ~75% used for training, ~25% held back as a *feedback-validation* set.
2. Candidate = same algorithm, fitted on **original training data + earlier feedback + new feedback**
   (feedback weighted x`FEEDBACK_WEIGHT`, so a few hundred labels are not drowned by ~300k rows).
3. Parent and candidate are evaluated on identical data: the **gate set** (half of the original hold-out,
   never trained on) and the feedback-validation set.
4. **Promote only if** no regression (macro-F1, detection rate and false-positive rate stay within
   the limits in `config.GATE`) **and** at least one measurable improvement (feedback-validation
   accuracy, macro-F1, or false-positive rate). Otherwise the candidate is stored as `rejected` and the
   labels stay pending for the next attempt. Improvement is tested, never assumed.

Every run (who, when, metrics before/after, decision, reasons) is stored in `retrain_runs`.

## Measuring it (Step 7)
```powershell
python scripts/05_adaptation_experiment.py --rounds 5 --review-budget 300 --strategy uncertain
python scripts/05_adaptation_experiment.py --rounds 5 --review-budget 300 --strategy random
python scripts/06_benchmark_api.py          # end-to-end latency (app.py must be running)
python -m unittest discover -s tests -v     # automated tests on synthetic data (~30 s)
```
The experiment reports metrics on a **final evaluation set separate from the gate set**, so the
numbers are not inflated by the promotion decision. Report whatever it shows, including "rejected".

## MySQL (optional; SQLite is the default)
1. Install MySQL (or XAMPP) and start it.
2. Copy `.env.example` to `.env`; set `AIDS_DB=mysql` and `MYSQL_USER` / `MYSQL_PASSWORD`.
3. `python scripts/04_init_db.py` creates the database and tables (or run `database/schema_mysql.sql`).
4. `python app.py`.

## Layout
```
app.py                    Flask app: login, prediction API, alerts, feedback, retraining, simulator
config.py  .env.example   settings (paths, DB, gate thresholds, severity map)
src/adaptive.py           feedback -> candidate -> evaluation -> promotion gate
src/repository.py src/db.py   all database access (SQLite / MySQL)
src/training.py           model recipes, reproducible split, replay data
src/model_registry.py     models/vN + active-version pointer
scripts/                  01 explore, 02 train, 03 API smoke test, 04 init DB, 05 experiment, 06 benchmark
templates/ static/        dashboard (Bootstrap + Chart.js)
tests/                    end-to-end tests
```

## Known limitations (state these in your report)
- Feedback can only use the model's **existing classes**; a genuinely new attack type needs a new
  class and a full retrain.
- The gate set is finite; very rare classes (Heartbleed, SQL injection: a handful of test samples)
  make macro-F1 noisy, so a promotion can be luck. Use the final-eval numbers from script 05.
- The simulator replays CIC-IDS2017 flows; it is not a live packet capture.
- Single shared admin account, no CSRF protection - fine for a project demo, not for production.
