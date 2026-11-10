"""End-to-end tests on synthetic data (no real dataset needed, ~1 minute).

    python -m unittest discover -s tests -v
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="aids_test_"))
os.environ.update(AIDS_WORKDIR=str(TMP), AIDS_DATA_DIR=str(TMP / "data"), AIDS_DB="sqlite",
                  AIDS_ADMIN_USER="admin", AIDS_ADMIN_PASSWORD="pw-for-tests")
sys.path.insert(0, str(ROOT))


def _run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=os.environ, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


class GateLogic(unittest.TestCase):
    """Unit tests for the promotion rule (no models needed)."""
    P = dict(macro_f1=0.80, detection_rate=0.99, false_positive_rate=0.005)

    def d(self, **cand):
        from src.adaptive import decide
        return decide(self.P, {**self.P, **cand}, None, None)

    def test_rejects_when_nothing_improves(self):
        ok, why = self.d()
        self.assertFalse(ok); self.assertIn("no measurable improvement", why[0])

    def test_promotes_on_macro_f1_gain(self):
        self.assertTrue(self.d(macro_f1=0.82)[0])

    def test_promotes_on_fpr_drop(self):
        self.assertTrue(self.d(false_positive_rate=0.004)[0])

    def test_regression_blocks_even_with_gain(self):
        ok, why = self.d(macro_f1=0.85, detection_rate=0.95)          # better F1 but misses attacks
        self.assertFalse(ok); self.assertTrue(any("detection rate fell" in w for w in why))

    def test_feedback_validation_improvement_counts(self):
        from src.adaptive import decide
        ok, why = decide(self.P, dict(self.P), 0.40, 0.60)
        self.assertTrue(ok); self.assertIn("feedback-validation", why[0])


class FullSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _run("scripts/make_synthetic_data.py", "--out", TMP / "data", "--scale", 2)
        _run("scripts/02_train_models.py", "--data-dir", TMP / "data", "--models", "rf",
             "--max-train-per-class", 2000)
        import app as appmod
        appmod.init_app()
        cls.app = appmod
        cls.c = appmod.app.test_client()

    def login(self):
        r = self.c.post("/login", data={"username": "admin", "password": "pw-for-tests"})
        self.assertEqual(r.status_code, 302)

    # -------------------------------------------------------------- auth
    def test_01_auth(self):
        self.assertEqual(self.c.get("/api/stats").status_code, 401)
        self.assertEqual(self.c.get("/").status_code, 302)
        self.assertEqual(self.c.post("/login", data={"username": "admin", "password": "bad"}).status_code, 401)
        self.login()
        self.assertEqual(self.c.get("/api/stats").status_code, 200)
        page = self.c.get("/")
        self.assertEqual(page.status_code, 200); self.assertIn(b"Adaptive AI", page.data)

    # -------------------------------------------------------------- prediction API
    def test_02_predict_validation_and_logging(self):
        self.assertEqual(self.c.post("/api/predict", json={}).status_code, 400)
        self.assertEqual(self.c.post("/api/predict", json={"features": {"Flow Duration": 1}}).status_code, 422)
        self.assertEqual(self.c.post("/api/predict/batch", json={"flows": [{}] * 1001}).status_code, 413)

        flow = self.c.get("/api/simulate?attack=1").json["features"]
        flow["Flow Bytes/s"] = float("inf")                           # inf must not crash inference
        r = self.c.post("/api/predict", json={"features": flow}).json
        self.assertTrue(r["logged"]); self.assertIn("prediction_id", r)
        self.assertEqual(r["model_version"], "v1")

        flows = [self.c.get("/api/simulate").json["features"] for _ in range(15)]
        b = self.c.post("/api/predict/batch", json={"flows": flows}).json
        self.assertEqual(b["count"], 15); self.assertTrue(b["logged"])
        self.assertEqual(sum(1 for x in b["results"] if x["alert_id"]), sum(1 for x in b["results"] if x["is_attack"] and x["confidence"] >= 0.5))
        self.assertFalse(self.c.post("/api/predict", json={"features": flow, "log": False}).json["logged"])

    # -------------------------------------------------------------- alerts + feedback validation
    def test_03_feedback_rules(self):
        self.login()
        for _ in range(30):                                            # find an alerting flow
            t = self.c.post("/api/simulate/tick", json={"attack_ratio": 1.0}).json
            if t.get("alert_id"):
                break
        self.assertTrue(t.get("alert_id"), "simulator should raise at least one alert")
        pid, label = t["prediction_id"], t["label"]
        classes = self.c.get("/api/classes").json
        self.assertEqual(self.c.post("/api/feedback", json={"prediction_id": pid, "verdict": "incorrect", "final_label": "Nonsense"}).status_code, 422)
        self.assertEqual(self.c.post("/api/feedback", json={"prediction_id": pid, "verdict": "incorrect", "final_label": label}).status_code, 422)
        self.assertEqual(self.c.post("/api/feedback", json={"prediction_id": 99999999, "verdict": "correct"}).status_code, 404)
        r = self.c.post("/api/feedback", json={"prediction_id": pid, "verdict": "incorrect", "final_label": "Benign" if label != "Benign" else classes[1]})
        self.assertEqual(r.status_code, 200)
        alert = [a for a in self.c.get("/api/alerts").json if a["alert_id"] == t["alert_id"]][0]
        self.assertEqual(alert["alert_status"], "false_positive")
        again = self.c.post("/api/feedback", json={"prediction_id": pid, "verdict": "correct"})   # editing before training is allowed
        self.assertEqual(again.status_code, 200)

    # -------------------------------------------------------------- the adaptive loop
    def test_04_adaptive_cycle(self):
        self.login()
        # too little feedback -> refused unless forced
        for _ in range(150):
            self.c.post("/api/simulate/tick", json={"attack_ratio": 0.5})
        q = self.c.get("/api/predictions?view=all&limit=300").json
        n = 0
        for p in q:
            if p["review_state"] == "reviewed":
                continue
            truth = self.c.get(f"/api/predictions/{p['id']}").json["sim_true_label"]
            if truth is None:                                          # API-sourced flow: no ground truth to give
                continue
            body = ({"verdict": "correct"} if truth == p["predicted_label"]
                    else {"verdict": "incorrect", "final_label": truth})
            self.assertEqual(self.c.post("/api/feedback", json={"prediction_id": p["id"], **body}).status_code, 200)
            n += 1
        self.assertGreater(n, 60)
        self.assertEqual(self.c.post("/api/adaptive/retrain", json={}).status_code, 202)
        self.assertEqual(self.c.post("/api/adaptive/retrain", json={}).status_code, 409)   # already running

        for _ in range(240):                                           # wait up to ~2 min
            st = self.c.get("/api/adaptive/status").json
            if not st["running"]:
                break
            time.sleep(0.5)
        self.assertFalse(st["running"]); self.assertTrue(st["last"]["ok"], st["last"])
        last = st["last"]
        versions = {v["version"]: v for v in self.c.get("/api/models").json}
        self.assertIn(last["candidate_version"], versions)
        active = [v for v, row in versions.items() if row["status"] == "active"]
        self.assertEqual(len(active), 1)
        health = self.c.get("/api/health").json
        self.assertEqual(health["model_version"], active[0])           # live model == DB's active model

        if last["promoted"]:
            self.assertEqual(active[0], last["candidate_version"])
            self.assertEqual(self.c.get("/api/feedback/summary").json["pending"], 0)
        else:
            self.assertEqual(active[0], "v1")
            self.assertEqual(versions[last["candidate_version"]]["status"], "rejected")
            self.assertGreater(self.c.get("/api/feedback/summary").json["pending"], 0)   # kept for next try
            self.assertEqual(self.c.post(f"/api/models/{last['candidate_version']}/promote", json={}).status_code, 409)
        run = self.c.get("/api/adaptive/runs").json[0]
        self.assertEqual(run["status"], "completed"); self.assertTrue(run["reasons"])

    def test_05_rollback(self):
        self.login()
        r = self.c.post("/api/models/v1/promote", json={})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.c.get("/api/health").json["model_version"], "v1")
        self.assertEqual(self.c.post("/api/models/v999/promote", json={}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
