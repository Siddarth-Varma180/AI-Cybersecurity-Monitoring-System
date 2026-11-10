"""Smoke-test the running API (start `python app.py` in another terminal first).

    python scripts/03_test_api.py
"""
import json
import urllib.request

BASE = "http://127.0.0.1:5000"


def call(path, payload=None):
    req = urllib.request.Request(BASE + path, method="POST" if payload is not None else "GET",
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


print("health  :", call("/api/health"))
model = call("/api/model")
print("model   :", model["version"], model["algorithm"], "| classes:", model["classes"])

for attack in (False, True):
    s = call("/api/simulate?attack=" + ("1" if attack else "0"))
    p = s["prediction"]
    print(f"simulate (attack={attack}): true={s['true_label']!r}  predicted={p['label']!r} "
          f"conf={p['confidence']}  {p['latency_ms']} ms")

s = call("/api/simulate")
r = call("/api/predict", {"features": s["features"]})
print("predict :", r["label"], r["confidence"])
b = call("/api/predict/batch", {"flows": [call("/api/simulate")["features"] for _ in range(20)]})
print("batch   :", b["count"], "flows,", b["attacks"], "flagged as attacks")
