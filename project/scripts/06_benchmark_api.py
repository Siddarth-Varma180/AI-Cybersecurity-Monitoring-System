"""Step 7 - Measure end-to-end prediction latency of the running API.

    python app.py                      # terminal 1
    python scripts/06_benchmark_api.py # terminal 2   (--n 300 for more samples)

Latency here includes HTTP + JSON + database logging, i.e. what a sensor would experience.
"""
import argparse
import json
import statistics
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:5000")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()


def call(path, payload=None):
    req = urllib.request.Request(a.base + path, method="POST" if payload is not None else "GET",
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


flows = [call("/api/simulate")["features"] for _ in range(min(a.n, 50))]
for log in (False, True):
    times = []
    for i in range(a.n):
        t0 = time.perf_counter()
        call("/api/predict", {"features": flows[i % len(flows)], "log": log})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    q = lambda p: times[min(len(times) - 1, int(p * len(times)))]
    print(f"single /api/predict (log={log!s:5}): mean {statistics.mean(times):6.1f} ms | "
          f"p50 {q(.5):6.1f} | p95 {q(.95):6.1f} | p99 {q(.99):6.1f}  (n={a.n})")

batch = flows * 20
t0 = time.perf_counter()
r = call("/api/predict/batch", {"flows": batch[:1000], "log": False})
dt = time.perf_counter() - t0
print(f"batch of {r['count']} flows: {dt*1000:.0f} ms total = {r['count']/dt:,.0f} flows/s")
