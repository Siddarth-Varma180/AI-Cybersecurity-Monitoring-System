"""Create a small CIC-IDS2017-shaped CSV dataset (same 77 feature names) for quick tests/demos.

    python scripts/make_synthetic_data.py --out synthetic_data
    python scripts/02_train_models.py --data-dir synthetic_data --models rf

The classes are synthetic, so scores say nothing about real-world performance.
Two classes (XSS / Brute Force) deliberately overlap so there is something to learn from feedback.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ['Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Avg Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init Fwd Win Bytes', 'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
assert len(FEATURES) == 77

# class -> (sample count at scale 1, how far its mean sits from benign, which files it appears in)
CLASSES = {
    "Benign": (6000, 0.0), "DDoS": (1500, 2.5), "DoS Hulk": (1200, 2.0), "PortScan": (500, 1.6),
    "Bot": (300, 1.2), "Web Attack - Brute Force": (400, 1.0), "Web Attack - XSS": (300, 1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    direction = {c: rng.normal(0, 1, 77) for c in CLASSES}
    # make XSS a noisy neighbour of Brute Force
    direction["Web Attack - XSS"] = direction["Web Attack - Brute Force"] + rng.normal(0, 0.35, 77)

    frames = []
    for c, (n, sep) in CLASSES.items():
        n = max(int(n * a.scale), 20)
        mean = direction[c] / np.linalg.norm(direction[c]) * sep * np.sqrt(77) * 0.45
        X = rng.normal(mean, 1.0, (n, 77))
        df = pd.DataFrame(X, columns=FEATURES)
        df["Label"] = "Web Attack \ufffd " + c.split(" - ")[1] if c.startswith("Web Attack") else c
        frames.append(df)
    allx = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
    allx.loc[:25, "Flow Bytes/s"] = np.inf          # real data has inf / NaN / duplicates
    allx.loc[25:50, "Flow Packets/s"] = np.nan
    parts = [allx.iloc[ix] for ix in np.array_split(np.arange(len(allx)), 3)]
    for name, part in zip(["Benign-Monday", "DoS-Wednesday", "WebAttacks-Thursday"], parts):
        pd.concat([part, part.iloc[:100]]).to_csv(out / f"{name}.csv", index=False)
    print(f"Wrote {len(allx):,} rows (+duplicates) to {out}/")


if __name__ == "__main__":
    main()
