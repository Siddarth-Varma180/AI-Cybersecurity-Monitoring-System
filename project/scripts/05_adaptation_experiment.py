"""Step 7 - Does adaptive learning actually help?  (offline experiment, no web app needed)

Simulates the live loop several times over:
    stream of unseen flows -> current model predicts -> admin reviews a limited budget of them
    -> validated labels -> controlled retrain + promotion gate -> next round with the (maybe) new model

Reports, after every round, the CURRENT model's metrics on a final evaluation set that is used
for nothing else (separate from the gate set that decides promotion).

    python scripts/05_adaptation_experiment.py --rounds 5 --review-budget 300 --strategy uncertain
    python scripts/05_adaptation_experiment.py --strategy random       # compare review strategies

Outputs reports/adaptation_<strategy>.csv and .png. Whatever the numbers show - improvement,
no change, or rejection by the gate - is the honest result to report.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import adaptive
from src import data_loader as dl
from src import evaluation as ev
from src import model_registry as reg
from src import training


def pick_for_review(model, X, y_true, budget, strategy, rng):
    """Which flows does the (simulated) admin look at? The admin supplies the true label."""
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    pred = model.classes_[proba.argmax(axis=1)]
    if strategy == "random":
        idx = rng.choice(len(X), min(budget, len(X)), replace=False)
    elif strategy == "uncertain":                 # what the dashboard's review queue shows first
        idx = np.argsort(conf)[:budget]
    elif strategy == "errors":                    # upper bound: admin only sees real mistakes
        wrong = np.where(pred != y_true.to_numpy())[0]
        idx = rng.choice(wrong, min(budget, len(wrong)), replace=False) if len(wrong) else np.array([], int)
    else:
        raise SystemExit("unknown strategy")
    return X.iloc[idx], y_true.iloc[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--stream-size", type=int, default=20000, help="flows arriving per round")
    ap.add_argument("--review-budget", type=int, default=300, help="flows the admin labels per round")
    ap.add_argument("--strategy", default="uncertain", choices=["uncertain", "random", "errors"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    parent, meta = reg.load_active()
    feats = meta["features"]
    lab = config.LABEL_COL
    hold = dl.load_holdout()
    gate_all, stream = dl.split_holdout(hold)                         # gate/eval half vs stream half
    gate, final = dl.split_holdout(gate_all, seed=11)                 # gate decides, final only reports
    base = training.get_base_train(meta)
    stream = stream.sample(frac=1, random_state=args.seed)
    print(f"base {len(base):,} | gate {len(gate):,} | final-eval {len(final):,} | stream {len(stream):,} "
          f"| strategy={args.strategy} budget={args.review_budget}/round")

    Xf, yf = final[feats], final[lab]
    m0, _ = ev.evaluate(parent, Xf, yf)
    rows = [dict(round=0, feedback_total=0, decision="baseline", version=meta["version"], **{k: m0[k] for k in
             ("macro_f1", "detection_rate", "false_positive_rate", "accuracy")}, val_acc_parent=None, val_acc_cand=None, reasons="")]
    print(f"round 0 | baseline {meta['version']}: macro-F1 {m0['macro_f1']:.4f}  FPR {m0['false_positive_rate']:.5f}")

    current = parent
    prev_X = prev_y = None            # labels already learned from (replayed in later rounds)
    pend_X = pend_y = None            # labels waiting for a successful update (mirrors feedback state 'new')
    chunk = len(stream) // args.rounds if args.stream_size * args.rounds > len(stream) else args.stream_size
    for r in range(1, args.rounds + 1):
        batch = stream.iloc[(r - 1) * chunk: r * chunk]
        Xr, yr = pick_for_review(current, batch[feats], batch[lab], args.review_budget, args.strategy, rng)
        pend_X = Xr if pend_X is None else pd.concat([pend_X, Xr])
        pend_y = yr if pend_y is None else pd.concat([pend_y, yr])
        res = adaptive.run_adaptation(current, base, gate, pend_X, pend_y, prev_X, prev_y, seed=args.seed + r)
        if res["promote"]:
            current = res["candidate"]
            prev_X = pend_X if prev_X is None else pd.concat([prev_X, pend_X])
            prev_y = pend_y if prev_y is None else pd.concat([prev_y, pend_y])
            pend_X = pend_y = None     # a rejected round keeps its labels pending, like the live system
        m, _ = ev.evaluate(current, Xf, yf)
        rows.append(dict(round=r, feedback_total=int(0 if prev_X is None else len(prev_X)),
                         decision="promoted" if res["promote"] else "rejected", version=f"after-round-{r}",
                         **{k: m[k] for k in ("macro_f1", "detection_rate", "false_positive_rate", "accuracy")},
                         val_acc_parent=res["val_accuracy_parent"], val_acc_cand=res["val_accuracy_candidate"],
                         reasons="; ".join(res["reasons"])))
        print(f"round {r} | {rows[-1]['decision']:8s} | final-eval macro-F1 {m['macro_f1']:.4f}  "
              f"FPR {m['false_positive_rate']:.5f}  DR {m['detection_rate']:.4f} | {rows[-1]['reasons']}")

    df = pd.DataFrame(rows)
    csv = config.REPORTS_DIR / f"adaptation_{args.strategy}.csv"
    df.to_csv(csv, index=False)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, col, title in zip(ax, ("macro_f1", "false_positive_rate", "detection_rate"),
                             ("Macro-F1 (higher is better)", "False-positive rate (lower)", "Attack detection rate")):
        a.plot(df["round"], df[col], marker="o")
        for _, row in df[df.decision == "rejected"].iterrows():
            a.scatter(row["round"], row[col], marker="x", color="red", zorder=5)
        a.set_title(title); a.set_xlabel("feedback round"); a.grid(alpha=.3)
    fig.suptitle(f"Effect of feedback-driven updates ({args.strategy} review, {args.review_budget} labels/round; red x = rejected by gate)")
    fig.tight_layout(); fig.savefig(csv.with_suffix(".png"), dpi=130)
    print(f"\nSaved {csv} and {csv.with_suffix('.png')}")


if __name__ == "__main__":
    main()
