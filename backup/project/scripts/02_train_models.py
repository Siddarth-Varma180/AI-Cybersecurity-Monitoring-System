"""Step 2 - Preprocess, train several models, compare, and register the best as v1.

    python scripts/02_train_models.py                       # full data (slow for big models)
    python scripts/02_train_models.py --max-train-per-class 100000    # recommended first run
    python scripts/02_train_models.py --models rf dt        # subset of models

Evaluation always uses the NATURAL (un-capped) test distribution so the false-positive
rate reflects a realistic benign:attack ratio.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # noqa: F401  (kept for reference)

import config
from src import data_loader as dl
from src import evaluation as ev
from src import model_registry as reg
from src import training
from src.training import build_models, cap_per_class

RS = config.RANDOM_STATE


def plot_confusion(cm: pd.DataFrame, title: str, path: Path):
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)   # row-normalised: per-class recall
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.index, fontsize=8)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm.values[i, j]:
                ax.text(j, i, f"{cm.values[i, j]:,}", ha="center", va="center", fontsize=6,
                        color="white" if norm.values[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, label="row-normalised")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--models", nargs="+", default=["dt", "rf", "hgb", "lr"], choices=["dt", "rf", "hgb", "lr"])
    ap.add_argument("--max-train-per-class", type=int, default=None,
                    help="cap training rows per class (test set is never capped)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-register", action="store_true", help="compare only; do not save/promote")
    args = ap.parse_args()

    config.REPORTS_DIR.mkdir(exist_ok=True)

    # 1. load + clean ----------------------------------------------------------------
    df = dl.clean(dl.load_dataset(Path(args.data_dir)))
    feats = dl.feature_columns(df)
    y = df[config.LABEL_COL]
    print(f"[data] {len(df):,} rows, {len(feats)} features, {y.nunique()} classes")

    # 2. split BEFORE any resampling so test data stays realistic ----------------------
    X_tr, X_te, y_tr, y_te = training.make_split(df, args.test_size, args.max_train_per_class)
    print(f"[split] train {len(X_tr):,} (capped={args.max_train_per_class})  test {len(X_te):,}")

    # 3. train + evaluate each model ----------------------------------------------------
    results, fitted = [], {}
    for key in args.models:
        name, model = build_models()[key]
        t0 = time.perf_counter(); model.fit(X_tr, y_tr); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); metrics, y_pred = ev.evaluate(model, X_te, y_te); pred_s = time.perf_counter() - t0
        metrics.update(model=name, key=key, train_seconds=round(fit_s, 1),
                       batch_predict_us_per_row=round(pred_s / len(X_te) * 1e6, 2),
                       single_row_latency_ms=round(ev.single_row_latency_ms(model, X_te), 2))
        results.append(metrics); fitted[key] = (model, y_pred)
        print(f"[{name:24s}] macro-F1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}  "
              f"DR={metrics['detection_rate']:.4f}  FPR={metrics['false_positive_rate']:.5f}  "
              f"train={fit_s:.0f}s  latency={metrics['single_row_latency_ms']}ms")

    table = pd.DataFrame(results).set_index("model")
    cols = ["macro_f1", "weighted_f1", "accuracy", "detection_rate", "false_positive_rate",
            "precision", "train_seconds", "single_row_latency_ms", "batch_predict_us_per_row"]
    table[cols].to_csv(config.REPORTS_DIR / "model_comparison.csv")
    print("\n", table[cols].round(4).to_string())

    # 4. pick best by macro-F1 (treats rare attack classes fairly, unlike accuracy) ------
    best_name = table["macro_f1"].idxmax()
    best_key = table.loc[best_name, "key"]
    best_model, best_pred = fitted[best_key]
    print(f"\n[best] {best_name} (selected on macro-F1)")

    labels = sorted(y.unique())
    cm = ev.confusion_df(y_te, best_pred, labels)
    cm.to_csv(config.REPORTS_DIR / "confusion_matrix_best.csv")
    plot_confusion(cm, f"{best_name} - confusion matrix (test)", config.REPORTS_DIR / "confusion_matrix_best.png")
    pc = ev.per_class_report(y_te, best_pred)
    pc.to_csv(config.REPORTS_DIR / "per_class_report_best.csv")
    print("\nPer-class report:\n", pc.round(3).to_string())

    # 5. register as v1 ---------------------------------------------------------------------
    if args.no_register:
        return
    metadata = {
        "algorithm": best_name,
        "features": feats,
        "classes": [str(c) for c in best_model.classes_],
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "max_train_per_class": args.max_train_per_class,
        "test_size": args.test_size,
        "metrics": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in table.loc[best_name].to_dict().items() if k not in ("key", "model")},
    }
    ev.use_single_thread(best_model)          # deployed configuration
    version = reg.register_model(best_model, metadata, parent=None, trigger="initial")
    reg.promote(version)
    # Keep the exact train/test sets: adaptation replays the training data and gates new
    # models on the untouched hold-out, so both must stay fixed.
    dl.write_table(pd.concat([X_te, y_te], axis=1), config.MODELS_DIR / "holdout_test")
    dl.write_table(pd.concat([X_tr, y_tr], axis=1), config.MODELS_DIR / "base_train")
    print(f"\n[registry] saved + activated {version}  ->  {config.MODELS_DIR / version}")


if __name__ == "__main__":
    main()
