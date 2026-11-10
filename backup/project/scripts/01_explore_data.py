"""Step 1 - Inspect the dataset: shapes, label distribution, NaN/inf, duplicates.

    python scripts/01_explore_data.py
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
from src import data_loader as dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    args = ap.parse_args()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    df = dl.load_dataset(Path(args.data_dir))
    feats = dl.feature_columns(df)
    print(f"\nLoaded {len(df):,} rows, {len(feats)} features, from {df[config.SOURCE_COL].nunique()} files\n")

    # --- per-file label breakdown -------------------------------------------------
    by_file = df.groupby([config.SOURCE_COL, config.LABEL_COL]).size().rename("rows").reset_index()
    print(by_file.to_string(index=False))
    by_file.to_csv(config.REPORTS_DIR / "label_by_file.csv", index=False)

    # --- overall class balance ----------------------------------------------------
    counts = df[config.LABEL_COL].value_counts()
    print("\nOverall class distribution:\n", counts.to_string())
    benign = dl.is_benign(df[config.LABEL_COL]).sum()
    print(f"\nBenign: {benign:,} ({benign/len(df):.1%})   Attack: {len(df)-benign:,} ({1-benign/len(df):.1%})")

    # --- data quality -------------------------------------------------------------
    num = df[feats].astype("float64")
    n_inf = int(np.isinf(num).sum().sum())
    n_nan = int(num.isna().sum().sum())
    n_dup = int(df.duplicated(subset=feats + [config.LABEL_COL]).sum())
    const_cols = [c for c in feats if df[c].nunique(dropna=True) <= 1]
    print(f"\ninf values: {n_inf:,}   NaN values: {n_nan:,}   duplicate rows: {n_dup:,}")
    print(f"Constant columns ({len(const_cols)}): {const_cols}")

    # --- plot ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.sort_values().plot.barh(ax=ax, logx=True)
    ax.set_xlabel("rows (log scale)")
    ax.set_title("CIC-IDS2017 class distribution")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "class_distribution.png", dpi=130)
    print(f"\nSaved plots/tables to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
