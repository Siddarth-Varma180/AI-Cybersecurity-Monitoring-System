"""Loading and cleaning CIC-IDS2017 (parquet or csv)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config


def list_data_files(data_dir: Path = config.DATA_DIR) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.parquet")) + sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No .parquet/.csv files found in {data_dir}. Extract the dataset archive there."
        )
    return files


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def normalise_label(s: pd.Series) -> pd.Series:
    """Strip whitespace and fix the odd encoding of 'Web Attack – ...' labels."""
    s = s.astype(str).str.strip()
    s = s.str.replace("\ufffd", "-", regex=False).str.replace("\u2013", "-", regex=False)
    s = s.str.replace(r"\s*-\s*", " - ", regex=True)
    return s


def is_benign(labels) -> np.ndarray:
    return pd.Series(labels).astype(str).str.strip().str.lower().isin(config.BENIGN_LABELS).to_numpy()


def load_dataset(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Load every file, tag each row with its source file, return one DataFrame."""
    frames = []
    for f in list_data_files(data_dir):
        df = _read(f)
        df.columns = [c.strip() for c in df.columns]
        df[config.LABEL_COL] = normalise_label(df[config.LABEL_COL])
        df[config.SOURCE_COL] = f.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (config.LABEL_COL, config.SOURCE_COL)]


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """inf -> NaN (imputed later inside the model pipeline), drop exact duplicates,
    drop classes too rare to split, downcast to float32 to save RAM."""
    feats = feature_columns(df)
    n0 = len(df)

    df = df.copy()
    df[feats] = df[feats].astype("float64").replace([np.inf, -np.inf], np.nan).astype("float32")

    # Duplicates inflate scores when the same row lands in both train and test.
    df = df.drop_duplicates(subset=feats + [config.LABEL_COL]).reset_index(drop=True)
    n1 = len(df)

    counts = df[config.LABEL_COL].value_counts()
    rare = counts[counts < config.MIN_SAMPLES_PER_CLASS].index.tolist()
    if rare:
        df = df[~df[config.LABEL_COL].isin(rare)].reset_index(drop=True)

    if verbose:
        print(f"[clean] rows: {n0:,} -> {n1:,} after dropping duplicates"
              f" -> {len(df):,} after dropping rare classes {rare or ''}")
    return df


# ----------------------------------------------------------------------------- tables on disk
def write_table(df: pd.DataFrame, stem: Path) -> Path:
    """Save as parquet if pyarrow is installed, otherwise csv.gz. `stem` has no extension."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        p = Path(str(stem) + ".parquet")
        df.to_parquet(p)
        return p
    except ImportError:
        p = Path(str(stem) + ".csv.gz")
        df.to_csv(p, index=False)
        return p


def _cast(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in df.columns if c not in (config.LABEL_COL, config.SOURCE_COL)]
    df[feats] = df[feats].astype("float32")
    return df


def read_table(stem: Path) -> pd.DataFrame | None:
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        p = Path(str(stem) + suffix)
        if p.exists():
            return _cast(reader(p))
    return None


def load_holdout() -> pd.DataFrame:
    """The original test set saved by training. Never used for fitting."""
    for stem in (config.MODELS_DIR / "holdout_test", config.MODELS_DIR / "v1" / "holdout_test"):
        df = read_table(stem)
        if df is not None:
            return df
    raise FileNotFoundError("No hold-out test set found. Run scripts/02_train_models.py first.")


def split_holdout(hold: pd.DataFrame, seed: int = 7, frac: float = 0.5):
    """Deterministic, disjoint halves of the hold-out set -> (gate, stream).
    gate   : used to decide whether a retrained model may be promoted.
    stream : source of simulated live traffic (so feedback never leaks into the gate)."""
    from sklearn.model_selection import train_test_split
    try:
        a, b = train_test_split(hold, test_size=frac, stratify=hold[config.LABEL_COL], random_state=seed)
    except ValueError:
        a, b = train_test_split(hold, test_size=frac, random_state=seed)
    return a, b
