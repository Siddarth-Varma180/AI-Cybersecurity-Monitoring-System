"""Model recipes and the reproducible train/test split (shared by training and adaptation)."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from src import data_loader as dl

RS = config.RANDOM_STATE


def build_models(n_jobs: int = -1) -> dict:
    """Every model shares the same preprocessing: median-impute (handles the NaNs made from
    inf) then scale. Trees don't need scaling but it is harmless and keeps one recipe."""
    def pipe(clf):
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", clf)])
    return {
        "dt": ("Decision Tree", pipe(DecisionTreeClassifier(class_weight="balanced", random_state=RS))),
        "rf": ("Random Forest", pipe(RandomForestClassifier(n_estimators=100, n_jobs=n_jobs,
                                                            class_weight="balanced_subsample", random_state=RS))),
        "hgb": ("Hist Gradient Boosting", pipe(HistGradientBoostingClassifier(max_iter=100, class_weight="balanced",
                                                                              random_state=RS))),
        "lr": ("Logistic Regression", pipe(LogisticRegression(max_iter=300, class_weight="balanced"))),
    }


def cap_per_class(X, y, cap, seed=RS):
    if not cap:
        return X, y
    idx = (pd.Series(y).groupby(y.values, group_keys=False)
           .apply(lambda s: s.sample(min(len(s), cap), random_state=seed)).index)
    return X.loc[idx], y.loc[idx]


def make_split(df: pd.DataFrame, test_size: float = 0.2, cap: int | None = None):
    """Stratified split BEFORE any capping, so the test set keeps its natural class mix."""
    feats = dl.feature_columns(df)
    X, y = df[feats], df[config.LABEL_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RS)
    X_tr, y_tr = cap_per_class(X_tr, y_tr, cap)
    return X_tr, X_te, y_tr, y_te


def get_base_train(meta: dict) -> pd.DataFrame:
    """The original (capped) training set, used as replay data during adaptation.
    Cached at models/base_train.*; if missing (e.g. v1 trained before this existed) it is
    rebuilt from the raw data with the same seed and verified against the saved hold-out."""
    cached = dl.read_table(config.MODELS_DIR / "base_train")
    if cached is not None:
        return cached

    print("[adaptive] base_train not cached - rebuilding it from the raw dataset (one-off)...")
    df = dl.clean(dl.load_dataset(), verbose=False)
    X_tr, X_te, y_tr, y_te = make_split(df, meta.get("test_size", 0.2), meta.get("max_train_per_class"))
    if meta.get("test_rows") and len(X_te) != meta["test_rows"]:
        raise RuntimeError(
            f"Rebuilt split has {len(X_te)} test rows but the model was evaluated on {meta['test_rows']}. "
            "The raw data or library versions changed; re-run scripts/02_train_models.py.")
    try:
        hold = dl.load_holdout()
        if not isinstance(hold.index, pd.RangeIndex) and set(hold.index) != set(X_te.index):
            raise RuntimeError("Rebuilt test split does not match the saved hold-out. "
                               "Re-run scripts/02_train_models.py to regenerate both consistently.")
    except FileNotFoundError:
        pass
    if meta.get("train_rows") and len(X_tr) != meta["train_rows"]:
        raise RuntimeError(f"Rebuilt training set has {len(X_tr)} rows but the model was trained on "
                           f"{meta['train_rows']}. Re-run scripts/02_train_models.py.")
    base = pd.concat([X_tr, y_tr], axis=1)
    dl.write_table(base, config.MODELS_DIR / "base_train")
    return base
