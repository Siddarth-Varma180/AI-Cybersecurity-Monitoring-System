"""Tiny file-based model registry: models/vN/{model.joblib, metadata.json} + registry.json.

The same registry is reused later by the Flask API and the adaptive retraining job:
    - register_model()  saves a new version (does NOT activate it)
    - promote()         makes a version the one used for predictions
    - load_active()     returns (pipeline, metadata) for the active version
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib

import config



def _registry_file():
    return config.MODELS_DIR / "registry.json"


def _read_registry() -> dict:
    if _registry_file().exists():
        return json.loads(_registry_file().read_text())
    return {"active": None, "versions": []}


def _write_registry(reg: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _registry_file().write_text(json.dumps(reg, indent=2))


def next_version_id() -> str:
    return f"v{len(_read_registry()['versions']) + 1}"


def register_model(pipeline, metadata: dict, parent: str | None = None, trigger: str = "initial") -> str:
    version = next_version_id()
    vdir = config.MODELS_DIR / version
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, vdir / "model.joblib", compress=3)

    metadata = {
        **metadata,
        "version": version,
        "parent_version": parent,
        "trigger": trigger,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (vdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    reg = _read_registry()
    reg["versions"].append(version)
    _write_registry(reg)
    return version


def promote(version: str) -> None:
    reg = _read_registry()
    if version not in reg["versions"]:
        raise ValueError(f"Unknown version {version}")
    reg["active"] = version
    _write_registry(reg)


def get_metadata(version: str) -> dict:
    return json.loads((config.MODELS_DIR / version / "metadata.json").read_text())


def load_version(version: str):
    return joblib.load(config.MODELS_DIR / version / "model.joblib"), get_metadata(version)


def load_active():
    reg = _read_registry()
    if not reg["active"]:
        raise RuntimeError("No active model. Run scripts/02_train_models.py first.")
    return load_version(reg["active"])


def list_versions() -> list[dict]:
    return [get_metadata(v) for v in _read_registry()["versions"]]


def active_version() -> str | None:
    return _read_registry()["active"]


def has_versions() -> bool:
    return bool(_read_registry()["versions"])
