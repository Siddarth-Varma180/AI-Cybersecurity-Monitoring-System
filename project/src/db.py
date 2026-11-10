"""Minimal database layer supporting SQLite (default, zero setup) and MySQL (PyMySQL).

All SQL elsewhere in the project is written with `?` placeholders; they are converted to
`%s` for MySQL here. Timestamps are stored as UTC ISO strings so both engines behave alike.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

import config


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def is_mysql() -> bool:
    return config.DB_BACKEND == "mysql"


def _connect(with_db: bool = True):
    if is_mysql():
        import pymysql
        kw = dict(config.MYSQL)
        if not with_db:
            kw.pop("database")
        return pymysql.connect(**kw, charset="utf8mb4",
                               cursorclass=pymysql.cursors.DictCursor, autocommit=False)
    import sqlite3
    conn = sqlite3.connect(str(config.SQLITE_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _sql(sql: str) -> str:
    return sql.replace("?", "%s") if is_mysql() else sql


@contextmanager
def connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class _Tx:
    def __init__(self, conn):
        self.cur = conn.cursor()

    def execute(self, sql, params=()):
        self.cur.execute(_sql(sql), params)
        return self.cur.lastrowid

    def query(self, sql, params=()):
        self.cur.execute(_sql(sql), params)
        return [dict(r) for r in self.cur.fetchall()]


@contextmanager
def transaction():
    with connection() as conn:
        yield _Tx(conn)


def query(sql, params=()):
    with transaction() as tx:
        return tx.query(sql, params)


def execute(sql, params=()):
    with transaction() as tx:
        return tx.execute(sql, params)


# ----------------------------------------------------------------------------- schema
def _types() -> dict:
    if is_mysql():
        return dict(PK="BIGINT AUTO_INCREMENT PRIMARY KEY", V16="VARCHAR(16)", V32="VARCHAR(32)",
                    V64="VARCHAR(64)", TS="VARCHAR(24)", BIG="LONGTEXT",
                    ENGINE=" ENGINE=InnoDB DEFAULT CHARSET=utf8mb4")
    return dict(PK="INTEGER PRIMARY KEY AUTOINCREMENT", V16="TEXT", V32="TEXT", V64="TEXT",
                TS="TEXT", BIG="TEXT", ENGINE="")


_TABLES = [
    """CREATE TABLE IF NOT EXISTS model_versions (
        version {V32} PRIMARY KEY, algorithm {V64}, trigger_type {V32}, parent_version {V32},
        status {V16}, created_at {TS}, train_rows BIGINT, metrics_json {BIG}, notes {BIG}
    ){ENGINE}""",
    """CREATE TABLE IF NOT EXISTS predictions (
        id {PK}, ts {TS} NOT NULL, model_version {V32}, predicted_label {V64} NOT NULL,
        is_attack TINYINT NOT NULL, confidence DOUBLE, latency_ms DOUBLE, source {V16},
        sim_true_label {V64}, top3_json {BIG}, features_json {BIG},
        review_state {V16} NOT NULL DEFAULT 'unreviewed'
    ){ENGINE}""",
    """CREATE TABLE IF NOT EXISTS alerts (
        id {PK}, prediction_id BIGINT NOT NULL, severity {V16}, status {V16} NOT NULL DEFAULT 'open',
        created_at {TS}, updated_at {TS}, handled_by {V64}
    ){ENGINE}""",
    """CREATE TABLE IF NOT EXISTS feedback (
        id {PK}, prediction_id BIGINT NOT NULL, reviewer {V64}, verdict {V16} NOT NULL,
        original_label {V64}, final_label {V64} NOT NULL, notes {BIG}, created_at {TS},
        state {V16} NOT NULL DEFAULT 'new', used_in_version {V32}
    ){ENGINE}""",
    """CREATE TABLE IF NOT EXISTS retrain_runs (
        id {PK}, started_at {TS}, finished_at {TS}, triggered_by {V64}, parent_version {V32},
        candidate_version {V32}, n_feedback_new BIGINT, n_train_new BIGINT, n_val BIGINT,
        status {V16}, decision {V16}, reasons_json {BIG}, metrics_parent_json {BIG},
        metrics_candidate_json {BIG}, error {BIG}
    ){ENGINE}""",
]

_INDEXES = [
    ("idx_pred_ts", "predictions", "ts"),
    ("idx_pred_review", "predictions", "review_state"),
    ("idx_alert_status", "alerts", "status"),
    ("idx_alert_pred", "alerts", "prediction_id"),
    ("idx_fb_pred", "feedback", "prediction_id"),
    ("idx_fb_state", "feedback", "state"),
]


def ddl_statements() -> list[str]:
    t = _types()
    return [s.format(**t) for s in _TABLES]


def init_db() -> None:
    if is_mysql():
        conn = _connect(with_db=False)
        try:
            conn.cursor().execute(
                f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL['database']}` CHARACTER SET utf8mb4")
            conn.commit()
        finally:
            conn.close()
    with connection() as conn:
        cur = conn.cursor()
        for ddl in ddl_statements():
            cur.execute(ddl)
    for name, table, col in _INDEXES:                     # indexes: ignore "already exists"
        stmt = f"CREATE INDEX {'IF NOT EXISTS ' if not is_mysql() else ''}{name} ON {table}({col})"
        try:
            with connection() as conn:
                conn.cursor().execute(stmt)
        except Exception:
            pass
