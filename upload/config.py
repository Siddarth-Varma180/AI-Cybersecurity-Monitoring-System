"""Central configuration for the Adaptive AI Cybersecurity Monitoring System.

Anything here can be overridden with environment variables or a `.env` file in the
project root (copy `.env.example` to `.env`).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# ----------------------------------------------------------------------------- paths
_WORK = Path(os.getenv("AIDS_WORKDIR", ROOT))            # where models/reports/db live
DATA_DIR = Path(os.getenv("AIDS_DATA_DIR", ROOT / "data"))  # the 8 CIC-IDS2017 files
MODELS_DIR = _WORK / "models"
REPORTS_DIR = _WORK / "reports"
SQLITE_PATH = _WORK / "ids.sqlite3"

# ----------------------------------------------------------------------------- dataset
LABEL_COL = "Label"
SOURCE_COL = "source_file"
BENIGN_LABELS = {"benign"}          # compared case-insensitively
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 10

# ----------------------------------------------------------------------------- database
# "sqlite" works out of the box. Set AIDS_DB=mysql (and the MYSQL_* values) to use MySQL.
DB_BACKEND = os.getenv("AIDS_DB", "sqlite").lower()
MYSQL = dict(
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_DB", "adaptive_ids"),
)

# ----------------------------------------------------------------------------- web app
SECRET_KEY = os.getenv("AIDS_SECRET", "dev-only-change-me")
ADMIN_USER = os.getenv("AIDS_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("AIDS_ADMIN_PASSWORD", "admin123")     # CHANGE THIS in .env

ALERT_MIN_CONFIDENCE = 0.5          # predicted attacks below this confidence raise no alert
REVIEW_CONFIDENCE_THRESHOLD = 0.8   # unreviewed predictions below this enter the review queue

# Severity is a policy choice (an assumption, not derived from data) - edit freely.
SEVERITY = {
    "Heartbleed": "Critical", "Infiltration": "Critical",
    "DDoS": "High", "DoS Hulk": "High", "DoS GoldenEye": "High", "Bot": "High",
    "Web Attack - Sql Injection": "High",
    "DoS Slowhttptest": "Medium", "DoS slowloris": "Medium", "FTP - Patator": "Medium",
    "SSH - Patator": "Medium", "Web Attack - Brute Force": "Medium", "Web Attack - XSS": "Medium",
    "PortScan": "Low",
}
DEFAULT_SEVERITY = "Medium"

# ----------------------------------------------------------------------------- adaptive learning
MIN_FEEDBACK_TO_RETRAIN = 50        # dashboard button needs at least this many new labels (or force)
FEEDBACK_WEIGHT = 5.0               # training weight of an admin-validated example vs. 1.0 for base data
FEEDBACK_VAL_FRACTION = 0.25        # share of NEW feedback held back to test the candidate

# A candidate model is promoted only if it passes every "no-regression" check on the
# gate hold-out AND shows at least one measurable improvement.
GATE = dict(
    max_macro_f1_drop=0.01,         # macro-F1 may not fall by more than this
    max_detection_drop=0.002,       # attack detection rate may not fall by more than this
    max_fpr_increase=0.001,         # false-positive rate may not rise by more than this
    min_macro_f1_gain=0.001,        # improvement criterion: macro-F1 up by at least this
    min_fpr_gain=0.0005,            # improvement criterion: FPR down by at least this
)
