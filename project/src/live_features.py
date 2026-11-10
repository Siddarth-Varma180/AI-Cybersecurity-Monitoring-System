
"""Convert PyCICFlowMeter CSV records to model feature format."""

import pandas as pd
import numpy as np


COLUMN_MAP = {
    "Fwd Packets Length Total": "Total Length of Fwd Packets",
    "Bwd Packets Length Total": "Total Length of Bwd Packets",
    "Packet Length Max": "Max Packet Length",
    "Packet Length Min": "Min Packet Length",
    "Avg Packet Size": "Average Packet Size",
    "Init Fwd Win Bytes": "Init_Win_bytes_forward",
    "Init Bwd Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Packets": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}


def convert_live_flows(csv_path, expected_features):
    """Read captured flows and map them to the model's feature order."""

    live = pd.read_csv(csv_path)
    live.columns = live.columns.str.strip()

    live = live.rename(columns={
        source: target
        for target, source in COLUMN_MAP.items()
        if source in live.columns
    })

    missing = [
        feature for feature in expected_features
        if feature not in live.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    result = live[expected_features].copy()

    for column in expected_features:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )

    result = result.replace([np.inf, -np.inf], np.nan)

    # Reject incomplete or invalid records instead of guessing values.
    valid = result.notna().all(axis=1)

    # Rates can be -1 when flow duration is zero.
    # Such records are excluded pending a verified compatibility rule.
    rate_columns = ["Flow Bytes/s", "Flow Packets/s"]
    for column in rate_columns:
        if column in result.columns:
            valid &= result[column] >= 0

    result = result.loc[valid].astype("float32")

    return result, int((~valid).sum())
