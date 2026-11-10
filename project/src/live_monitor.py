
"""Live network flow monitoring and ML prediction."""

import csv
import io
import os
import subprocess
import threading
import time
import traceback

import config
from src import live_features
from src import model_registry as reg
from src import data_loader as dl


class LiveMonitor:

    def __init__(self):
        self.process = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.last_error = None
        self.processed = 0
        self.rejected = 0
        self.on_flows = None

        self.csv_path = os.path.join(
            config.ROOT_DIR if hasattr(config, "ROOT_DIR") else os.getcwd(),
            "live_flows.csv"
        )

    def status(self):
        return {
            "running": self.running,
            "processed": self.processed,
            "rejected": self.rejected,
            "last_error": self.last_error,
        }

    def start(self, interface):
        with self.lock:
            if self.running:
                return False, "Live monitoring is already running"

            if not interface:
                return False, "Network interface is required"

            if os.path.exists(self.csv_path):
                os.remove(self.csv_path)

            self.processed = 0
            self.rejected = 0

            command = [
                "pycicflowmeter",
                "-i", interface,
                "-o", self.csv_path,
                "--activity-timeout", "5",
            ]

            try:
                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as exc:
                self.last_error = str(exc)
                return False, str(exc)

            self.running = True
            self.last_error = None

            self.thread = threading.Thread(
                target=self._watch_csv,
                daemon=True
            )
            self.thread.start()

        return True, "Live monitoring started"

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()

            self.running = False

        return True, "Live monitoring stopped"

    def _watch_csv(self):
        seen = 0

        while self.running:
            try:
                if self.process and self.process.poll() is not None:
                    self.last_error = "Flow capture process stopped"
                    self.running = False
                    break

                if os.path.exists(self.csv_path):
                    with open(
                        self.csv_path,
                        "r",
                        newline="",
                        encoding="utf-8-sig"
                    ) as file:
                        content = file.read()

                    lines = content.splitlines(keepends=True)

                    # Ignore the final line if it is still being written.
                    if lines and not lines[-1].endswith("\n"):
                        lines = lines[:-1]

                    complete_content = "".join(lines)

                    if complete_content:
                        rows = list(
                            csv.DictReader(io.StringIO(complete_content))
                        )

                        if len(rows) > seen:
                            new_rows = rows[seen:]

                            self._process_rows(new_rows)

                            # Update only after successful processing.
                            seen = len(rows)

                time.sleep(2)

            except Exception as exc:
                self.last_error = str(exc)
                traceback.print_exc()
                time.sleep(2)

    def _process_rows(self, rows):
        if not rows:
            return

        callback = getattr(self, "on_flows", None)

        if callback is None:
            self.last_error = (
                "Live prediction callback is not configured"
            )
            return

        result = callback(rows)

        self.processed += result["processed"]
        self.rejected += result["rejected"]
