#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAQ Station Incoming Watcher (extension-based routing)

Behaviour:
- Reads manifest.json if present; no "files" key is required (and is ignored if present).
- Moves/Copies all *.csv into CSV destination; all *.pdf into PDF destination.
- CSV presence is required (at least one); PDFs are optional.
- Pass/Fail routing:
    PASS/TRUE  -> CSVs → BASE/<op>/<job>/<valve>/Attempt <attempt>/CSV/<section>/
                  PDFs → BASE/<op>/<job>/<valve>/Attempt <attempt>/PDF/
    FAIL/FALSE -> all (CSVs & PDFs) → BASE/<op>/<job>/<valve>/Attempt <attempt>/Failed/<section>/
- Writes an ACK file: BASE/Acks/<run_id>.ok
- Optionally deletes the run directory after successful delivery + ACK.
"""

import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
from logging.handlers import RotatingFileHandler

# --------------------
# Config
# --------------------

BASE = Path("V:/Userdoc/R & D/DAQ_Station")
INCOMING = BASE / "Incoming"
ACKS = BASE / "Acks"
LOGDIR = BASE / "logs"

POLL_INTERVAL = 5                 # seconds between scans
DELIVERY_MODE = "COPY"            # "MOVE" or "COPY"
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 5

# Remove the run directory from Incoming after a successful delivery + ACK.
# If you only want deletion when using MOVE, add: and DELIVERY_MODE.upper() == "MOVE"
DELETE_RUN_DIR_ON_SUCCESS = True

# --------------------
# Logging
# --------------------

def setup_logger() -> logging.Logger:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("daq_watcher")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOGDIR / "daq_watcher.log",
                                  maxBytes=LOG_MAX_BYTES,
                                  backupCount=LOG_BACKUPS)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

log = setup_logger()

# --------------------
# Signals
# --------------------

class GracefulExit(Exception):
    pass

def _sigterm_handler(signum, frame):
    raise GracefulExit()

signal.signal(signal.SIGINT, _sigterm_handler)
signal.signal(signal.SIGTERM, _sigterm_handler)

# --------------------
# Utils
# --------------------

def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def list_run_dirs() -> List[Path]:
    if not INCOMING.exists():
        return []
    # Only consider immediate child directories (each is a "run")
    return sorted([p for p in INCOMING.iterdir() if p.is_dir()])

def _overwrite_safe_move(src: Path, dst: Path) -> None:
    """Move with overwrite semantics (Windows-friendly)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if dst.is_file() or dst.is_symlink():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        except Exception:
            log.exception("Failed to remove existing destination %s before move", dst)
            raise
    shutil.move(str(src), str(dst))

def move_or_copy(src: Path, dst: Path):
    """
    Copy or move a file to dst. Copy overwrites; move overwrites safely across platforms.
    """
    if DELIVERY_MODE.upper() == "COPY":
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        _overwrite_safe_move(src, dst)

def sanitize_component(value: Optional[str], default: str = "Unknown", allow_empty: bool = False) -> str:
    """
    Normalise a manifest field for use as a path component.
    - Removes control chars
    - Replaces path separators
    - Trims whitespace
    - If empty after cleaning, returns `default` (unless allow_empty=True)
    """
    if value is None:
        s = ""
    else:
        s = str(value)
    # Drop control characters (anything below space)
    s = "".join(ch for ch in s if ch >= " ")
    # Replace path separators and trim surrounding whitespace
    s = s.replace("/", "-").replace("\\", "-").strip()
    if not s and not allow_empty:
        return default
    return s

def normalise_attempt(value: Optional[str]) -> str:
    """
    Ensure attempt is a positive integer string; default to "1" if blank/invalid.
    Trims whitespace and strips leading zeros.
    """
    s = sanitize_component(value, default="1")
    try:
        n = int(str(s).strip())
        if n <= 0:
            return "1"
        return str(n)
    except Exception:
        return "1"

def truthy_decision(s: Optional[str]) -> bool:
    if s is None:
        return True
    return str(s).strip() in {"TRUE","True","true","PASS","OK","1","Yes","YES"}

def remove_tree(path: Path):
    """Best-effort recursive delete of a directory tree."""
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        log.exception("Failed to delete directory tree %s", path)

# --------------------
# Manifest I/O
# --------------------

def load_manifest(run_dir: Path) -> Dict:
    """
    Load manifest.json if present, otherwise infer a minimal one.
    No 'files' key is used anymore; if present it is ignored.
    """
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        # Infer minimal manifest if missing
        run_id = run_dir.name
        return {
            "run_id": run_id,
            "decision": "TRUE",
            "operator": "Unknown",
            "job_number": "Unknown",
            "valve_drawing_no": "Unknown",
            "attempt": "1",
            "section": "Unknown",
            "test_name": "chart",
        }
    with open(mpath, "r", encoding="utf-8") as f:
        m = json.load(f)
        # Be tolerant to legacy manifests
        m.pop("files", None)
        # Ensure required keys exist (fall back sensibly)
        m.setdefault("run_id", run_dir.name)
        m.setdefault("decision", "TRUE")
        m.setdefault("operator", "Unknown")
        m.setdefault("job_number", "Unknown")
        m.setdefault("valve_drawing_no", "Unknown")
        m.setdefault("attempt", "1")
        m.setdefault("section", "Unknown")
        m.setdefault("test_name", "chart")
        return m

# --------------------
# Routing rules
# --------------------

def build_destinations(m: Dict) -> Dict[str, Path]:
    """
    PASS/TRUE:
      CSVs → BASE/<op>/<job>/<valve>/Attempt <attempt>/CSV/<section>/
      PDFs → BASE/<op>/<job>/<valve>/Attempt <attempt>/PDF/
    FAIL/FALSE:
      CSVs & PDFs → BASE/<op>/<job>/<valve>/Attempt <attempt>/Failed/<section>/
    """
    op  = sanitize_component(m.get("operator"),         default="Unknown")
    job = sanitize_component(m.get("job_number"),       default="Unknown")
    val = sanitize_component(m.get("valve_drawing_no"), default="Unknown")
    att = normalise_attempt(m.get("attempt"))
    sec = sanitize_component(m.get("section"),          default="Unknown")

    base_attempt = BASE / op / job / val / f"Attempt {att}"

    return {
        "csv_pass":  base_attempt / "CSV" / sec,
        "pdf_pass":  base_attempt / "PDF",
        "fail_both": base_attempt / "Failed" / sec
    }

# --------------------
# Processing
# --------------------

def _warn_if_blank(field: str, value: Optional[str], default_used: str):
    """
    If the raw field is blank/whitespace-only or None, log a warning that default was used.
    """
    raw = "" if value is None else str(value)
    if raw.strip() == "":
        log.warning("Manifest field '%s' was blank; using default '%s'", field, default_used)

def process_run(run_dir: Path) -> None:
    """
    Process a single run directory:
      - Reads manifest (or infers minimal one)
      - Globs by extension: *.csv (required), *.pdf (optional)
      - Routes/moves (or copies) files per decision
      - Writes ACK and cleans up
    """
    run_id = run_dir.name
    marker = run_dir / ".processing"
    if marker.exists():
        return

    completed_ok = False  # track successful completion for cleanup

    try:
        atomic_write(marker, b"processing")

        m = load_manifest(run_dir)
        dec_pass = truthy_decision(m.get("decision"))

        # Pre-sanitisation warnings for observability
        _warn_if_blank("operator",         m.get("operator"),         "Unknown")
        _warn_if_blank("job_number",       m.get("job_number"),       "Unknown")
        _warn_if_blank("valve_drawing_no", m.get("valve_drawing_no"), "Unknown")
        _warn_if_blank("attempt",          m.get("attempt"),          "1")
        _warn_if_blank("section",          m.get("section"),          "Unknown")

        # Find files by extension
        csv_files = sorted(run_dir.glob("*.csv"))
        pdf_files = sorted(run_dir.glob("*.pdf"))

        if not csv_files:
            log.error("Run %s: no CSV files found in %s", run_id, run_dir)
            return  # leave for retry/manual fix

        # Build destinations (sanitisation/normalisation done inside)
        dests = build_destinations(m)
        if dec_pass:
            csv_dest = dests["csv_pass"]
            pdf_dest = dests["pdf_pass"]
        else:
            csv_dest = dests["fail_both"]
            pdf_dest = dests["fail_both"]

        # Deliver
        try:
            ensure_dirs(csv_dest, pdf_dest)

            # CSVs (required)
            for src in csv_files:
                move_or_copy(src, csv_dest / src.name)

            # PDFs (optional)
            for src in pdf_files:
                move_or_copy(src, pdf_dest / src.name)

        except Exception as e:
            log.exception("Run %s: delivery failed: %s", run_id, e)
            return

        # Delete manifest.json now (best effort)
        try:
            (run_dir / "manifest.json").unlink(missing_ok=True)
            log.info("Run %s: manifest.json deleted", run_id)
        except Exception as e:
            log.exception("Run %s: failed to delete manifest.json: %s", run_id, e)
            # Ack is already written; continue

        # If we reached here, we consider the run completed successfully
        completed_ok = True

    finally:
        # Remove the .processing marker first
        try:
            if marker.exists():
                marker.unlink(missing_ok=True)
        except Exception:
            pass

        # If successful and configured, delete the entire run directory
        if completed_ok and DELETE_RUN_DIR_ON_SUCCESS:
            remove_tree(run_dir)
            log.info("Run %s: run directory removed from Incoming", run_id)

# --------------------
# Main loop
# --------------------

def main():
    ensure_dirs(INCOMING, ACKS, LOGDIR)
    log.info("DAQ watcher started. Base=%s, Mode=%s", BASE, DELIVERY_MODE.upper())
    try:
        while True:
            for run_dir in list_run_dirs():
                try:
                    process_run(run_dir)
                except Exception:
                    log.exception("Unexpected error processing %s", run_dir)
            time.sleep(POLL_INTERVAL)
    except GracefulExit:
        log.info("DAQ watcher stopping.")
    except Exception:
        log.exception("Fatal error; exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()