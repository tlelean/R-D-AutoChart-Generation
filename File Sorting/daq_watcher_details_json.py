#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAQ Station Incoming Watcher (extension-based routing)

Behaviour:
- Reads manifest.json if present; if missing, falls back to any *_details.json file (reading from metadata section).
- If neither manifest.json nor *_details.json are present, places run in BASE/Unknown (single subfolder only).
- Moves/Copies all *.csv into CSV destination; all *.pdf into PDF destination.
- Does not transfer manifest.json with the files.
- CSV presence is required (at least one); PDFs are optional.
- Pass/Fail routing:
    PASS/TRUE  -> CSVs → BASE/<op>/<job>/<valve>/Attempt <attempt>/CSV/<section>/
                  PDFs → BASE/<op>/<job>/<valve>/Attempt <attempt>/PDF/
    FAIL/FALSE -> all (CSVs & PDFs) → BASE/<op>/<job>/<valve>/Attempt <attempt>/Failed/<section>/
- Optionally deletes the run directory after successful delivery.
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
LOGDIR = BASE / "logs"

POLL_INTERVAL = 5
DELIVERY_MODE = "MOVE"
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 5
DELETE_RUN_DIR_ON_SUCCESS = True

# --------------------
# Logging
# --------------------

def setup_logger() -> logging.Logger:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("daq_watcher")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOGDIR / "daq_watcher.log", maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS)
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
    return sorted([p for p in INCOMING.iterdir() if p.is_dir()])

def _overwrite_safe_move(src: Path, dst: Path) -> None:
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
    if DELIVERY_MODE.upper() == "COPY":
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        _overwrite_safe_move(src, dst)

def sanitize_component(value: Optional[str], default: str = "Unknown", allow_empty: bool = False) -> str:
    if value is None:
        s = ""
    else:
        s = str(value)
    s = "".join(ch for ch in s if ch >= " ")
    s = s.replace("/", "-").replace("\\", "-").strip()
    if not s and not allow_empty:
        return default
    return s

def normalise_attempt(value: Optional[str]) -> str:
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
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        log.exception("Failed to delete directory tree %s", path)

# --------------------
# Manifest I/O (with fallback)
# --------------------

def load_manifest(run_dir: Path) -> Dict:
    mpath = run_dir / "manifest.json"
    details_files = list(run_dir.glob("*_details.json"))
    dpath = details_files[0] if details_files else None

    if mpath.exists():
        log.info("Run %s: Using manifest.json", run_dir.name)
        with open(mpath, "r", encoding="utf-8") as f:
            m = json.load(f)
        m.pop("files", None)
        m.setdefault("run_id", run_dir.name)
        m.setdefault("decision", "TRUE")
        m.setdefault("operator", "Unknown")
        m.setdefault("job_number", "Unknown")
        m.setdefault("valve_drawing_no", "Unknown")
        m.setdefault("attempt", "1")
        m.setdefault("section", "Unknown")
        m.setdefault("test_name", "chart")
        return m

    elif dpath:
        log.info("Run %s: Using details file %s", run_dir.name, dpath.name)
        with open(dpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        metadata = d.get("metadata", {})

        return {
            "run_id": metadata.get("Date Time", run_dir.name),
            "decision": d.get("decision", "TRUE"),
            "operator": metadata.get("Operator", "Unknown"),
            "rig_no": metadata.get("Data Logger", "Unknown"),
            "job_number": metadata.get("Job Number", "Unknown"),
            "valve_drawing_no": metadata.get("Valve Drawing Number", "Unknown"),
            "attempt": metadata.get("Attempt", "1"),
            "test_name": metadata.get("Test Name", "chart"),
            "section": metadata.get("Test Section Number", "Unknown"),
        }

    log.warning("Run %s: No manifest.json or *_details.json found; placing into Unknown folder.", run_dir.name)
    return {"unknown_path": True}

# --------------------
# Routing rules
# --------------------

def build_destinations(m: Dict) -> Dict[str, Path]:
    if m.get("unknown_path"):
        base_attempt = BASE / "Unknown"
        return {"csv_pass": base_attempt, "pdf_pass": base_attempt, "fail_both": base_attempt}

    op  = sanitize_component(m.get("operator"), default="Unknown")
    job = sanitize_component(m.get("job_number"), default="Unknown")
    val = sanitize_component(m.get("valve_drawing_no"), default="Unknown")
    att = normalise_attempt(m.get("attempt"))
    sec = sanitize_component(m.get("section"), default="Unknown")

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
    raw = "" if value is None else str(value)
    if raw.strip() == "":
        log.warning("Manifest field '%s' was blank; using default '%s'", field, default_used)

def process_run(run_dir: Path) -> None:
    run_id = run_dir.name
    marker = run_dir / ".processing"
    if marker.exists():
        return

    completed_ok = False

    try:
        atomic_write(marker, b"processing")

        # Identify files
        csv_files = sorted(run_dir.glob("*.csv"))
        pdf_files = sorted(run_dir.glob("*.pdf"))
        manifest_path = run_dir / "manifest.json"
        details_files = sorted(run_dir.glob("*_details.json"))
        # JSONs to transfer (exclude manifest.json)
        json_files = [f for f in run_dir.glob("*.json") if f.name != "manifest.json"]

        # Enforce required files: at least one CSV and one *_details.json
        if not csv_files:
            log.warning("Run %s: missing required file type: CSV", run_id)
            return
        if not details_files:
            log.warning("Run %s: missing required file type: *_details.json", run_id)
            return

        # Load manifest (prefers manifest.json; else use details)
        m = load_manifest(run_dir)

        # Honour decision unless unknown_path (then treat as pass)
        dec_pass = truthy_decision(m.get("decision")) if not m.get("unknown_path") else True

        dests = build_destinations(m)

        try:
            if dec_pass:
                csv_dest = dests["csv_pass"]
                pdf_dest = dests["pdf_pass"]
                ensure_dirs(csv_dest, pdf_dest)

                # CSVs + (non-manifest) JSONs (incl. *_details.json) go to CSV destination
                for src in csv_files:
                    move_or_copy(src, csv_dest / src.name)
                for src in json_files:
                    move_or_copy(src, csv_dest / src.name)

                # PDFs to PDF destination (optional)
                for src in pdf_files:
                    move_or_copy(src, pdf_dest / src.name)

                log.info("Run %s: delivered to PASS destinations", run_id)

            else:
                fail_dest = dests["fail_both"]
                ensure_dirs(fail_dest)

                # On FAIL: everything (CSV, PDF, other JSON) goes to Failed/<section>/
                for src in csv_files + pdf_files + json_files:
                    move_or_copy(src, fail_dest / src.name)

                log.info("Run %s: delivered to FAIL destination %s", run_id, fail_dest)

        except Exception as e:
            log.exception("Run %s: delivery failed: %s", run_id, e)
            return

        # Never transfer manifest.json; remove it from the run dir if present
        if manifest_path.exists():
            try:
                manifest_path.unlink()
                log.info("Run %s: manifest.json deleted", run_id)
            except Exception as e:
                log.exception("Run %s: failed to delete manifest.json: %s", run_id, e)

        completed_ok = True

    finally:
        try:
            if marker.exists():
                marker.unlink(missing_ok=True)
        except Exception:
            pass

        if completed_ok and DELETE_RUN_DIR_ON_SUCCESS:
            remove_tree(run_dir)
            log.info("Run %s: run directory removed from Incoming", run_id)

# --------------------
# Main loop
# --------------------

def main():
    ensure_dirs(INCOMING, LOGDIR)
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