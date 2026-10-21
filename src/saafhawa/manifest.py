"""The raw layer and its manifest.

Every raw file is saved exactly as fetched and recorded in data/manifest.csv
with its source, query, fetch time, SHA-256 and licence. Raw files are never
edited. `verify()` re-hashes every file against the manifest, which is how
`make data` proves the raw layer is the evidence it claims to be.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .settings import path

FIELDS = ["source", "query", "fetch_time", "sha256", "licence", "raw_path", "bytes"]

LICENCES = {
    "openaq": "Per data provider, as listed by OpenAQ (see each location's licence in the raw file)",
    "open-meteo-weather": "CC BY 4.0 (Open-Meteo; ERA5/ECMWF data)",
    "open-meteo-air": "CC BY 4.0 (Open-Meteo; Copernicus CAMS data)",
    "nasa-firms": "NASA open data (FIRMS); cite NASA FIRMS",
    "docs": "As stated by each publisher (see docs/sources.yaml)",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read() -> list[dict]:
    p = path("manifest")
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _raw_root() -> Path:
    return path("raw")


@dataclass
class Saved:
    raw_path: str
    sha256: str
    new: bool


_lock = threading.Lock()


@contextlib.contextmanager
def _file_lock(timeout: float = 120.0):
    """A lock shared by separate processes (two overlapping pipeline runs once wrote the same
    raw file at the same moment, leaving a checksum that matched neither payload)."""
    lock = path("data") / ".manifest.lock"
    fh = open(lock, "a+")
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fh.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def save(source: str, query: str, relpath: str, data: bytes, licence: str | None = None) -> Saved:
    """Write one raw file and append its manifest row (skips exact duplicates).
    Safe to call from parallel downloads and from separate processes."""
    with _lock, _file_lock():
        return _save(source, query, relpath, data, licence)


def _save(source, query, relpath, data, licence) -> Saved:
    raw = _raw_root() / relpath
    raw.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(data)
    rel = raw.relative_to(path("data")).as_posix()
    existing = {r["raw_path"]: r for r in read()}
    if rel in existing and existing[rel]["sha256"] == digest and raw.exists():
        return Saved(rel, digest, new=False)
    if rel in existing:
        # Raw files are immutable: a different payload gets a new name.
        stem, dot, ext = raw.name.partition(".")
        raw = raw.with_name(f"{stem}__{digest[:12]}{dot}{ext}")
        rel = raw.relative_to(path("data")).as_posix()
    raw.write_bytes(data)
    row = {
        "source": source,
        "query": query,
        "fetch_time": now_utc(),
        "sha256": digest,
        "licence": licence or LICENCES.get(source, "unknown"),
        "raw_path": rel,
        "bytes": len(data),
    }
    p = path("manifest")
    new_file = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    return Saved(rel, digest, new=True)


def save_json(source: str, query: str, relpath: str, obj) -> Saved:
    return save(source, query, relpath, json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def verify() -> dict:
    """Re-hash every raw file. Returns counts; raises if anything is missing or changed."""
    rows = read()
    bad, missing = [], []
    for r in rows:
        p = path("data") / r["raw_path"]
        if not p.exists():
            missing.append(r["raw_path"])
        elif sha256_file(p) != r["sha256"]:
            bad.append(r["raw_path"])
    if bad or missing:
        raise RuntimeError(f"Manifest check failed: {len(bad)} changed, {len(missing)} missing. First: {(bad + missing)[:3]}")
    return {"files": len(rows), "bytes": sum(int(r["bytes"]) for r in rows)}


def has(relpath: str) -> bool:
    """True if a raw file was already saved under this name (or a later version of it).

    Lets long history downloads resume where they stopped instead of starting over."""
    rel = (_raw_root() / relpath).relative_to(path("data")).as_posix()
    folder, _, name = rel.rpartition("/")
    stem = name.partition(".")[0]
    for r in read():
        f, _, n = r["raw_path"].rpartition("/")
        if f == folder and (n == name or n.startswith(stem + "__")):
            return True
    return False


def files(source: str) -> list[Path]:
    """Raw files of one source, in manifest order."""
    return [path("data") / r["raw_path"] for r in read() if r["source"] == source]
