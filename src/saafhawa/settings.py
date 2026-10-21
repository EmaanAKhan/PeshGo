"""Configuration and secrets.

Settings come from config.yaml. Secrets (API keys, tokens) come only from the
environment or a local .env file, never from the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines, no override of real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Secrets:
    openaq_key: str | None
    firms_key: str | None
    hf_token: str | None
    hf_dataset_repo: str | None
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str | None
    hf_space_repo: str | None = None  # the static Space that shows the page; default <dataset owner>/peshgo


@lru_cache(maxsize=1)
def config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else Path(os.environ.get("SAAFHAWA_CONFIG", ROOT / "config.yaml"))
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(os.environ.get("SAAFHAWA_HOME", ROOT))
    cfg["paths"] = {k: str((base / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg


def secrets() -> Secrets:
    _load_dotenv(ROOT / ".env")
    g = os.environ.get
    return Secrets(
        openaq_key=g("OPENAQ_API_KEY") or None,
        firms_key=g("FIRMS_MAP_KEY") or None,
        hf_token=g("HF_TOKEN") or None,
        hf_dataset_repo=g("HF_DATASET_REPO") or None,
        llm_api_key=g("LLM_API_KEY") or None,
        llm_base_url=g("LLM_BASE_URL") or None,
        llm_model=g("LLM_MODEL") or None,
        hf_space_repo=g("HF_SPACE_REPO") or None,
    )


def path(name: str) -> Path:
    p = Path(config()["paths"][name])
    if name in {"data", "raw", "models", "reports", "snapshot"}:
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p
