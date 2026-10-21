"""Hugging Face Space entry point.

The Space never sees the team's warehouse. Every hourly pipeline run uploads a
snapshot to a Hugging Face dataset repo (HF_DATASET_REPO); this app downloads
it on start and every 10 minutes, and shows it read-only.

Space settings -> Variables: HF_DATASET_REPO = <user>/peshgo-snapshot
(a public dataset repo needs no token here).
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from saafhawa.dashboard import build_app

REPO = os.environ.get("HF_DATASET_REPO", "")
LOCAL = Path("snapshot")
LOCAL.mkdir(exist_ok=True)


def refresh():
    if REPO:
        snapshot_download(REPO, repo_type="dataset", local_dir=str(LOCAL), token=os.environ.get("HF_TOKEN") or None)


refresh()
build_app(LOCAL, allow_approvals=False, refresh=refresh).launch()
