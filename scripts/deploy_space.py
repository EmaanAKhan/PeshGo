"""Deploy PeshGo to Hugging Face: a dataset repo (the hourly snapshot) and a Space (the public page).

    python scripts/deploy_space.py                       # static Space (free): the page the pipeline re-renders every hour
    python scripts/deploy_space.py --user my-hf-name     # create the repos under this account or organisation
    python scripts/deploy_space.py --gradio              # the interactive Gradio app instead (Hugging Face requires PRO for this)

It uses your Hugging Face login (`hf auth login`) or HF_TOKEN. It creates <user>/peshgo-snapshot (dataset)
and <user>/peshgo (Space), uploads the current snapshot and page, and tells you which HF_DATASET_REPO to put
in .env so that every hourly pipeline run keeps the Space fresh. Repos are public; no secret is uploaded.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STATIC_README = """---
title: PeshGo
emoji: 🌫️
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: Lahore PM2.5 forecast, with advice in Urdu and English
---

PeshGo: a 24-hour PM2.5 forecast for Lahore with a likely range and cited advice in Urdu and English,
for outdoor workers, parents and schools. The page is re-rendered every hour by the team's pipeline.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", help="Hugging Face user or organisation (default: the logged-in user)")
    ap.add_argument("--space-name", default="peshgo")
    ap.add_argument("--dataset-name", default="peshgo-snapshot")
    ap.add_argument("--gradio", action="store_true", help="deploy the Gradio app (needs a PRO plan on Hugging Face)")
    a = ap.parse_args(argv)

    from huggingface_hub import HfApi

    from saafhawa.settings import secrets
    api = HfApi(token=secrets().hf_token or None)  # None: the cached `hf auth login` token
    who = api.whoami()
    user = a.user or who["name"]
    dataset, space = f"{user}/{a.dataset_name}", f"{user}/{a.space_name}"
    print(f"Logged in as {who['name']}; deploying to {user}")

    snap = ROOT / "data" / "snapshot"
    if not (snap / "status.json").exists():
        print("No snapshot yet: run `make run` first so there is something to show.")
        return 1
    if not (snap / "index.html").exists():
        from saafhawa.static_site import render
        (snap / "index.html").write_text(render(snap), encoding="utf-8")

    api.create_repo(dataset, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(snap), repo_id=dataset, repo_type="dataset", commit_message="snapshot")
    print(f"Dataset: https://huggingface.co/datasets/{dataset}")

    if a.gradio:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_space.py")], check=True)
        api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
        api.upload_folder(folder_path=str(ROOT / "build" / "space"), repo_id=space, repo_type="space", commit_message="deploy PeshGo app")
        api.add_space_variable(space, "HF_DATASET_REPO", dataset)
    else:
        api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=STATIC_README.encode("utf-8"), path_in_repo="README.md", repo_id=space, repo_type="space", commit_message="Space card")
        api.upload_file(path_or_fileobj=str(snap / "index.html"), path_in_repo="index.html", repo_id=space, repo_type="space", commit_message="page")
        # The files that make the page installable as an app (manifest, icons, service worker); uploaded once, they never change.
        import tempfile

        from saafhawa.static_site import write_assets
        with tempfile.TemporaryDirectory() as tmp:
            names = write_assets(tmp)
            api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space", commit_message="app install files")
            print("App files:", ", ".join(names))
    print(f"Space:   https://huggingface.co/spaces/{space}")
    print(f"Live at: https://{user.lower()}-{a.space_name.lower()}.static.hf.space" if not a.gradio else f"Live at: https://{user.lower()}-{a.space_name.lower()}.hf.space")
    print(f"Next: set HF_DATASET_REPO={dataset} in .env (and restart `make serve`) so each pipeline run refreshes the Space.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
