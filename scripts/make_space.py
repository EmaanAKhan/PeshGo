"""Assemble the folder to push to the Hugging Face Space.

    python scripts/make_space.py            # writes build/space/
    cd build/space && git init && git remote add origin https://huggingface.co/spaces/<user>/peshgo && git add . && git commit -m deploy && git push -f origin HEAD:main

The Space gets the saafhawa package, config.yaml and the app; no data, keys or models.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "space"

README = """---
title: PeshGo
emoji: 🌫️
colorFrom: gray
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

PeshGo: 24-hour PM2.5 forecast for Lahore with cited Urdu and English advice.
Reads the hourly snapshot published by the team's Prefect pipeline.
"""

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "src").mkdir(parents=True)
shutil.copytree(ROOT / "src" / "saafhawa", OUT / "src" / "saafhawa", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(ROOT / "docs", OUT / "docs", ignore=shutil.ignore_patterns("*.pdf", "*.md"))
shutil.copy2(ROOT / "config.yaml", OUT / "config.yaml")
shutil.copy2(ROOT / "space" / "requirements.txt", OUT / "requirements.txt")
app = (ROOT / "space" / "app.py").read_text(encoding="utf-8")
# The package lives in src/ next to app.py on the Space.
app = app.replace("from huggingface_hub import snapshot_download",
                  "import sys\nsys.path.insert(0, str(Path(__file__).parent / 'src'))\n\nfrom huggingface_hub import snapshot_download", 1)
(OUT / "app.py").write_text(app, encoding="utf-8")
(OUT / "README.md").write_text(README, encoding="utf-8")
print(f"Space folder ready: {OUT}")
