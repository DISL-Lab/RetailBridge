"""Download the RetailBridge benchmark from the Hugging Face Hub.

Fetches the 1,680 questions and the evaluation database (five SQLite tables and
three text stores) from https://huggingface.co/datasets/DISLab/RetailBridge into
data/RetailBridge/, the default location read by retailrag.config and evaluate.py.

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --local-dir /path/to/RetailBridge
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "DISLab/RetailBridge"
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "RetailBridge"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir", default=str(DEFAULT_DIR))
    args = p.parse_args()
    path = snapshot_download(REPO_ID, repo_type="dataset", local_dir=args.local_dir,
                             allow_patterns=["queries/*", "database/raw/*"])
    print(f"questions: {Path(path) / 'queries' / 'test.jsonl'}")
    print(f"database:  {Path(path) / 'database' / 'raw'}")


if __name__ == "__main__":
    main()
