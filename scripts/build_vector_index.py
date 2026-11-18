"""Build dense embeddings for the three jsonl text sources.

Usage:
  python scripts/build_vector_index.py            # builds all three
  python scripts/build_vector_index.py reviews_text
  python scripts/build_vector_index.py --force    # rebuild even if cached

Outputs land under data/index/<source>.npy and data/index/<source>.pkl.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the package importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retailrag.config import VECTOR_SOURCES
from retailrag.databases import VectorDB


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs="*", help="subset of sources to build")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    targets = args.sources or list(VECTOR_SOURCES.keys())
    unknown = [t for t in targets if t not in VECTOR_SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}. Available: {list(VECTOR_SOURCES)}")

    db = VectorDB()
    for src in targets:
        t0 = time.time()
        n = db.build_index(src, force=args.force)
        dt = time.time() - t0
        if n < 0:
            print(f"[skip] {src}: cached (use --force to rebuild)")
        else:
            print(f"[ok]   {src}: {n} docs in {dt:.1f}s")


if __name__ == "__main__":
    main()
