"""Load the RetailBridge questions for RetailRAG evaluation.

The Hugging Face release (DISLab/RetailBridge, queries/test.jsonl or
queries/test.parquet) stores one record per question. The evaluator wants items
exposing ``question``, ``answer``, ``hops``, ``type`` and ``sources_used``; this
module adds those keys and keeps every original field for per-slice analysis.

Field mapping
-------------
    hops          <- reasoning_level      (1 / 2 / 3)
    type          <- evidence_modality    (sql-only / text-only / composite)
    sources_used  <- sources              (gold evidence sources, in hop order)
"""
import json
from pathlib import Path
from typing import Any, Dict, List


def _sources_for(item: Dict[str, Any]) -> List[str]:
    val = item.get("sources")
    if val is not None:
        return list(val)
    sc = item.get("source_chaining")
    if isinstance(sc, dict):  # {"Homo": [...]} / {"Hetero": [...]}
        for v in sc.values():
            if v:
                return list(v)
    return []


def to_eval_item(q: Dict[str, Any]) -> Dict[str, Any]:
    """Return an evaluator-shaped item, preserving the original fields."""
    item = dict(q)
    item["question"] = q.get("question") or q.get("query") or ""
    item["answer"] = q.get("answer")
    item["hops"] = q.get("reasoning_level")
    item["type"] = q.get("evidence_modality")
    item["sources_used"] = _sources_for(q)
    return item


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load questions from a .jsonl, .json (list or {"questions": [...]}) or .parquet file."""
    p = Path(path)
    if p.suffix == ".jsonl":
        questions = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    elif p.suffix == ".parquet":
        import pandas as pd
        questions = pd.read_parquet(p).to_dict(orient="records")
        for q in questions:
            if q.get("sources") is not None:
                q["sources"] = list(q["sources"])
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
        questions = obj["questions"] if isinstance(obj, dict) else obj
    return [to_eval_item(q) for q in questions]
