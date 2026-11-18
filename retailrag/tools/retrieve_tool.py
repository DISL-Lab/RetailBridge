"""dense_retrieve tool: query text Vector DB(s) optionally filtered by entity IDs.

Per-call routing: every call must declare which text source(s) to query —
either ``source=<name>`` for a single store or ``sources=[...]`` for a
subset. There is no implicit "fan out across all 3 sources" path; that
collapsing was the text-side analog of the SQL ATTACH free lunch and
silently bypassed the routing decision.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import DEFAULT_CONFIG, VECTOR_SOURCES
from ..databases import VectorDB
from ..llm import call_llm_json


# ---------------- text-source picker (multi-select over 3 vector stores) ---


_TEXT_PICKER_SYSTEM = """You decide which retail TEXT (vector) stores must be queried for a sub-question.

Available text stores (each is a dense-indexed corpus of retail passages):
  • sales_scripts  — broadcast/host scripts; product feature talk, promo wording
  • inquiries_text — buyer-side support questions (pre-/post-purchase)
  • reviews_text   — buyer reviews on broadcasts, sellers, products

Rules:
  - Pick the MINIMUM set of stores that can answer the sub-question. Do NOT pick all three by default.
  - It is perfectly fine to pick only one store.
  - Output JSON: {"sources": ["reviews_text"], "why": "one short reason"}
"""


def pick_text_sources(sub_query: str, fallback: Optional[List[str]] = None,
                      temperature: float = 0.0) -> List[str]:
    valid = set(VECTOR_SOURCES.keys())
    fallback = fallback if fallback is not None else sorted(valid)
    try:
        out = call_llm_json(_TEXT_PICKER_SYSTEM,
                            f"Sub-question: {sub_query}\n\nOutput JSON.",
                            temperature=temperature, max_retries=1)
        cand = out.get("sources") or []
        if isinstance(cand, str):
            cand = [cand]
        cleaned = [s for s in cand if isinstance(s, str) and s in valid]
        if cleaned:
            seen = set()
            res = []
            for s in cleaned:
                if s not in seen:
                    seen.add(s)
                    res.append(s)
            return res
    except Exception:
        pass
    return fallback


class RetrieveTool:
    def __init__(self, vector_db: Optional[VectorDB] = None) -> None:
        self.vector_db = vector_db or VectorDB()

    @property
    def available_sources(self) -> List[str]:
        return list(VECTOR_SOURCES.keys())

    def call(
        self,
        query: str,
        source: str,
        filter_ids: Optional[Iterable[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Strip "[SQL] " / "[TEXT] " prefix some models (notably GPT-5.4) copy
        # verbatim from db_block section headers.
        import re as _re
        source = _re.sub(r"^\s*\[(?:SQL|TEXT)\]\s+", "", source or "")
        if source not in VECTOR_SOURCES:
            return {
                "tool": "dense_retrieve",
                "source": source,
                "results": [],
                "status": f"unknown source '{source}'. Choose from {self.available_sources}",
            }
        k = top_k or DEFAULT_CONFIG.pipeline.retrieve_top_k
        try:
            results = self.vector_db.search(
                source=source,
                query=query,
                top_k=k,
                filter_ids=filter_ids,
                filter_metadata=filter_metadata,
            )
            return {
                "tool": "dense_retrieve",
                "source": source,
                "query": query,
                "filter_ids": list(filter_ids) if filter_ids else None,
                "results": results,
                "status": f"OK ({len(results)} hits)",
            }
        except FileNotFoundError as e:
            return {
                "tool": "dense_retrieve",
                "source": source,
                "results": [],
                "status": f"INDEX MISSING: {e}",
            }
        except Exception as e:
            return {
                "tool": "dense_retrieve",
                "source": source,
                "results": [],
                "status": f"ERROR: {e}",
            }
