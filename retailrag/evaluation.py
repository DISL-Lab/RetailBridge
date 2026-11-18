"""Lightweight evaluation utilities for RetailRAG.

The QA gold answers in this project are short — names, dates, numeric
values, or short phrases — and the predictions are full-sentence answers
emitted by the alignment agent. We therefore report:

  * exact_match     — strict normalized equality
  * substring_match — gold appears (normalized) inside pred (relaxed)
  * f1              — SQuAD-style token F1
  * numeric_match   — for answers that parse as numbers, ±tol relative match

Aggregated per hop and per source-mix so the paper can show breakdowns.
"""
from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TBL = str.maketrans("", "", string.punctuation)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize(text: Any) -> str:
    s = "" if text is None else str(text)
    s = s.lower()
    s = s.translate(_PUNCT_TBL)
    s = _ARTICLES.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def exact_match(pred: Any, gold: Any) -> int:
    return int(normalize(pred) == normalize(gold))


def substring_match(pred: Any, gold: Any) -> int:
    p = normalize(pred)
    g = normalize(gold)
    if not g:
        return 0
    return int(g in p)


def f1(pred: Any, gold: Any) -> float:
    p_tokens = normalize(pred).split()
    g_tokens = normalize(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _parse_numbers(text: Any) -> List[float]:
    return [float(m) for m in _NUM_RE.findall("" if text is None else str(text))]


_DATE_RE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}:\d{2}")
_DATE_PART_RE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")  # YYYY-MM-DD only
_TIME_PART_RE = re.compile(r"\d{1,2}:\d{2}")           # HH:MM only


def _looks_like_date(s: Any) -> bool:
    """True if the string contains a date or time pattern."""
    return bool(_DATE_RE.search("" if s is None else str(s)))


def numeric_match(pred: Any, gold: Any, rel_tol: float = 0.05, abs_tol: float = 1.0) -> Optional[int]:
    """Return 1/0 if both pred and gold are numeric scalars; else None.

    A numeric pred is "correct" when *some* number it mentions is close
    enough (relative or absolute tolerance) to the gold number. This keeps
    things sane for full-sentence answers like "...is 0." / "237.08 USD".

    Dates/times are excluded — those are matched by the date-aware exact
    check, since `numeric_match` would treat "2024-12-29 10:00" as the list
    [2024, 12, 29, 10, 0] and the abs_tol of 1.0 on the leading year would
    match across consecutive years.
    """
    if _looks_like_date(gold) or _looks_like_date(pred):
        return None
    g_nums = _parse_numbers(gold)
    p_nums = _parse_numbers(pred)
    if not g_nums or not p_nums:
        return None
    g = g_nums[0]
    for p in p_nums:
        if abs(p - g) <= abs_tol:
            return 1
        denom = abs(g) if g != 0 else 1.0
        if abs(p - g) / denom <= rel_tol:
            return 1
    return 0


def date_match(pred: Any, gold: Any) -> Optional[int]:
    """Date-aware match.

    Earlier version returned 1 if ANY date OR time substring matched, which
    caused "2025-10-22 10:00" vs "2024-02-09 10:00" to match on '10:00' alone
    despite completely different dates. This version requires:
      - the YYYY-MM-DD component to match (if gold has one)
      - AND the HH:MM component to match (only if gold has one)
    Pred can have additional/different sub-components beyond what's required.
    """
    g_text = "" if gold is None else str(gold)
    p_text = "" if pred is None else str(pred)
    g_dates = _DATE_PART_RE.findall(g_text)
    g_times = _TIME_PART_RE.findall(g_text)
    if not g_dates and not g_times:
        return None  # gold has no date/time component
    p_dates = _DATE_PART_RE.findall(p_text)
    p_times = _TIME_PART_RE.findall(p_text)
    # Require date component to match if gold has one.
    if g_dates and not any(g == p for g in g_dates for p in p_dates):
        return 0
    # Require time component to match if gold has one.
    if g_times and not any(g == p for g in g_times for p in p_times):
        return 0
    return 1


def score_item(pred: Any, gold: Any) -> Dict[str, Any]:
    em = exact_match(pred, gold)
    sub = substring_match(pred, gold)
    f = f1(pred, gold)
    num = numeric_match(pred, gold)
    dt = date_match(pred, gold)
    # acc semantics (2026-05-03, tightened):
    #   * substring(gold ⊆ pred) ONLY. Numeric- and date-partial matches were
    #     producing false-positives (e.g., a single shared number in a long
    #     answer flagged the prediction correct even when the entity differed).
    # any_match keeps the relaxed OR-of-everything for diagnostic reference.
    return {
        "em": em,
        "substring": sub,
        "f1": round(f, 4),
        "numeric_match": num,
        "date_match": dt,
        "any_match": int(em or sub or (num == 1) or (dt == 1)),
        "acc":       int(sub),
    }


# ---------------------------------------------------------------- LLM judge

LLM_JUDGE_SYSTEM = """You are a strict but semantically-aware grading judge.
You will receive a question, the gold answer, and a predicted answer. Decide
if the prediction is CORRECT given the gold.

Rules:
  - Accept paraphrases, equivalent units, and capitalization differences.
  - Accept equivalent numeric formats when the underlying value matches.
  - Reject if the predicted value is a different entity, number, or date.
  - Reject "INSUFFICIENT" or "I don't know" predictions.

Output JSON only:
{ "correct": 1 | 0, "why": "one short sentence" }
"""


def llm_judge(
    question: Any,
    gold: Any,
    pred: Any,
    cfg: Any = None,
) -> Dict[str, Any]:
    """LLM-based binary correctness judge."""
    from .llm import call_llm_json

    user = (
        f"Question: {question}\n"
        f"Gold Answer: {gold}\n"
        f"Predicted Answer: {pred}\n\n"
        "Judge."
    )
    try:
        out = call_llm_json(LLM_JUDGE_SYSTEM, user, cfg=cfg, temperature=0.0, max_retries=1)
        v = out.get("correct")
        if isinstance(v, bool):
            v = int(v)
        if v not in (0, 1):
            v = 0
        return {"llm_eval": v, "llm_eval_why": out.get("why", "")}
    except Exception as e:
        return {"llm_eval": None, "llm_eval_why": f"judge error: {e}"}


# ---------------------------------------------------------------- aggregate


def _avg(values: Iterable[float]) -> float:
    vs = list(values)
    return round(sum(vs) / len(vs), 4) if vs else 0.0


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-item evaluations into overall + per-hop + per-mix views.

    Each input item is expected to have ``score`` (output of score_item)
    and ``meta`` with at least ``hops`` and ``sources_used`` (or ``type``).
    """
    overall = {
        "n": len(results),
        "acc":          _avg(r["score"]["acc"] for r in results) if results and "acc" in results[0]["score"] else _avg(r["score"]["any_match"] for r in results),
        "em":           _avg(r["score"]["em"] for r in results),
        "f1":           _avg(r["score"]["f1"] for r in results),
        "any_match":    _avg(r["score"]["any_match"] for r in results),
        "substring":    _avg(r["score"]["substring"] for r in results),
    }
    nums = [r["score"]["numeric_match"] for r in results
            if r["score"]["numeric_match"] is not None]
    overall["numeric_match"] = _avg(nums) if nums else None
    overall["numeric_n"] = len(nums)
    llms = [r["score"].get("llm_eval") for r in results
            if r["score"].get("llm_eval") is not None]
    overall["llm_eval"] = _avg(llms) if llms else None
    overall["llm_eval_n"] = len(llms)
    apis = [r.get("api_calls") for r in results
            if r.get("api_calls") is not None]
    overall["api_calls_mean"] = _avg(apis) if apis else None
    overall["api_calls_n"] = len(apis)

    by_hop: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    by_mix: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        meta = r.get("meta") or {}
        hops = meta.get("hops")
        if hops is not None:
            by_hop[int(hops)].append(r)
        mix = meta.get("type") or _infer_mix(meta.get("sources_used") or [])
        by_mix[mix].append(r)

    def _hop_block(items):
        d = {
            "n": len(items),
            "acc":        _avg(r["score"].get("acc", r["score"]["any_match"]) for r in items),
            "em":         _avg(r["score"]["em"] for r in items),
            "f1":         _avg(r["score"]["f1"] for r in items),
            "any_match":  _avg(r["score"]["any_match"] for r in items),
            "substring":  _avg(r["score"]["substring"] for r in items),
        }
        ll = [r["score"].get("llm_eval") for r in items if r["score"].get("llm_eval") is not None]
        d["llm_eval"] = _avg(ll) if ll else None
        d["llm_eval_n"] = len(ll)
        ap = [r.get("api_calls") for r in items if r.get("api_calls") is not None]
        d["api_calls_mean"] = _avg(ap) if ap else None
        d["api_calls_n"] = len(ap)
        return d

    breakdown_hop = {f"{h}-hop": _hop_block(items) for h, items in sorted(by_hop.items())}
    breakdown_mix = {
        mix: {
            "n": len(items),
            "em":         _avg(r["score"]["em"] for r in items),
            "any_match":  _avg(r["score"]["any_match"] for r in items),
        }
        for mix, items in sorted(by_mix.items())
    }
    return {
        "overall": overall,
        "by_hop": breakdown_hop,
        "by_mix": breakdown_mix,
    }


def _infer_mix(sources: List[str]) -> str:
    if not sources:
        return "unknown"
    has_text = any(s.endswith("_text") or s in {"sales_scripts"} for s in sources)
    has_sql = any(not (s.endswith("_text") or s == "sales_scripts") for s in sources)
    if has_text and has_sql:
        return "mix=sql+text"
    return "mix=sql" if has_sql else "mix=text"
