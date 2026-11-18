"""PlanReviseAgent: DB-adaptive plan, execute, diagnose and revise over SQL and text.

The PLAN and REVISE system prompts are built at construction time from a schema
briefing (``db_block``, composed once per database by ``db_block.make_db_block``),
so no retail-specific rules are baked into the agent.

Flow per query
  * Routing. Questions that ask about free-text passages go to a single dense
    retrieval; a cheap complexity classifier sends 1-hop lookups to a single SQL
    call. Either path escalates to the full loop when it fails.
  * Plan, execute, diagnose, revise. The planner emits atomic steps (a 1-step plan
    is allowed); failed SQL or text steps are diagnosed and the remaining plan is
    revised within a revision budget.
  * Plan retry (optional). A suspicious first attempt triggers a second plan at a
    higher temperature, and the better of the two answers is kept.
  * Answer. The final answer is composed from the gathered evidence.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import DEFAULT_CONFIG, Config
from ..llm import call_llm, call_llm_json
from ..tools import RetrieveTool, SQLTool
from .prompts import (
    ANSWER_TEMPLATE,
    DIAGNOSE_TEMPLATE,
    PLAN_TEMPLATE,
    REVISE_TEMPLATE,
)


# ---------------- adaptive complexity routing ----------------
#
# Plan-revise's atomic decomposition adds noise on simple lookups: it competes
# with Naive Text-to-SQL on 1-hop. Empirically (an internal split, n=384), 100 % of
# wrong 1-hop cases over-planned (avg 3.37 steps for a 1-step question). A
# cheap LLM gate routes simple lookups to a single-shot SQL path that mirrors
# Naive Text-to-SQL while still benefiting from SQLTool's safety nets
# (column-hallucination retry, zero-rows relax).

COMPLEXITY_SYSTEM = """You classify a retail QA query's reasoning complexity. Output ONE label.

  "simple"  — 1 hop. The answer is obtainable from a single source via ONE SQL/retrieval step. No bridge entity, no chaining. Plannable as a single atomic step.
  "hard"    — 2+ hop chain. The query requires at least one bridge entity ("the seller of the product that buyer X ordered"), or multiple aggregations across an entity chain. Needs 2 or more atomic steps.

Output JSON only:
{ "label": "simple"|"hard", "why": "one short reason" }
"""


# ----------------- placeholder substitution -----------------


_RE_DOTTED = re.compile(r"\{prev_(\d+)\.([a-zA-Z_][a-zA-Z0-9_]*)\}")
_RE_PLAIN = re.compile(r"\{prev_(\d+)\}")


def _primary_values(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, (int, float, str)):
        return [val]
    if isinstance(val, dict):
        if not val:
            return []
        for k, v in val.items():
            if isinstance(v, str):
                return [v]
        return [next(iter(val.values()))]
    if isinstance(val, list):
        if not val:
            return []
        if all(isinstance(v, (int, float, str)) for v in val):
            return list(val)
        if all(isinstance(v, dict) for v in val):
            first = val[0]
            primary_col = None
            for k, v in first.items():
                if isinstance(v, str):
                    primary_col = k
                    break
            if primary_col is None:
                primary_col = list(first.keys())[0]
            return [r.get(primary_col) for r in val]
    return [val]


def _substitute_in_dict(d: Dict[str, Any], prev_results: Dict[int, Any]
                        ) -> Dict[str, Any]:
    """Recursively substitute `{prev_N}` / `{prev_N.col}` placeholders inside
    a dict's STRING values. Used for `metadata_filter` on dense_retrieve
    steps so the planner can write e.g. `{"buyer_name": "{prev_1.buyer_name}"}`
    and have hop-1's resolved value bound at execution time. Non-string
    values pass through unchanged. Empty/None substitutions drop the key
    entirely (avoids a NULL-valued filter that returns 0 hits)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            sub = _substitute_placeholders(v, prev_results, for_sql=False)
            if sub and sub != "NULL":
                out[k] = sub.strip(",").strip()
        elif isinstance(v, dict):
            inner = _substitute_in_dict(v, prev_results)
            if inner:
                out[k] = inner
        else:
            out[k] = v
    return out


def _substitute_placeholders(template: str, prev_results: Dict[int, Any],
                             for_sql: bool) -> str:
    def _format_values(vals: List[Any]) -> str:
        if for_sql:
            return ",".join("'" + str(v).replace("'", "''") + "'" for v in vals)
        return ",".join(str(v) for v in vals)

    def _dotted_repl(m: re.Match) -> str:
        sid = int(m.group(1))
        col = m.group(2)
        v = prev_results.get(sid)
        vals: List[Any] = []
        if isinstance(v, list):
            for r in v:
                if isinstance(r, dict) and col in r:
                    vals.append(r[col])
        elif isinstance(v, dict) and col in v:
            vals.append(v[col])
        return _format_values(vals) or "NULL"

    def _plain_repl(m: re.Match) -> str:
        sid = int(m.group(1))
        v = prev_results.get(sid)
        vals = _primary_values(v)
        return _format_values(vals) or "NULL"

    out = _RE_DOTTED.sub(_dotted_repl, template)
    out = _RE_PLAIN.sub(_plain_repl, out)
    return out


# ----------------- failure detection -----------------


def _is_failure(out: Dict[str, Any], min_top_score: float = 0.2) -> Tuple[bool, str]:
    status = (out.get("status") or "").upper()
    # Routing-contract violation: the SQL referenced a table whose store is
    # not declared in this step's `sources`. Diagnose / revise treats this as
    # a separate cause so the planner can EXPAND the source list rather than
    # rewrite the SQL into a smaller subset.
    if status.startswith("SQL ROUTING ERROR"):
        return True, "target_sources_too_narrow"
    if status.startswith("SQL ERROR") or status.startswith("ERROR"):
        return True, "sql_error"
    if status.startswith("INDEX MISSING") or status.startswith("SUBSTITUTE_ERROR"):
        return True, "infra_error"
    output = out.get("output")
    action = (out.get("action") or "").lower()
    if action == "sql_query":
        # SQL: empty rows = filter_too_narrow / empty_upstream
        if isinstance(output, list) and len(output) == 0:
            return True, "filter_too_narrow"
        return False, ""
    if action == "dense_retrieve":
        if not output:
            return True, "query_too_specific"
        # Off-topic: top hit score below threshold (cosine sim).
        try:
            top_score = float(output[0].get("score", 0.0))
            if top_score < min_top_score:
                return True, "query_too_generic"
        except Exception:
            pass
        return False, ""
    return False, ""


# ----------------- trace -----------------


@dataclass
class PlanReviseTrace:
    query: str
    method: str = "retailrag"
    initial_plan: Dict[str, Any] = field(default_factory=dict)
    plan_versions: List[Dict[str, Any]] = field(default_factory=list)
    step_outputs: List[Dict[str, Any]] = field(default_factory=list)
    diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    revisions_used: int = 0
    answer: Optional[str] = None
    confidence: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    intermediate: Dict[str, Any] = field(default_factory=dict)


def trace_to_dict(t: PlanReviseTrace) -> Dict[str, Any]:
    return asdict(t)


# ----------------- agent -----------------


class PlanReviseAgent:
    """DB-block driven agent with routing, plan-revise and multi-modal failure recovery."""

    def __init__(
        self,
        db_block: str,
        sql_tool: Optional[SQLTool] = None,
        retrieve_tool: Optional[RetrieveTool] = None,
        cfg: Optional[Config] = None,
        max_revisions: int = 10,
        per_step_revisions: int = 3,
        retrieve_top_k: int = 5,
        retrieve_min_score: float = 0.2,
        adaptive_routing: bool = True,
        enable_plan_retry: bool = True,
        retry_trigger_mode: str = "basic",   # "basic" | "expanded"
    ) -> None:
        self.cfg = cfg or DEFAULT_CONFIG
        self.sql_tool = sql_tool
        self.retrieve_tool = retrieve_tool
        self.max_revisions = max_revisions
        self.per_step_revisions = per_step_revisions
        self.top_k = retrieve_top_k
        self.min_score = retrieve_min_score
        self.adaptive_routing = adaptive_routing
        self.enable_plan_retry = enable_plan_retry
        self.retry_trigger_mode = retry_trigger_mode

        # Build runtime prompts from db_block (one-shot per agent).
        self.db_block = db_block
        self.plan_system = PLAN_TEMPLATE.format(db_block=db_block)
        self.diagnose_system = DIAGNOSE_TEMPLATE  # no db_block needed
        self.revise_system = REVISE_TEMPLATE.format(db_block=db_block)
        self.answer_system = ANSWER_TEMPLATE

    # ------------------------------------------------------------------ public

    def answer(self, query: str) -> PlanReviseTrace:
        trace = PlanReviseTrace(query=query)
        t0 = time.time()
        try:
            # Adaptive complexity routing: cheap LLM classifier gates the
            # heavy plan-revise loop. 1-hop lookups go through a single SQL
            # call (mirrors Naive Text-to-SQL but reuses SQLTool's safety
            # nets). Skips 2 LLM calls (plan + final answer) and avoids
            # the over-decomposition that hurt 1-hop on the internal split.
            label: Optional[str] = None
            if self.adaptive_routing and self.sql_tool is not None:
                # Text-only fast path: questions that read like asks about
                # free-form passages (a review's wording, an inquiry's
                # dialogue, a broadcast's pitch) should NOT enter the SQL
                # plan-revise loop, which empirically over-decomposes them
                # into long SQL chains and never recovers. We detect such
                # questions with a cheap lexical cue check and route them
                # straight to a single dense_retrieve.
                if self.retrieve_tool is not None and \
                   self._looks_text_only(query):
                    trace.diagnoses.append({"text_fast_path": True})
                    fast_trace = self._answer_text_only(query, trace, t0)
                    if fast_trace is not None:
                        return fast_trace
                    trace.diagnoses.append({"escalated_from": "text_fast"})

                label = self._classify_complexity(query)
                trace.diagnoses.append({"complexity": label})
                # simple → naive single-SQL fast path.
                # hard → straight to plan-revise.
                if label == "simple":
                    fast_trace = self._answer_simple(query, trace, t0)
                    if fast_trace is not None:
                        return fast_trace
                    trace.diagnoses.append({"escalated_from": label})

            # First plan-revise attempt at default temperature.
            self._plan_revise_attempt(query, trace, base_temperature=0.2)

            # Plan retry: if first attempt looks suspicious, regenerate the
            # plan with higher temperature and try again. Best-of-2: pick
            # the better answer between the two attempts via a heuristic
            # comparator (avoids regression when retry produces worse output).
            if self.enable_plan_retry and self._should_retry(query, trace):
                # Snapshot first attempt
                a1 = {
                    "answer": trace.answer,
                    "evidence": list(trace.evidence or []),
                    "confidence": trace.confidence,
                    "step_outputs": list(trace.step_outputs or []),
                    "plan_versions": list(trace.plan_versions or []),
                    "revisions_used": trace.revisions_used,
                }
                trace.diagnoses.append({"plan_retry": "triggered",
                                        "trigger_mode": self.retry_trigger_mode})
                trace.step_outputs = []
                trace.plan_versions.append({"retry_marker": True})
                self._plan_revise_attempt(query, trace, base_temperature=0.5)
                # Best-of-2: prefer higher-quality answer
                a2 = {
                    "answer": trace.answer,
                    "evidence": list(trace.evidence or []),
                    "confidence": trace.confidence,
                    "step_outputs": list(trace.step_outputs or []),
                    "plan_versions": list(trace.plan_versions or []),
                    "revisions_used": trace.revisions_used,
                }
                pick = self._select_better(query, a1, a2)
                if pick is a1:
                    trace.answer = a1["answer"]
                    trace.evidence = a1["evidence"]
                    trace.confidence = a1["confidence"]
                    trace.step_outputs = a1["step_outputs"]
                    trace.revisions_used = a1["revisions_used"]
                    trace.diagnoses.append({"best_of_2": "kept_attempt_1"})
                else:
                    trace.diagnoses.append({"best_of_2": "kept_attempt_2"})
        except Exception as e:
            import traceback
            trace.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        trace.timings["total_s"] = round(time.time() - t0, 3)
        return trace

    def close(self) -> None:
        for tool in (self.sql_tool, self.retrieve_tool):
            try:
                if hasattr(tool, "close"):
                    tool.close()
            except Exception:
                pass

    @staticmethod
    def _is_failed_answer(ans: Optional[str]) -> bool:
        """Heuristic: plan-revise produced an unusable answer."""
        if ans is None:
            return True
        s = str(ans).strip()
        if not s:
            return True
        upper = s.upper()
        if upper in ("INSUFFICIENT", "NULL", "NONE", "N/A"):
            return True
        return False

    _ID_PATTERN = re.compile(r"^[A-Z]{2,}-?[A-Z0-9]+$")
    _NAME_INTENT_RE = re.compile(
        r"\b(who|which\s+(seller|buyer|product|customer|user|name)|"
        r"what\s+is\s+the\s+name|name\s+of)\b", re.IGNORECASE)

    def _should_retry(self, query: str, trace: "PlanReviseTrace") -> bool:
        """Decide whether to fire the plan-retry second attempt.

        basic    — fail-marker answer only (INSUFFICIENT/None/empty).
        expanded — Tier 1 (mechanical fail signals) + Tier 2 (semantic
                   suspicion: low confidence, ID-shaped answer when query
                   asks for a name, all-zero numeric answer).
        """
        ans = trace.answer
        # Tier 0 — basic fail markers (always retry on these).
        if self._is_failed_answer(ans):
            return True
        if self.retry_trigger_mode != "expanded":
            return False

        # Tier 1 — mechanical fail signals
        last = (trace.step_outputs or [])[-1] if trace.step_outputs else None
        if last is not None:
            status = (last.get("status") or "").upper()
            output = last.get("output")
            if status.startswith("SQL ERROR") or status.startswith("SQL ROUTING ERROR"):
                return True
            if isinstance(output, list) and len(output) == 0:
                return True
        if (trace.revisions_used or 0) >= self.max_revisions:
            # Used up the entire revise budget — likely not happy with anything.
            return True

        # Tier 2 — semantic suspicion
        if (trace.confidence or "").lower() == "low":
            return True
        s = str(ans).strip()
        # ID-shaped answer when the query asks for a name
        if self._NAME_INTENT_RE.search(query) and self._ID_PATTERN.match(s):
            return True
        # Numeric 0 or null when query implies a positive count/value
        try:
            num = float(s)
            if num == 0.0 and "how many" in query.lower():
                # 0 may be correct sometimes, but it's a frequent failure
                # mode for entity-based queries; treat as suspicious.
                return True
        except (TypeError, ValueError):
            pass
        return False

    @staticmethod
    def _confidence_rank(c: Optional[str]) -> int:
        return {"high": 2, "medium": 1, "low": 0}.get((c or "").lower(), 1)

    def _select_better(self, query: str, a: Dict[str, Any],
                       b: Dict[str, Any]) -> Dict[str, Any]:
        """Pick the better of two attempts. Used after plan-retry to avoid
        regression when retry produces a worse answer than the first try.
        """
        a_fail = self._is_failed_answer(a.get("answer"))
        b_fail = self._is_failed_answer(b.get("answer"))
        if a_fail and not b_fail:
            return b
        if b_fail and not a_fail:
            return a
        if a_fail and b_fail:
            return a  # both failed — keep first
        # Self-consistency: same answer → keep first (cheaper)
        if str(a.get("answer")) == str(b.get("answer")):
            return a
        # Confidence comparison
        ar = self._confidence_rank(a.get("confidence"))
        br = self._confidence_rank(b.get("confidence"))
        if ar > br:
            return a
        if br > ar:
            return b
        # Tie — prefer the attempt with fewer revisions (cleaner solution)
        if (a.get("revisions_used") or 0) <= (b.get("revisions_used") or 0):
            return a
        return b

    def _plan_revise_attempt(self, query: str, trace: "PlanReviseTrace",
                             base_temperature: float = 0.2) -> None:
        """One full plan → execute → revise → answer pass. Mutates `trace`.

        Run twice with different temperatures to implement plan-retry.
        """
        plan = self._plan(query, base_temperature=base_temperature)
        trace.initial_plan = dict(plan)
        trace.plan_versions.append(dict(plan))

        steps: List[Dict[str, Any]] = list(plan.get("steps") or [])
        outputs: List[Dict[str, Any]] = []
        revisions = 0
        per_step_used: Dict[int, int] = {}
        i = 0
        iter_cap = 16
        iters = 0
        while i < len(steps) and iters < iter_cap:
            iters += 1
            step = steps[i]
            step_id = int(step.get("step") or i + 1)
            out = self._execute(step, outputs)
            failed, reason = _is_failure(out, min_top_score=self.min_score)
            step_used = per_step_used.get(step_id, 0)
            if (failed
                    and revisions < self.max_revisions
                    and step_used < self.per_step_revisions):
                revisions += 1
                per_step_used[step_id] = step_used + 1
                diag = self._diagnose(step, out, reason)
                trace.diagnoses.append({"at_step": step_id,
                                        "auto_reason": reason,
                                        **diag})
                new_steps = self._revise(query, steps, outputs, i, diag, out)
                if new_steps:
                    steps = steps[:i] + new_steps
                    trace.plan_versions.append({"steps": steps})
                    continue
                outputs.append(out)
                i += 1
                continue
            outputs.append(out)
            i += 1

        trace.step_outputs = list(outputs)
        trace.revisions_used = revisions

        ans = self._answer(query, outputs)
        trace.answer = ans.get("answer")
        trace.evidence = ans.get("evidence", []) or []
        trace.confidence = ans.get("confidence")

    # ------------------------------------------------------------------ adaptive routing

    def _classify_complexity(self, query: str) -> str:
        """One cheap LLM call to gate adaptive routing (simple vs complex)."""
        try:
            res = call_llm_json(
                COMPLEXITY_SYSTEM,
                f"Query: {query}\n\nClassify.",
                temperature=0.0, max_retries=1,
            )
            label = (res.get("label") or "hard").strip().lower()
            if label in {"simple", "easy"}:
                return "simple"
            if label in {"medium", "hard"}:
                # A "medium" label is treated as "hard": the single-SQL fast
                # path does not benefit from a separate tier.
                return "hard" if label == "hard" else "simple"
            return "hard"
        except Exception:
            return "hard"

    # ---------------------------------------------------------------- text fast path

    # Lexical cues that strongly suggest the answer lives inside free-form
    # narrative text rather than a structured column. If a query contains
    # ANY of these phrases AND no obvious aggregation/rank keyword, we send
    # it directly to dense_retrieve instead of the SQL plan-revise loop.
    _TEXT_HINT_RE = re.compile(
        r"\b(said|say|says|saying|stated|states|describe[sd]?|"
        r"described|mention(?:s|ed)?|explain(?:s|ed)?|note(?:s|d)?|"
        r"according to|"
        r"in (?:the|a|an|his|her)\s+(?:\w+\s+){0,3}"
        r"(?:review|inquiry|dialogue|broadcast|script|conversation|"
        r"message|comment|feedback|complaint|post|reply|response)|"
        r"what (?:did|does) [A-Z][^?]+ (?:say|write|think|describe|"
        r"explain|note|mention)|"
        r"what (?:concerns?|issues?|complaints?|feedback|opinions?|"
        r"comments?|reactions?|quotes?|words?|phrases?))\b",
        re.IGNORECASE,
    )
    _AGG_RE = re.compile(
        r"\b(how many|count|total|sum|average|avg|mean|maximum|max|"
        r"minimum|min|highest|lowest|most|least|top \d|ranked?|"
        r"per |group(?:ed)? by)\b",
        re.IGNORECASE,
    )

    def _looks_text_only(self, query: str) -> bool:
        if not self._TEXT_HINT_RE.search(query):
            return False
        # Avoid hijacking a query that explicitly asks for an aggregate or
        # rank — that is still SQL territory even if narrative words appear.
        if self._AGG_RE.search(query):
            return False
        return True

    def _pick_text_source(self, query: str) -> str:
        ql = query.lower()
        if any(k in ql for k in ("inquiry", "inquiries", "conversation",
                                  "dialogue", "complaint", "support", "ask",
                                  "asked", "asking")):
            return "inquiries_text"
        if any(k in ql for k in ("broadcast", "script", "live", "stream",
                                  "pitch", "host")):
            return "sales_scripts"
        return "reviews_text"

    def _answer_text_only(self, query: str,
                          trace: "PlanReviseTrace",
                          t0: float) -> Optional["PlanReviseTrace"]:
        """Single dense_retrieve + answer synthesis. Used for questions whose
        answer is in free-form narrative text (a review's opinion, an
        inquiry's resolution, a broadcast's pitch). Skips SQL plan-revise
        entirely, which empirically over-decomposes such queries and
        produces wrong intermediates."""
        source = self._pick_text_source(query)
        try:
            res = self.retrieve_tool.call(
                query=query, source=source, top_k=self.top_k,
            )
        except Exception:
            return None
        hits = res.get("results") or []
        if not hits:
            return None

        out = {
            "step": 1, "sub_query": query, "action": "dense_retrieve",
            "tool_result": res, "output": hits,
            "status": res.get("status"),
        }
        trace.step_outputs = [out]
        trace.plan_versions = [{"steps": [{
            "step": 1, "sub_query": query, "action": "dense_retrieve",
            "source": source,
        }]}]
        trace.initial_plan = trace.plan_versions[0]
        trace.revisions_used = 0

        # Reuse the global answer synthesizer (now metadata-aware via the
        # updated ANSWER_TEMPLATE) — it will read the doc.metadata fields
        # for entity-typed answer slots.
        synth = self._answer(query, trace.step_outputs)
        ans = synth.get("answer")
        # Escalate to plan-revise when the single-shot retrieval cannot
        # ground an answer (INSUFFICIENT, empty, or low confidence). This
        # is the common failure mode for composite L2/L3 questions whose
        # query embedding is too entity-dense to retrieve a single
        # passage — the SQL plan-revise loop is better suited for those.
        if not ans or str(ans).strip().upper() == "INSUFFICIENT":
            return None
        if (synth.get("confidence") or "").lower() == "low":
            return None
        trace.answer = ans
        trace.confidence = synth.get("confidence") or "medium"
        trace.evidence = synth.get("evidence") or [
            {"step": 1, "source": source,
             "snippet": (hits[0].get("content") or "")[:200]}
        ]
        trace.timings["total_s"] = round(time.time() - t0, 3)
        return trace

    def _answer_simple(self, query: str, trace: "PlanReviseTrace",
                       t0: float) -> Optional["PlanReviseTrace"]:
        """Single-shot SQL path for queries the classifier marked 'simple'.

        Bypasses plan-revise. Uses SQLTool's auto source picker + built-in
        repair (zero-rows relax, schema-error retry). Answer is formatted
        deterministically (mirroring naive_sql) — no LLM formatter, which
        avoids column-misselection and date/float reformatting errors that
        cost ~6pt vs naive_sql on the same SQL outputs.

        Returns None on hard failure (SQL ERROR / empty rows) so the caller
        can escalate to the full plan-revise loop.
        """
        res = self.sql_tool.call(
            sub_query=query, previous_steps=None, target_sources=None,
            temperature=0.0,
        )
        rows = res.get("rows") or []
        status = (res.get("status") or "").upper()
        # Hard failure → escalate. SQLTool already retried internally (zero-
        # rows relax, schema-error rewrite); if we still have nothing usable
        # the simple path cannot recover.
        if status.startswith("SQL ERROR") or status.startswith("SQL ROUTING ERROR"):
            return None
        if not rows:
            return None

        out = {
            "step": 1, "sub_query": query, "action": "sql_query",
            "tool_result": res, "output": rows,
            "status": res.get("status"),
        }
        trace.step_outputs = [out]
        trace.plan_versions = [{"steps": [{
            "step": 1, "sub_query": query, "action": "sql_query",
            "sources": None,
        }]}]
        trace.initial_plan = trace.plan_versions[0]
        trace.revisions_used = 0
        trace.answer = self._format_rows(rows)
        trace.evidence = [{"step": 1, "source": "simple_sql",
                           "snippet": str(res.get("sql"))[:200]}]
        trace.confidence = "medium"
        trace.timings["total_s"] = round(time.time() - t0, 3)
        return trace

    @staticmethod
    def _format_rows(rows: List[Dict[str, Any]]) -> str:
        """Deterministic row-to-answer formatter (matches naive_sql)."""
        if not rows:
            return "INSUFFICIENT"
        if len(rows) == 1 and len(rows[0]) == 1:
            return str(next(iter(rows[0].values())))
        keys = list(rows[0].keys())
        if len(keys) == 1:
            return ", ".join(str(r[keys[0]]) for r in rows)
        return "; ".join(", ".join(f"{k}={v}" for k, v in r.items())
                         for r in rows[:5])

    # ------------------------------------------------------------------ stages

    def _plan(self, query: str, base_temperature: float = 0.2) -> Dict[str, Any]:
        user = (
            f"User query: {query}\n\n"
            "Step 1 of your thinking: list every named entity in the query "
            "(persons, dates, long product/work names) before deciding "
            "the plan order. Resolution steps for those entities come FIRST.\n"
            "Output the plan JSON only (no entity-list section)."
        )
        for attempt in range(2):
            try:
                temp = base_temperature if attempt == 0 else max(0.1, base_temperature - 0.1)
                raw = call_llm(self.plan_system, user, temperature=temp)
                obj = self._extract_plan(raw)
                if obj.get("steps"):
                    return obj
            except Exception:
                pass
        # Fallback: 1-step plan with the user query as a free-text retrieve.
        return {
            "steps": [{"step": 1, "sub_query": query, "action": "dense_retrieve",
                       "source": None, "sql_template": None,
                       "search_template": query, "depends_on": None}],
            "rationale": "fallback (plan parse failed)",
        }

    @staticmethod
    def _extract_plan(raw: str) -> Dict[str, Any]:
        m = re.search(r"```json\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
        candidate = m.group(1) if m else raw
        if "steps" not in candidate:
            start = raw.find("{"); end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw[start: end + 1]
        try:
            obj = json.loads(candidate)
        except Exception:
            obj = json.loads(candidate.strip().strip("`"))
        valid_actions = {"sql_query", "dense_retrieve"}
        steps = [s for s in (obj.get("steps") or [])
                 if (s.get("action") or "").lower() in valid_actions]
        for k, s in enumerate(steps):
            s["step"] = k + 1
        obj["steps"] = steps
        return obj

    def _execute(self, step: Dict[str, Any], prior: List[Dict[str, Any]]
                 ) -> Dict[str, Any]:
        action = (step.get("action") or "").lower()
        sub_query = step.get("sub_query", "")
        sid = step.get("step")
        prev_results = {p.get("step"): p.get("output") for p in prior}

        if action == "sql_query":
            sql_template = step.get("sql_template")
            if not sql_template or self.sql_tool is None:
                return {"step": sid, "sub_query": sub_query, "action": action,
                        "tool_result": {"error": "no_sql_tool_or_template"},
                        "output": None, "status": "INFRA_ERROR: no SQL tool"}
            try:
                sql = _substitute_placeholders(sql_template, prev_results, for_sql=True)
            except Exception as e:
                return {"step": sid, "sub_query": sub_query, "action": action,
                        "tool_result": {"error": f"substitute failed: {e}"},
                        "output": None, "status": f"SUBSTITUTE_ERROR: {e}"}
            # Resolve the routing contract for this step. Plans may use
            # either ``sources`` (preferred — list, multi-source allowed
            # within the call) or ``source`` (a single store).
            step_sources = step.get("sources")
            if step_sources is None and step.get("source"):
                step_sources = [step.get("source")]
            try:
                rows, status = self.sql_tool.db.execute(
                    sql, target_sources=step_sources,
                )
            except Exception as e:
                return {"step": sid, "sub_query": sub_query, "action": action,
                        "tool_result": {"sql": sql, "error": str(e)},
                        "output": None, "status": f"SQL ERROR: {e}"}
            if (status.startswith("SQL ERROR")
                    or status.startswith("SQL ROUTING ERROR")
                    or (status.startswith("SQL OK") and not rows)):
                # Fall back to SQLTool's repair logic. Pass the same routing
                # contract so the LLM rewrites within the allowed sources.
                res = self.sql_tool.call(
                    sub_query=sub_query, previous_steps=prior,
                    target_sources=step_sources,
                )
                return {"step": sid, "sub_query": sub_query, "action": action,
                        "tool_result": res, "output": res.get("rows"),
                        "status": res.get("status", status)}
            return {"step": sid, "sub_query": sub_query, "action": action,
                    "tool_result": {"sql": sql}, "output": rows,
                    "status": status}

        if action == "dense_retrieve":
            template = step.get("search_template") or sub_query
            try:
                template = _substitute_placeholders(template, prev_results, for_sql=False)
            except Exception:
                pass
            if self.retrieve_tool is None:
                return {"step": sid, "sub_query": sub_query, "action": action,
                        "tool_result": {"error": "no_retrieve_tool"},
                        "output": None, "status": "INFRA_ERROR: no retriever"}
            source = step.get("source")
            # metadata_filter resolves prior-step entity bindings into hard
            # restrictions on the candidate pool — see PLAN_TEMPLATE RULE 3b.
            # An empty dict after substitution means every key resolved to
            # NULL; we drop the filter rather than send an unsatisfiable
            # constraint to the retriever.
            raw_filter = step.get("metadata_filter") or {}
            md_filter = None
            if isinstance(raw_filter, dict) and raw_filter:
                try:
                    md_filter = _substitute_in_dict(raw_filter, prev_results) or None
                except Exception:
                    md_filter = None
            try:
                res = self.retrieve_tool.call(
                    query=template,
                    source=source if source else (
                        getattr(self.retrieve_tool, "available_sources", ["corpus"])[0]
                    ),
                    top_k=self.top_k,
                    filter_metadata=md_filter,
                )
            except Exception as e:
                return {"step": sid, "sub_query": sub_query, "action": action,
                        "tool_result": {"error": str(e)},
                        "output": None, "status": f"ERROR: {e}"}
            return {"step": sid, "sub_query": sub_query, "action": action,
                    "tool_result": res, "output": res.get("results"),
                    "status": res.get("status", "OK")}

        return {"step": sid, "sub_query": sub_query, "action": action,
                "tool_result": {"error": f"unknown action {action}"},
                "output": None, "status": "INFRA_ERROR: unknown action"}

    @staticmethod
    def _build_diagnose_user(step: Dict[str, Any], out: Dict[str, Any],
                             auto_reason: str) -> str:
        """User prompt for the diagnose call."""
        return (
            f"Step: {json.dumps(step, ensure_ascii=False)}\n"
            f"action: {step.get('action')}\n"
            f"output_size: {len(out.get('output') or []) if isinstance(out.get('output'), list) else 'scalar'}\n"
            f"status: {out.get('status')}\n"
            f"auto_reason: {auto_reason}"
        )

    def _diagnose(self, step: Dict[str, Any], out: Dict[str, Any],
                  auto_reason: str) -> Dict[str, Any]:
        ctx = self._build_diagnose_user(step, out, auto_reason)
        try:
            return call_llm_json(self.diagnose_system, ctx,
                                 temperature=0.0, max_retries=1)
        except Exception:
            return {"cause": auto_reason or "other",
                    "explanation": "diagnose call failed"}

    @staticmethod
    def _build_revise_user(query: str, steps: List[Dict[str, Any]],
                           prior: List[Dict[str, Any]], i: int,
                           diag: Dict[str, Any]) -> str:
        """User prompt for the revise call."""
        prior_compact = [
            {"step": p.get("step"), "sub_query": p.get("sub_query"),
             "top": str(p.get("output"))[:120]}
            for p in prior
        ]
        return (
            f"User query: {query}\n"
            f"Failed step (index {i+1}): {json.dumps(steps[i], ensure_ascii=False)}\n"
            f"Diagnose: {diag}\n"
            f"Prior outputs (compact): {json.dumps(prior_compact, ensure_ascii=False)[:1200]}\n"
            f"Original full plan: {json.dumps(steps, ensure_ascii=False)[:1500]}"
        )

    def _revise(self, query: str, steps: List[Dict[str, Any]],
                prior: List[Dict[str, Any]], i: int,
                diag: Dict[str, Any], failed_out: Dict[str, Any]
                ) -> Optional[List[Dict[str, Any]]]:
        ctx = self._build_revise_user(query, steps, prior, i, diag)
        try:
            res = call_llm_json(self.revise_system, ctx,
                                temperature=0.1, max_retries=1)
            new_steps = res.get("new_remaining_steps") or []
            for k, s in enumerate(new_steps):
                s["step"] = i + 1 + k
            return new_steps
        except Exception:
            return None

    # ----------------- answer with deterministic plural short-circuit -----------------

    _PLURAL_CUES = (
        r"\bwhich\s+(products?|categories|buyers?|sellers?|reviews?|scripts?|"
        r"texts?|inquiries|orders?|items?|tiers?|ratings?|names?)\b",
        r"\bwhat\s+(products?|categories|buyers?|sellers?|reviews?|scripts?|"
        r"texts?|inquiries|orders?|items?|tiers?|ratings?|names?)\b",
        r"\bwhat\s+are\s+the\s+",
        r"\bwhat\s+is\s+the\s+(spending\s+tier|rating\s+category)",
        r"\b(list|all|every|each)\b",
        r"\bfor\s+each\b",
        r"\b(buyers?|products?|categories|sellers?)\s+who\b",
    )

    @classmethod
    def _is_plural_query(cls, query: str) -> bool:
        q = query.lower()
        for s in (r"\bthe\s+highest\b", r"\bthe\s+most\b", r"\bthe\s+single\b",
                  r"\bwhich\s+one\b", r"\bthe\s+one\b"):
            if re.search(s, q):
                if re.search(r"\bfor\s+each\b", q):
                    return True
                return False
        for p in cls._PLURAL_CUES:
            if re.search(p, q):
                return True
        return False

    @staticmethod
    def _flatten_step_rows(out: Any) -> List[Any]:
        if out is None:
            return []
        if isinstance(out, (int, float, str)):
            return [out]
        if isinstance(out, dict):
            for k, v in out.items():
                if isinstance(v, str):
                    return [v]
            return [next(iter(out.values()))] if out else []
        if isinstance(out, list):
            if not out:
                return []
            if all(isinstance(v, (int, float, str)) for v in out):
                return list(out)
            if all(isinstance(v, dict) for v in out):
                first = out[0]
                primary_col = None
                for k, v in first.items():
                    if isinstance(v, str):
                        primary_col = k; break
                if primary_col is None:
                    primary_col = list(first.keys())[0]
                return [r.get(primary_col) for r in out
                        if r.get(primary_col) is not None]
        return []

    @staticmethod
    def _looks_like_id(s: str) -> bool:
        # Match single-hyphen IDs (e.g. "FOO-1234", "S9") and multi-segment
        # IDs ("FOO-INQ-POS-0317"), so that `_deterministic_plural_answer`
        # never emits a raw identifier as a user-facing answer.
        return bool(re.match(r'^[A-Z]{2,}(?:-[A-Z0-9]+)+$', s.strip())
                    or re.match(r'^[A-Z]\d+$', s.strip()))

    def _deterministic_plural_answer(
        self, query: str, outputs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not self._is_plural_query(query):
            return None
        for s in reversed(outputs):
            out = s.get("output")
            # Dense-retrieve outputs are NEVER safe for the deterministic
            # plural shortcut: `_flatten_step_rows` would surface the first
            # string field (doc_id), which is internal plumbing, not a
            # user-facing answer. Force these through the LLM synthesizer
            # which knows to read metadata names / content prose instead.
            if (s.get("action") or "").lower() == "dense_retrieve":
                return None
            vals = self._flatten_step_rows(out)
            if len(vals) >= 2:
                str_vals = [str(v) for v in vals]
                if all(isinstance(v, str) and self._looks_like_id(v)
                       for v in str_vals):
                    return None
                joined = ", ".join(str_vals)
                return {"answer": joined,
                        "evidence": [{"step": s.get("step"),
                                      "source": "deterministic_plural",
                                      "snippet": f"{len(str_vals)} rows"}],
                        "confidence": "high"}
            if len(vals) == 1:
                return None
        return None

    @staticmethod
    def _build_answer_user(query: str, outputs: List[Dict[str, Any]]) -> str:
        """Serialize executed step outputs into the answer-synth user prompt."""
        ctx_lines: List[str] = []
        for s in outputs:
            sid = s.get("step")
            intent = s.get("sub_query") or ""
            out = s.get("output")
            action = s.get("action") or ""
            if isinstance(out, list):
                head = f"{len(out)} row(s)"
            elif out is None:
                head = "(no output)"
            elif isinstance(out, dict):
                head = "object"
            else:
                head = "scalar"
            # dense_retrieve: metadata first (entity names are often the
            # answer), content capped per-doc; OMIT *_id fields (the LLM
            # otherwise echoes internal ids as the user-facing answer).
            if action == "dense_retrieve" and isinstance(out, list) and out:
                compact = []
                for d in out[:5]:
                    if not isinstance(d, dict):
                        continue
                    md = d.get("metadata") or {}
                    if isinstance(md, dict):
                        md = {k: v for k, v in md.items()
                              if not (isinstance(k, str) and k.endswith("_id"))}
                    compact.append({
                        "metadata": md,
                        "content":  (d.get("content") or "")[:700],
                    })
                preview = json.dumps(compact, ensure_ascii=False, default=str)[:8000]
            else:
                preview = json.dumps(out, ensure_ascii=False, default=str)[:1500]
            ctx_lines.append(f"[step {sid}] INTENT: {intent}\n"
                             f"          STATUS: {s.get('status', '')}\n"
                             f"          OUTPUT ({head}): {preview}")
        return (f"User query: {query}\n\nExecuted plan:\n"
                + "\n".join(ctx_lines)
                + "\n\nProduce the final JSON answer.")

    def _answer(self, query: str, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        plural = self._deterministic_plural_answer(query, outputs)
        if plural is not None:
            return plural
        user = self._build_answer_user(query, outputs)
        try:
            return call_llm_json(self.answer_system, user,
                                 temperature=0.1, max_retries=1)
        except Exception as e:
            return {"answer": None, "evidence": [], "confidence": "low",
                    "error": str(e)}
