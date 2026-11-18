"""sql_query tool: NL sub-query → SQL → exec on SQLDB.

The Executor calls this with a natural-language sub-query and the upstream
context (results from previous steps). The tool asks the LLM to produce a
single SELECT statement against the schema, then runs it.

Per-call routing contract: every call MUST declare ``target_sources`` —
a non-empty subset of {orders, products, scripts, inquiries, reviews}.
The SQL is validated to only reference tables in that subset; cross-DB
JOIN is allowed within selected sources but not outside. This forces every
caller to make an explicit "which DBs to fetch" decision per call rather
than silently leaning on a unified ATTACH endpoint.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..config import DEFAULT_CONFIG, SCHEMA_HINTS, SQL_SOURCES
from ..databases import SQLDB
from ..llm import call_llm, call_llm_json


# Pipe-separated, one table per line. Compact but still complete.
SCHEMA_BLOCK = "\n".join(
    f"  {SQL_SOURCES[alias]}({'|'.join(cols)})"
    for alias, cols in SCHEMA_HINTS.items()
    if alias in SQL_SOURCES
)


def _schema_block_for(target_sources: Sequence[str]) -> str:
    """Render only the schema lines for the selected sources."""
    keep = set(target_sources)
    return "\n".join(
        f"  {SQL_SOURCES[alias]}({'|'.join(cols)})"
        for alias, cols in SCHEMA_HINTS.items()
        if alias in SQL_SOURCES and alias in keep
    ) or "  (no sources selected)"


# ---------------- source picker (multi-select over 5 SQL DBs) ----------------


_SQL_PICKER_SYSTEM = """You decide which retail SQL databases must be queried to answer a sub-question.

Available databases (logical names, each backed by a separate sqlite file):
  • products   — table Product(product_id, product_name, product_category, original_price, store, features)
  • scripts    — table Script(script_id, seller_id, seller_name, product_id, product_name, broadcast_datetime, discount_rate, original_price, discount_price, discount_end_date, shipping_fee)
  • orders     — table "Order"(order_id, script_id, buyer_id, buyer_name, product_id, product_name, quantity, total_price, status, order_date, delivery_date, post_inquiry_type, seller_id, seller_name)
  • inquiries  — table Inquiry(inquiry_id, inquiry_type, buyer_id, buyer_name, seller_id, seller_name, product_id, product_name, script_id)
  • reviews    — table Review(review_id, review_category, review_rating, buyer_id, buyer_name, seller_id, seller_name, product_id, product_name, script_id, inquiry_id)

Rules:
  - Pick the MINIMUM set that can answer the sub-question. Do NOT pick all five "to be safe".
  - If two tables must be JOIN'd (e.g. products + orders), include both.
  - Output JSON: {"sources": ["orders", "reviews"], "why": "one short reason"}
  - sources MUST be a non-empty subset of the five names above.
"""


def pick_sql_sources(sub_query: str, fallback: Optional[List[str]] = None,
                     temperature: float = 0.0) -> List[str]:
    """LLM-driven source-subset selector. Returns a list of valid SQL source
    names (subset of SQL_SOURCES keys). On parse failure, returns ``fallback``
    (default: all visible sources)."""
    valid = set(SQL_SOURCES.keys())
    fallback = fallback if fallback is not None else sorted(valid)
    try:
        out = call_llm_json(_SQL_PICKER_SYSTEM,
                            f"Sub-question: {sub_query}\n\nOutput JSON.",
                            temperature=temperature, max_retries=1)
        cand = out.get("sources") or []
        if isinstance(cand, str):
            cand = [cand]
        cleaned = [s for s in cand if isinstance(s, str) and s in valid]
        if cleaned:
            # Dedup, preserve order.
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


_GENERIC_SQL_PROMPT = """SQL writer.

Tables and columns are described in the SCHEMA brief below. Use the EXACT
names listed there — case-sensitive, no plural / singular flips.

Generic rules:
  • ONE SELECT (or WITH...SELECT). No DDL/DML.
  • If a table name is a SQL reserved word, quote it with double-quotes
    (NOT backticks).
  • Use only listed columns. Always spell out the full column name from
    the schema after a table alias — single-letter or two-letter
    abbreviations after a dot are never valid.
  • Cross-table → JOIN on shared id columns (whatever IDs appear in both).
  • Return ONLY SQL inside ```sql ... ```.

Filter precision:
  • DATETIME: never match exact seconds/minutes. Use one of:
        DATE(col) = 'YYYY-MM-DD'
        col LIKE 'YYYY-MM-DD%'
        col LIKE 'YYYY-MM-DD HH:%'
  • LONG STRING FILTERS (long strings with delimiters or punctuation):
    use col LIKE '%distinctive-keyword%' on 1-3 most unique words instead
    of exact equality.
  • SHORT NAMES / IDs: use exact match.
  • If the question references a SPECIFIC entity, your WHERE clause MUST
    filter by that entity.

================================================================
SCHEMA
================================================================
{schema_block}
================================================================
"""

_ADAPTIVE_SQL_PROMPT = """SQL writer.

Tables, enums, and auto-derived KPI formulas are in the DATA STORES brief
below. Use the EXACT names listed there — case-sensitive, no plural /
singular flips.

Generic rules:
  • ONE SELECT (or WITH...SELECT). No DDL/DML.
  • If a table name is a SQL reserved word, quote it with double-quotes
    (NOT backticks).
  • Use only listed columns. Always spell out the full column name from
    the schema after a table alias — single-letter or two-letter
    abbreviations after a dot are never valid.
  • Cross-table → JOIN on shared id columns (whatever IDs appear in both).
  • Use enum values exactly as listed (case + spacing). Do not paraphrase.
  • Return ONLY SQL inside ```sql ... ```.

Filter precision:
  • DATETIME: never match exact seconds/minutes. Use one of:
        DATE(col) = 'YYYY-MM-DD'
        col LIKE 'YYYY-MM-DD%'
        col LIKE 'YYYY-MM-DD HH:%'
  • LONG STRING FILTERS (long strings with delimiters or punctuation):
    use col LIKE '%distinctive-keyword%' on 1-3 most unique words instead
    of exact equality.
  • SHORT NAMES / IDs: use exact match.
  • If the question references a SPECIFIC entity, your WHERE clause MUST
    filter by that entity.

KPI derivation:
  • Use the auto-derived KPI formulas in [Derived KPIs] VERBATIM,
    substituting only the entity literal. Do not invent your own metric
    definitions.
  • For "rank-N by KPI" queries, compute the KPI per entity in a CTE then
    ORDER BY the KPI with LIMIT N.

================================================================
DATA STORES (auto-derived; trust these names verbatim)
================================================================
{db_block}
================================================================
"""


# Module-level default = GENERIC: no retail-specific KPI / enum hardcoding,
# only the schema and generic SQL-writer rules. RetailRAG overrides this at
# construction with the DB-adaptive prompt (make_adaptive_sql_prompt).
SYSTEM_PROMPT = _GENERIC_SQL_PROMPT.format(schema_block=SCHEMA_BLOCK)


def make_adaptive_sql_prompt(db_block: str) -> str:
    """Build the DB-adaptive SQL prompt from the auto-derived db_block.

    This is RetailRAG's framework feature: the agent gets the full
    [Derived KPIs] section so it can use canonical KPI formulas verbatim.
    Baselines stay on the generic prompt to keep the comparison honest.
    """
    return _ADAPTIVE_SQL_PROMPT.format(db_block=db_block)


_SQL_FENCE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_sql(text: str) -> str:
    m = _SQL_FENCE.search(text)
    if m:
        return m.group(1).strip().rstrip(";")
    # Fallback: assume the whole thing is SQL.
    return text.strip().strip("`").rstrip(";")


def _format_context(previous: Optional[List[Dict[str, Any]]]) -> str:
    if not previous:
        return "No prior context."
    blob = []
    for step in previous:
        sub = step.get("sub_query", "")
        out = step.get("output")
        # Truncate big outputs.
        if isinstance(out, list):
            out_str = json.dumps(out[:8], ensure_ascii=False, default=str)
            if len(out) > 8:
                out_str += f"  (+{len(out)-8} more rows)"
        else:
            out_str = json.dumps(out, ensure_ascii=False, default=str)[:1000]
        blob.append(f"Step {step.get('step', '?')} ({sub}): {out_str}")
    return "\n".join(blob)


class SQLTool:
    """Stateful wrapper that holds the open SQLDB connection."""

    def __init__(self, db: Optional[SQLDB] = None,
                 system_prompt: Optional[str] = None) -> None:
        """``system_prompt`` overrides the SQL writer prompt for this tool
        instance. Baselines instantiate ``SQLTool()`` with the module
        default (GENERIC schema-only). RetailRAG passes
        ``make_adaptive_sql_prompt(db_block)`` so the agent can use the
        auto-derived KPI formulas verbatim.
        """
        self.db = db or SQLDB()
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def call(
        self,
        sub_query: str,
        previous_steps: Optional[List[Dict[str, Any]]] = None,
        target_source: Optional[str] = None,
        target_sources: Optional[Sequence[str]] = None,
        temperature: float = 0.1,
        disable_auto_repair: bool = False,
    ) -> Dict[str, Any]:
        # A single ``target_source`` is accepted as a one-element list.
        if target_sources is None and target_source:
            target_sources = [target_source]
        # If no subset declared, auto-pick via LLM. This makes the routing
        # decision explicit even when the caller didn't choose one.
        if target_sources is None:
            target_sources = pick_sql_sources(sub_query)
        target_sources = list(target_sources)

        focus = (f"\nROUTING: This call is restricted to target_sources="
                 f"{target_sources}. Use ONLY tables from these sources:\n"
                 f"{_schema_block_for(target_sources)}\n"
                 "Cross-DB JOIN is allowed only across the selected sources.")

        user_prompt = (
            f"Sub-question: {sub_query}\n"
            f"{focus}\n"
            f"Upstream context:\n{_format_context(previous_steps)}\n\n"
            "Write the SQL now."
        )
        raw = call_llm(self.system_prompt, user_prompt, temperature=temperature)
        sql = _extract_sql(raw)
        rows, status = self.db.execute(sql, target_sources=target_sources)

        # Strict path: skip every auto-repair branch and return the first
        # attempt verbatim — even on 0 rows / SQL errors.
        if disable_auto_repair:
            return {
                "tool": "sql_query",
                "sql": sql,
                "rows": rows,
                "status": status,
                "raw_llm": raw,
            }

        # Auto-repair on ZERO ROWS: a deep-chain step's first SQL frequently
        # returns 0 rows because of an over-strict filter (full equality on a
        # long product name, exact-minute datetime, etc.). One LLM call to
        # rewrite with LIKE / DATE() variants almost always recovers the
        # entity. Triggers ONLY when status is OK and rows is empty —
        # SQL ERROR is handled by the next branch.
        if status.startswith("SQL OK") and not rows:
            relax_user = (
                f"Sub-question: {sub_query}\n"
                f"{focus}\n"
                f"Upstream context:\n{_format_context(previous_steps)}\n\n"
                f"Your previous SQL ran fine but returned 0 rows:\n```sql\n{sql}\n```\n\n"
                "Likely cause: an entity literal in the WHERE clause is too "
                "strict. Rewrite ONCE with relaxed entity matching:\n"
                "  - product_name: replace full equality with "
                "`LIKE '%distinctive_keyword%'` using the most unique 1–3 "
                "words from the name.\n"
                "  - broadcast_datetime / order_date: use `LIKE 'YYYY-MM-DD%'` "
                "or `DATE(col) = 'YYYY-MM-DD'`. Never match on exact minutes.\n"
                "  - person name: confirm spelling; if uncertain, try "
                "`LIKE 'first_initial%'`.\n"
                "Keep the rest of the WHERE clause; only relax the literal "
                "matchers. Return ONLY the SQL inside ```sql ... ```."
            )
            raw_r = call_llm(self.system_prompt, relax_user, temperature=0.0)
            sql_r = _extract_sql(raw_r)
            rows_r, status_r = self.db.execute(sql_r)
            if not status_r.startswith("SQL ERROR") and rows_r:
                return {
                    "tool": "sql_query",
                    "sql": sql_r,
                    "rows": rows_r,
                    "status": status_r,
                    "raw_llm": raw_r,
                    "relaxed_from": sql,
                }
            # Fallthrough: use original 0-rows result so the agent can
            # diagnose+revise as before.

        # Auto-repair on schema errors: most often a hallucinated column name
        # ("no such column: o.b"). Show the LLM its own broken SQL plus the
        # exact error, ask it to rewrite once, and use whichever ran.
        if status.startswith("SQL ERROR"):
            repair_user = (
                f"Sub-question: {sub_query}\n"
                f"{focus}\n"
                f"Upstream context:\n{_format_context(previous_steps)}\n\n"
                f"Your previous SQL ran but failed:\n```sql\n{sql}\n```\n"
                f"DB error: {status}\n\n"
                "Rewrite the SQL. The most likely cause is a column name "
                "that is not in the schema (often a one- or two-letter "
                "abbreviation after a table alias). Spell out the FULL "
                "column name from the schema block. Return ONLY the SQL "
                "inside ```sql ... ```."
            )
            raw2 = call_llm(self.system_prompt, repair_user, temperature=0.0)
            sql2 = _extract_sql(raw2)
            rows2, status2 = self.db.execute(sql2)
            if not status2.startswith("SQL ERROR"):
                return {
                    "tool": "sql_query",
                    "sql": sql2,
                    "rows": rows2,
                    "status": status2,
                    "raw_llm": raw2,
                    "repaired_from": sql,
                    "original_error": status,
                }
            # Both attempts errored — return the second attempt's failure
            # so the agent's diagnose/revise sees the latest, but keep
            # diagnostic context.
            return {
                "tool": "sql_query",
                "sql": sql2,
                "rows": rows2,
                "status": status2,
                "raw_llm": raw2,
                "repaired_from": sql,
                "original_error": status,
            }

        return {
            "tool": "sql_query",
            "sql": sql,
            "rows": rows,
            "status": status,
            "raw_llm": raw,
        }
