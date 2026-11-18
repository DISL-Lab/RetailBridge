"""Unified SQL connection over the six sqlite files.

QA outputs reference table names like ``Order``, ``Script``, ``Inquiry``
without DB-prefix, so we attach all sqlite files into one connection and
expose those names directly. ``Order`` is reserved in SQLite, so we wrap it
in double quotes as the benchmark's gold queries do.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..config import HIDDEN_TABLES, HIDE_USER_LOG, SQL_FILES, SQL_SOURCES


# Map table-name -> the attached schema (alias) it lives under. Each sqlite
# file is attached using its key from SQL_FILES, so e.g. the table ``Product``
# inside product.sqlite becomes ``product.Product``. We then create a view
# in the main schema with the bare name so both ``Product`` and
# ``product.Product`` resolve.
TABLE_TO_FILE: Dict[str, str] = {
    "Product":       "product",
    "Script":        "script",
    "Order":         "order",
    "Inquiry":       "inquiry",
    "Review":        "review",
    "Buyer_Metric":  "user_log",
    "Seller_Metric": "user_log",
    "Seller":        "user_log",
}


_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def _is_safe_select(sql: str) -> bool:
    s = sql.strip().rstrip(";").strip()
    if not s:
        return False
    if ";" in s:
        return False
    if _FORBIDDEN.search(s):
        return False
    head = s.lstrip().split(None, 1)[0].upper()
    return head in {"SELECT", "WITH"}


def _resolve_sql_source(name: str) -> str | None:
    """Map a declared store name to a canonical SQL_SOURCES key, or None.
    Accepts keys ("orders"), table names ("Order"), case and plural variants."""
    if not name:
        return None
    table_to_key = {v: k for k, v in SQL_SOURCES.items()}
    if name in SQL_SOURCES:
        return name
    if name in table_to_key:
        return table_to_key[name]
    low = name.lower()
    lower_key = {k.lower(): k for k in SQL_SOURCES}
    lower_tbl = {v.lower(): k for k, v in SQL_SOURCES.items()}
    for cand in (low, low + "s", low.rstrip("s")):
        if cand in lower_key:
            return lower_key[cand]
        if cand in lower_tbl:
            return lower_tbl[cand]
    return None


def _sources_referenced_in_sql(sql: str) -> List[str]:
    """Canonical SQL_SOURCES keys whose table the SQL text references."""
    found = []
    for key, tbl in SQL_SOURCES.items():
        if re.search(rf'(?<![A-Za-z0-9_])"?{re.escape(tbl)}"?(?![A-Za-z0-9_])', sql):
            found.append(key)
    return found


def normalize_target_sources(target_sources, sql: str) -> Tuple[List[str], str]:
    """Repair malformed store declarations before the routing check.

    Malformed declarations seen in practice: a generic backend name instead of a
    table ("sql", "sqlite", "SQL", "main", "backend"), a schema prefix
    ("sqlite.Review", "sqlite:Review", "product.Product"), several tables in one
    string ("Order, Script"), a bare string that callers iterate per character
    (['I','n','q',...]), and leftover header brackets ("[SQL").
    Names that still do not resolve are replaced by the SQL stores the query
    actually references. Returns (sources, note); note is "" when nothing changed.
    """
    original = target_sources
    if isinstance(target_sources, str):
        items = [target_sources]
    else:
        items = [str(x) for x in (target_sources or []) if x is not None]
        if items and all(len(x) == 1 for x in items):  # a string split into characters
            items = ["".join(items)]
    parts: List[str] = []
    for raw in items:
        s = re.sub(r"^\s*\[?(?:SQL|TEXT)\]?\s+", "", raw.strip())
        s = s.strip().strip("[]\"'` ")
        for p in re.split(r"[,/&+]|\band\b", s):
            p = p.strip().strip("[]\"'` ")
            if p:
                parts.append(p)
    resolved: List[str] = []
    unresolved: List[str] = []
    for p in parts:
        key = _resolve_sql_source(p)
        if key is None and re.search(r"[.:]", p):
            key = _resolve_sql_source(re.split(r"[.:]", p)[-1].strip())
        if key is None:
            unresolved.append(p)
        elif key not in resolved:
            resolved.append(key)
    if unresolved:
        for key in _sources_referenced_in_sql(sql):
            if key not in resolved:
                resolved.append(key)
        if not resolved:
            # The SQL names no real table either; skip the routing check so the
            # query runs and returns the real SQLite error (e.g. "no such table").
            return [], f" [routing normalized: {list(items)} -> []]"
    canonical_in = [_resolve_sql_source(x) for x in items] if not isinstance(original, str) else [None]
    if not unresolved and canonical_in == resolved:
        return resolved, ""
    return resolved, f" [routing normalized: {list(items)} -> {resolved}]"


class SQLDB:
    """Read-only multi-attach SQLite wrapper."""

    def __init__(self, hide_user_log: bool = HIDE_USER_LOG) -> None:
        """If ``hide_user_log`` is True, the user_log file is not attached and
        its tables are NOT exposed — any SQL referring to ``Buyer_Metric``,
        ``Seller_Metric``, ``Seller`` will fail with "no such table". This
        forces the agent to derive those KPIs from base tables.
        """
        self.hide_user_log = hide_user_log
        # check_same_thread=False lets the same SQLDB instance be shared
        # across worker threads. The Lock below serializes cursor use
        # so concurrent execute() calls don't corrupt cursor state.
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.row_factory = sqlite3.Row

        for alias, path in SQL_FILES.items():
            if self.hide_user_log and alias == "user_log":
                # Don't even attach — SQLite would otherwise fall back to
                # `attached.table` lookup and the hide would leak.
                continue
            if not Path(path).exists():
                raise FileNotFoundError(f"missing sqlite file: {path}")
            self.conn.execute(f'ATTACH DATABASE \'{path}\' AS "{alias}"')

        self._create_bare_name_views()

    def _create_bare_name_views(self) -> None:
        """Expose every visible table under its bare name in main schema.

        ``HIDDEN_TABLES`` are skipped when ``hide_user_log`` is on so the
        agent's plain-name SQL queries fail for those tables.
        """
        cur = self.conn.cursor()
        for table, alias in TABLE_TO_FILE.items():
            if self.hide_user_log and table in HIDDEN_TABLES:
                continue
            quoted = f'"{table}"'
            cur.execute(
                f'CREATE TEMP VIEW IF NOT EXISTS {quoted} AS '
                f'SELECT * FROM "{alias}".{quoted}'
            )
        self.conn.commit()

    # ------------------------------------------------------------------ API

    def execute(self, sql: str, params: Iterable[Any] = (),
                target_sources: Iterable[str] | None = None
                ) -> Tuple[List[Dict[str, Any]], str]:
        """Execute a SELECT and return (rows_as_dicts, status_string).

        If ``target_sources`` is provided, the SQL is validated to only
        reference tables belonging to those logical sources. This enforces
        the per-call routing contract: the agent must explicitly declare
        which DBs it wants per call, and any reference to a non-selected
        table fails with a routing error before execution.

        ``target_sources`` ⊆ keys of SQL_SOURCES (e.g. {'orders','reviews'}).
        ``target_sources=None`` (default) leaves every ATTACH'd table
        visible. Set it explicitly to enforce routing.
        """
        if not _is_safe_select(sql):
            return [], "SQL REJECTED (only single SELECT/WITH statements are allowed)"

        routing_note = ""
        if target_sources is not None:
            target_sources, routing_note = normalize_target_sources(target_sources, sql)
            valid_keys = set(SQL_SOURCES.keys())
            # Accept either canonical source keys ("orders") OR the table
            # names they back ("Order"). Also tolerate the casing+plural
            # variants the LLM tends to emit ("order", "scripts").
            table_to_key = {v: k for k, v in SQL_SOURCES.items()}
            lower_key = {k.lower(): k for k in valid_keys}
            lower_tbl = {v.lower(): k for k, v in SQL_SOURCES.items()}
            wanted: set = set()
            unknown: list = []
            for raw in target_sources:
                # Strip "[SQL] " / "[TEXT] " prefix some models (notably GPT-5.4)
                # copy verbatim from the db_block section headers.
                stripped = re.sub(r"^\s*\[(?:SQL|TEXT)\]\s+", "", raw or "")
                if stripped in valid_keys:
                    wanted.add(stripped)
                    continue
                if stripped in table_to_key:
                    wanted.add(table_to_key[stripped])
                    continue
                low = stripped.lower()
                if low in lower_key:
                    wanted.add(lower_key[low])
                    continue
                if low in lower_tbl:
                    wanted.add(lower_tbl[low])
                    continue
                # also try singular ↔ plural toggle
                if low + "s" in lower_key:
                    wanted.add(lower_key[low + "s"])
                    continue
                if low.rstrip("s") in lower_key:
                    wanted.add(lower_key[low.rstrip("s")])
                    continue
                unknown.append(raw)
            if unknown:
                return [], (f"SQL ROUTING ERROR: unknown target_sources={unknown} "
                            f"(valid: {sorted(valid_keys)})")
            allowed_tables = {SQL_SOURCES[s] for s in wanted}
            forbidden_tables = {SQL_SOURCES[s] for s in valid_keys - wanted}
            # bare-name reference, optionally double-quoted
            for tbl in forbidden_tables:
                pat = re.compile(rf'(?<![A-Za-z0-9_])"?{re.escape(tbl)}"?(?![A-Za-z0-9_])')
                if pat.search(sql):
                    return [], (f"SQL ROUTING ERROR: query references table '{tbl}' "
                                f"but target_sources={sorted(wanted)} only allows "
                                f"{sorted(allowed_tables)}. Add the source to "
                                "target_sources or rewrite the query.")

        try:
            with self._lock:
                cur = self.conn.cursor()
                cur.execute(sql, tuple(params))
                rows = [dict(r) for r in cur.fetchall()]
            return rows, f"SQL OK ({len(rows)} rows){routing_note}"
        except sqlite3.Error as e:
            return [], f"SQL ERROR: {e}"

    def schema(self, source_alias: str) -> List[str]:
        """Return column names for one of the logical sources in SQL_SOURCES."""
        table = SQL_SOURCES.get(source_alias)
        if not table:
            raise KeyError(f"unknown SQL source: {source_alias}")
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(f'PRAGMA table_info("{table}")')
            return [r["name"] for r in cur.fetchall()]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
