"""DB-block composer.

Given the available data stores (a SQLite/PostgreSQL connection and/or a
list of dense text retrievers), produce a short planner-friendly summary
that the PLAN_SYSTEM prompt injects via the ``{db_block}`` placeholder.

The composer runs ONCE per dataset/connection set:
    1. Inspect each store cheaply (SQL: SHOW TABLES + 3 sample rows;
       Text: 5 sample passages).
    2. Pass the raw inspection to a single LLM call with a strict
       instruction template.
    3. Cache the resulting block on disk (``.db_block.txt``) so runtime
       cost is paid once per dataset.

The block is intentionally small (≤ 2K chars) and *generic* — no
KPI formulas, no enum lists. Domain-specific knowledge is supplied
separately via an optional ``kb_text`` argument (free-form, e.g. the
retail KPI base-table formulas).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..llm import call_llm


DB_BLOCK_SYSTEM = """You produce a planner brief from an inspection JSON of one or more data stores. The brief is read by a downstream SQL planner at every query. Output plain text (no JSON, no fences); use exact store / table / column / value names from the inspection — never invent.

Required sections, in this order:

[Stores]
  For each store, one line: "[SQL] <id>  — short description" or "[TEXT] <id>  — short description". Under each SQL store, list every column with one of: "<col> ∈ {v1, v2, ...}" (enum, exact distinct_values), "<col> (numeric)", "<col> (e.g. <s1>, <s2>, <s3>)" (freeform, sample_values), or "<col> (id-shaped)" (high-cardinality identifier). Under each TEXT store, one sentence covering topic + distinctive vocabulary.

[Enums]
  One line per categorical column: <table>.<col> ∈ {v1, v2, ...}

[Entity-grouping columns]
  PRIMARY: columns ending in _name / _label / _title that appear in 2+ tables. SECONDARY: columns ending in _id that appear in 2+ tables (join keys, paired with the corresponding _name column when one exists).
  Format:
    PRIMARY:
      - <col>  (in: <table1>, <table2>, ...)
    SECONDARY:
      - <id_col>  (in: <tables>)  paired with <name_col>

[Derived KPIs]
  SQL formulas auto-derived from the schema. Use the 4 templates below; substitute placeholders with real names from the inspection. Each KPI is scoped to ONE most-natural entity column — do not replicate the same KPI across every entity column.

  CRITICAL: in every template, <entity_col> MUST come from the PRIMARY list in [Entity-grouping columns], AND MUST be a column that actually exists in the <table> being aggregated. It MUST NOT be the same column as <enum_col> or <numeric_col> being aggregated. For example, "refunded_rate(status) = ... WHERE status='<X>'" is wrong because the filter column equals the enum column; "avg_quantity(product_category) = AVG(quantity) FROM Order WHERE product_category='<X>'" is wrong because Order has no product_category column. Emit "refunded_rate(seller_name) = ... WHERE seller_name='<X>'" instead.

  1. ENUM-RATE — one KPI per enum value:
       <value>_rate(<entity_col>) =
         SUM(CASE WHEN <enum_col>='<value>' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
         FROM <table> WHERE <entity_col>='<X>'

  2. NUMERIC AGGREGATE — avg / total / max / min per numeric column:
       avg_<numeric_col>(<entity_col>)   = AVG(<numeric_col>)   FROM <table> WHERE <entity_col>='<X>'
       total_<numeric_col>(<entity_col>) = SUM(<numeric_col>)   FROM <table> WHERE <entity_col>='<X>'
       max_<numeric_col>(<entity_col>)   = MAX(<numeric_col>)   FROM <table> WHERE <entity_col>='<X>'
       min_<numeric_col>(<entity_col>)   = MIN(<numeric_col>)   FROM <table> WHERE <entity_col>='<X>'

  3. NUMERIC × ENUM CROSS — emit only when semantically meaningful:
       avg_<numeric_col>_<enum_value>(<entity_col>) = AVG(<numeric_col>) FROM <table>
         WHERE <entity_col>='<X>' AND <enum_col>='<enum_value>'

  4. CROSS-TABLE CONVERSION — when two tables share a foreign-key column:
       conversion_rate(<entity_col>) =
         COUNT(DISTINCT child.<shared_id>) * 100.0 / COUNT(DISTINCT parent.<shared_id>)
         FROM <parent_table> parent LEFT JOIN <child_table> child
           ON parent.<shared_id> = child.<shared_id>
         WHERE parent.<entity_col>='<X>'

[Conventions]
  - Date columns: filter with LIKE 'YYYY-MM-DD%' or DATE(col)='YYYY-MM-DD'.
  - Long freeform string columns: use LIKE '%keyword%' on 1-3 distinctive words.
  - For "rank-N by KPI" queries: compute the KPI per entity in a CTE, then ORDER BY ... LIMIT N. Do not pre-filter by "above average".
  - Use enum values verbatim from [Enums]; do not paraphrase or pluralize.
"""


# --------------- inspection helpers ---------------


_ENUM_DETECTION_LIMIT = 25     # column counts as categorical only if ≤ this many distinct
_NUMERIC_HINT_NAMES = (
    "rating", "score", "price", "amount", "quantity", "qty",
    "count", "total", "discount", "rate",
)


def _is_numeric_string(s: str) -> bool:
    """True if `s` parses as an int or float (covers negatives, decimals,
    sentinel anomaly values like -4, 109, 254). Used to reject enum
    classification for columns whose distinct values are all numbers."""
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _column_profile(conn: sqlite3.Connection, schema: str, table: str,
                    col: str, total_rows: int) -> Dict[str, Any]:
    """Inspect a single column. Three kinds:

      numeric  — column-name hint (rating/score/price/amount/quantity/…).
                 Checked FIRST so that a numeric column with few distinct
                 values (e.g. quantity ∈ {1,2,3,4,5}) is treated as
                 numeric, not as an enum. Previously such columns got
                 misclassified, polluting [Enums] and [Derived KPIs].
      enum     — distinct count ≤ _ENUM_DETECTION_LIMIT and values are
                 short (≤ 40 chars). All distinct values are returned so
                 the LLM can write per-value KPIs (e.g. <value>_rate).
                 Skipped for numeric-hint columns.
      freeform — high-cardinality strings/text. Returns 3 representative
                 sample values so the LLM understands the column's shape
                 (entity name vs free text vs id-shaped) without flooding
                 the prompt.
    """
    qualified = f'"{schema}"."{table}"' if schema != "main" else f'"{table}"'
    cur = conn.cursor()
    try:
        cur.execute(
            f'SELECT COUNT(DISTINCT "{col}") FROM {qualified}'
        )
        n_distinct = (cur.fetchone() or [0])[0]
    except Exception:
        return {"kind": "freeform"}

    # Numeric hint wins over enum detection — true numeric columns should
    # be aggregated (AVG/SUM/MAX/MIN), not enumerated as discrete labels.
    name_is_numeric = any(h in col.lower() for h in _NUMERIC_HINT_NAMES)
    if name_is_numeric:
        return {"kind": "numeric", "name_hint": True}

    if (n_distinct and n_distinct <= _ENUM_DETECTION_LIMIT and
            (total_rows == 0 or n_distinct < max(2, total_rows // 2))):
        try:
            cur.execute(
                f'SELECT DISTINCT "{col}" FROM {qualified} '
                f'WHERE "{col}" IS NOT NULL '
                f'LIMIT {_ENUM_DETECTION_LIMIT}'
            )
            vals = [r[0] for r in cur.fetchall()]
            short_vals = [v for v in vals
                          if v is not None and len(str(v)) <= 40]
            # Reject "enum" if values look numeric (e.g. ratings or sentinel
            # codes that slipped through the name-hint check above).
            looks_numeric = (
                short_vals
                and all(_is_numeric_string(str(v)) for v in short_vals)
            )
            if (short_vals
                    and not looks_numeric
                    and len(short_vals) >= max(1, len(vals) // 2)):
                return {"kind": "enum",
                        "distinct_values": [str(v) for v in short_vals]}
        except Exception:
            pass
    # Freeform: pull 3 sample values so the LLM can recognize shape
    # (id-shaped, name, free text). Truncate each sample at 60 chars to
    # keep the inspection JSON compact.
    try:
        cur.execute(
            f'SELECT DISTINCT "{col}" FROM {qualified} '
            f'WHERE "{col}" IS NOT NULL LIMIT 3'
        )
        samples = [str(r[0])[:60] for r in cur.fetchall()
                   if r[0] is not None]
    except Exception:
        samples = []
    return {"kind": "freeform", "n_distinct": n_distinct,
            "sample_values": samples}


def _sample_sqlite(conn: sqlite3.Connection, max_tables: int = 30,
                   max_rows: int = 3) -> Dict[str, Any]:
    """Inspect tables across the main DB AND any ATTACH'd DBs.

    Our project's SQLDB attaches one .sqlite per source (Product, Script,
    Order, Inquiry, Review), so tables live in the attached schema
    namespaces — sqlite_master on `main` is empty. We enumerate all
    schemas via ``PRAGMA database_list`` and fetch each schema's tables.

    For each column we also profile it as enum / numeric / freeform so the
    composer can derive useful metrics (refund_rate, avg_rating, …) without
    domain hand-coding.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA database_list")
    schemas = [(r[1], r[2]) for r in cur.fetchall()]  # (name, file)
    out: Dict[str, Any] = {"backend": "sqlite", "tables": []}
    seen = 0
    for sch_name, _file in schemas:
        if sch_name in ("temp",):
            continue
        try:
            cur.execute(
                f'SELECT name FROM "{sch_name}".sqlite_master '
                f"WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            t_names = [r[0] for r in cur.fetchall()]
        except Exception:
            t_names = []
        for t in t_names:
            if seen >= max_tables:
                break
            qualified = f'"{sch_name}"."{t}"' if sch_name != "main" else f'"{t}"'
            try:
                cur.execute(f'PRAGMA "{sch_name}".table_info("{t}")')
                cols = [r[1] for r in cur.fetchall()]
                cur.execute(f'SELECT * FROM {qualified} LIMIT {max_rows}')
                rows = cur.fetchall()
                # Cheap row count for enum-coverage sanity check.
                try:
                    cur.execute(f'SELECT COUNT(*) FROM {qualified}')
                    total_rows = (cur.fetchone() or [0])[0]
                except Exception:
                    total_rows = 0
                col_profiles: Dict[str, Dict[str, Any]] = {}
                for c in cols:
                    col_profiles[c] = _column_profile(
                        conn, sch_name, t, c, total_rows)
                out["tables"].append({
                    "name": t,
                    "schema": sch_name,
                    "columns": cols,
                    "n_rows": total_rows,
                    "sample_rows": [list(map(_truncate_cell, r)) for r in rows],
                    "column_profiles": col_profiles,
                })
            except Exception as e:
                out["tables"].append({"name": t, "schema": sch_name,
                                      "error": str(e)})
            seen += 1
    return out


def _truncate_cell(v: Any, n: int = 80) -> str:
    s = "" if v is None else str(v)
    return s if len(s) <= n else s[:n] + "…"


@dataclass
class TextStoreInspection:
    """User-provided description for a text store + a few samples."""
    name: str
    role: str = ""           # short hint, e.g. "broadcast scripts"
    n_docs: Optional[int] = None
    sample_passages: List[str] = field(default_factory=list)


# --------------- main composer ---------------


def make_db_block(
    sql_conn: Optional[sqlite3.Connection] = None,
    text_stores: Optional[Sequence[TextStoreInspection]] = None,
    kb_text: str = "",
    cache_path: Optional[str | Path] = None,
    force_rebuild: bool = False,
) -> str:
    """Compose (or load from cache) the DB block for the planner.

    Returns a plain-text block ≤ ~2K chars suitable for direct injection
    into PLAN_SYSTEM via the ``{db_block}`` placeholder.
    """
    text_stores = list(text_stores or [])

    # Build inspection payload.
    inspection: Dict[str, Any] = {"sql": None, "text_stores": []}
    if sql_conn is not None:
        try:
            inspection["sql"] = _sample_sqlite(sql_conn)
        except Exception as e:
            inspection["sql"] = {"backend": "sqlite", "error": str(e)}
    for ts in text_stores:
        inspection["text_stores"].append({
            "name": ts.name,
            "role": ts.role,
            "n_docs": ts.n_docs,
            "samples": [s[:300] for s in ts.sample_passages[:5]],
        })

    # Cache key from inspection content + kb hash.
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_rebuild:
            return cache_path.read_text(encoding="utf-8")
    key = hashlib.md5(
        json.dumps(inspection, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + kb_text.encode("utf-8")
    ).hexdigest()

    user_prompt = (
        "Stores inspection (raw JSON):\n"
        + json.dumps(inspection, ensure_ascii=False, indent=2)
        + ("\n\nDomain knowledge (optional, append verbatim under header):\n"
           + kb_text if kb_text else "")
        + "\n\nProduce the brief now. List EVERY table that appears in "
          "the inspection — do not skip any."
    )
    raw = call_llm(DB_BLOCK_SYSTEM, user_prompt, temperature=0.1,
                   max_tokens=12000)
    block = raw.strip()
    # Light post-processing.
    block = re.sub(r"```[a-zA-Z]*\n?", "", block).replace("```", "")
    # Drop lines containing CJK / Hangul / Hiragana / Katakana / Cyrillic /
    # Arabic / etc. Qwen occasionally injects mid-identifier hallucinations
    # like "product游戏副本" that would crash SQL when copied verbatim.
    # Whitelist common safe non-ASCII (math/em-dash/curly quotes/ellipsis).
    cjk_re = re.compile(r"[　-ヿ㐀-䶿一-鿿"
                        r"가-힯Ѐ-ӿ؀-ۿ]")
    cleaned: List[str] = [ln for ln in block.split("\n")
                          if not cjk_re.search(ln)]
    block = "\n".join(cleaned).strip()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(block, encoding="utf-8")
    return block


# --------------- convenience: build block from paths only ---------------


def _sample_text_file(path: Path, max_passages: int = 5,
                      max_chars_per_passage: int = 300) -> List[str]:
    """Sample up to N passages from a text-store file. Auto-detects jsonl
    (`{"id":..., "contents":...}`) and plain text (one passage per line)."""
    samples: List[str] = []
    if not path.exists():
        return samples
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_passages:
                break
            line = line.strip()
            if not line:
                continue
            text = ""
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    text = (d.get("contents") or d.get("content")
                            or d.get("text") or d.get("passage")
                            or "")
                else:
                    text = str(d)
            except json.JSONDecodeError:
                text = line  # plain-text line
            if text:
                samples.append(text[:max_chars_per_passage])
    return samples


def build_db_block_from_paths(
    sql_paths: Optional[Sequence[str | Path]] = None,
    text_paths: Optional[Sequence[str | Path]] = None,
    kb_text: str = "",
    cache_path: Optional[str | Path] = None,
    force_rebuild: bool = False,
    text_role_hint: Optional[Dict[str, str]] = None,
) -> str:
    """Generic entry — caller supplies just file paths; this function loads
    them, samples them, and composes the planner's DB block.

    sql_paths    : list of `.sqlite` files. Each is ATTACH'd to an in-memory
                   connection under an alias = its filename stem.
    text_paths   : list of jsonl / txt corpus files. Auto-samples 5 passages
                   from each. Store name = filename stem.
    kb_text      : optional domain-knowledge text (appended verbatim).
    cache_path   : disk path; cached block is reused unless force_rebuild.
    text_role_hint : optional {filename_stem: role_string} to override the
                     default role label.
    """
    sql_paths = [Path(p) for p in (sql_paths or [])]
    text_paths = [Path(p) for p in (text_paths or [])]
    text_role_hint = text_role_hint or {}

    # ---- SQL: ATTACH each .sqlite onto an in-memory main connection ----
    conn: Optional[sqlite3.Connection] = None
    if sql_paths:
        conn = sqlite3.connect(":memory:")
        for p in sql_paths:
            if not p.exists():
                continue
            alias = re.sub(r"[^A-Za-z0-9_]", "_", p.stem) or "db"
            try:
                conn.execute(
                    f'ATTACH DATABASE "{p.as_posix()}" AS "{alias}"'
                )
            except Exception as e:
                # Skip unreadable file but keep the rest.
                print(f"[db_block] warning: failed to attach {p}: {e}")

    # ---- Text: read first 5 passages from each file ----
    text_stores: List[TextStoreInspection] = []
    for p in text_paths:
        samples = _sample_text_file(p)
        # Cheap doc-count: line count (works for jsonl and txt).
        n_docs: Optional[int] = None
        try:
            with p.open("rb") as f:
                n_docs = sum(1 for _ in f)
        except Exception:
            pass
        name = p.stem
        text_stores.append(TextStoreInspection(
            name=name,
            role=text_role_hint.get(name, f"text passages from '{p.name}'"),
            n_docs=n_docs,
            sample_passages=samples,
        ))

    return make_db_block(
        sql_conn=conn,
        text_stores=text_stores,
        kb_text=kb_text,
        cache_path=cache_path,
        force_rebuild=force_rebuild,
    )


# --------------- format-auto entry: caller passes paths, we sniff types ---------------


_SQL_EXTS = {".sqlite", ".sqlite3", ".db"}
_TEXT_EXTS = {".jsonl", ".txt"}
_CSV_EXTS = {".csv", ".tsv"}


def _csv_to_inmem_sqlite(conn: sqlite3.Connection, path: Path,
                         max_rows_per_table: int = 100000) -> str:
    """Load a CSV/TSV into the given SQLite connection as a new table.
    Returns the alias the table was attached under (== filename stem)."""
    import csv as _csv
    alias = re.sub(r"[^A-Za-z0-9_]", "_", path.stem) or "tbl"
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = _csv.reader(f, delimiter=delim)
        try:
            header = next(reader)
        except StopIteration:
            return alias
        cols = [re.sub(r"[^A-Za-z0-9_]", "_", h.strip()) or f"c{i}"
                for i, h in enumerate(header)]
        # Make column names unique.
        seen: Dict[str, int] = {}
        unique_cols: List[str] = []
        for c in cols:
            n = seen.get(c, 0)
            unique_cols.append(c if n == 0 else f"{c}_{n}")
            seen[c] = n + 1
        cur = conn.cursor()
        cur.execute(
            f'CREATE TABLE IF NOT EXISTS "{alias}" '
            "(" + ", ".join(f'"{c}" TEXT' for c in unique_cols) + ")"
        )
        placeholders = ", ".join("?" for _ in unique_cols)
        rows_in = 0
        for row in reader:
            if rows_in >= max_rows_per_table:
                break
            if len(row) < len(unique_cols):
                row = row + [""] * (len(unique_cols) - len(row))
            elif len(row) > len(unique_cols):
                row = row[: len(unique_cols)]
            cur.execute(
                f'INSERT INTO "{alias}" VALUES ({placeholders})', row
            )
            rows_in += 1
    conn.commit()
    return alias


def build_db_block_from_files(
    paths: Sequence[str | Path],
    kb_text: str = "",
    cache_path: Optional[str | Path] = None,
    force_rebuild: bool = False,
    text_role_hint: Optional[Dict[str, str]] = None,
) -> str:
    """One-shot DB block from a flat list of file paths. The function
    auto-dispatches by extension:

      .sqlite / .sqlite3 / .db   →  ATTACH onto an in-memory SQLite
      .csv / .tsv                →  loaded as a table in the same SQLite
      .jsonl / .txt              →  treated as a TEXT store (sample 5)
      .json                      →  if top-level is a list of dicts → loaded
                                    as a table; otherwise sampled as TEXT

    Unknown extensions are treated as plain-text stores (best effort).
    """
    paths = [Path(p) for p in paths]
    sql_paths: List[Path] = []
    text_paths: List[Path] = []
    csv_paths: List[Path] = []
    json_table_paths: List[Path] = []

    for p in paths:
        ext = p.suffix.lower()
        if ext in _SQL_EXTS:
            sql_paths.append(p)
        elif ext in _CSV_EXTS:
            csv_paths.append(p)
        elif ext == ".json":
            # Heuristic: top-level array of dicts → table. Else treat as text.
            try:
                with p.open("r", encoding="utf-8") as f:
                    head = f.read(8192).lstrip()
                if head.startswith("[") and "{" in head:
                    json_table_paths.append(p)
                else:
                    text_paths.append(p)
            except Exception:
                text_paths.append(p)
        elif ext in _TEXT_EXTS:
            text_paths.append(p)
        else:
            print(f"[db_block] warning: unknown extension '{ext}' for "
                  f"{p.name}; treating as text.")
            text_paths.append(p)

    # Build SQL connection (only if we have any structured source).
    conn: Optional[sqlite3.Connection] = None
    if sql_paths or csv_paths or json_table_paths:
        conn = sqlite3.connect(":memory:")
        # ATTACH .sqlite files
        for p in sql_paths:
            if not p.exists():
                continue
            alias = re.sub(r"[^A-Za-z0-9_]", "_", p.stem) or "db"
            try:
                conn.execute(
                    f'ATTACH DATABASE "{p.as_posix()}" AS "{alias}"'
                )
            except Exception as e:
                print(f"[db_block] warning: attach {p} failed: {e}")
        # CSV → table
        for p in csv_paths:
            try:
                _csv_to_inmem_sqlite(conn, p)
            except Exception as e:
                print(f"[db_block] warning: csv {p} failed: {e}")
        # JSON-array → table
        for p in json_table_paths:
            try:
                with p.open("r", encoding="utf-8") as f:
                    arr = json.load(f)
                if not (isinstance(arr, list) and arr and isinstance(arr[0], dict)):
                    continue
                cols = list(arr[0].keys())
                cols_safe = [re.sub(r"[^A-Za-z0-9_]", "_", c) or f"c{i}"
                             for i, c in enumerate(cols)]
                alias = re.sub(r"[^A-Za-z0-9_]", "_", p.stem) or "tbl"
                cur = conn.cursor()
                cur.execute(
                    f'CREATE TABLE IF NOT EXISTS "{alias}" '
                    "(" + ", ".join(f'"{c}" TEXT' for c in cols_safe) + ")"
                )
                ph = ", ".join("?" for _ in cols_safe)
                for rec in arr[:100000]:
                    vals = [str(rec.get(c, "")) for c in cols]
                    cur.execute(
                        f'INSERT INTO "{alias}" VALUES ({ph})', vals
                    )
                conn.commit()
            except Exception as e:
                print(f"[db_block] warning: json-table {p} failed: {e}")

    # Build text store inspections.
    text_role_hint = text_role_hint or {}
    text_stores: List[TextStoreInspection] = []
    for p in text_paths:
        samples = _sample_text_file(p)
        n_docs: Optional[int] = None
        try:
            with p.open("rb") as f:
                n_docs = sum(1 for _ in f)
        except Exception:
            pass
        name = p.stem
        text_stores.append(TextStoreInspection(
            name=name,
            role=text_role_hint.get(name, f"text passages from '{p.name}'"),
            n_docs=n_docs,
            sample_passages=samples,
        ))

    return make_db_block(
        sql_conn=conn,
        text_stores=text_stores,
        kb_text=kb_text,
        cache_path=cache_path,
        force_rebuild=force_rebuild,
    )


# --------------- convenience: build block for our retail VectorDB ---------------


def build_retail_db_block(
    sql_conn: sqlite3.Connection,
    vector_db,
    sources: Sequence[str],
    kb_text: str = "",
    cache_path: Optional[str | Path] = None,
    force_rebuild: bool = False,
) -> str:
    """Specialised entry-point for our in-domain RetailQA setup. Pulls a
    handful of sample passages from each text source via the existing
    ``VectorDB.search`` interface (using a generic probe query)."""
    text_stores = []
    for src in sources:
        samples: List[str] = []
        try:
            hits = vector_db.search(source=src, query="overview", top_k=3)
            samples = [(h.get("content") or "")[:300] for h in hits]
        except Exception:
            pass
        text_stores.append(TextStoreInspection(
            name=src, role=f"text passages from source '{src}'",
            sample_passages=samples,
        ))
    return _checked_block(
        lambda rebuild: make_db_block(
            sql_conn=sql_conn,
            text_stores=text_stores,
            kb_text=kb_text,
            cache_path=cache_path,
            force_rebuild=rebuild,
        ),
        sql_conn=sql_conn, text_store_names=list(sources),
        cache_path=cache_path, force_rebuild=force_rebuild,
    )


BRIEFING_MAX_ATTEMPTS = 3


def _checked_block(generate, sql_conn, text_store_names, cache_path, force_rebuild):
    """Repair identifiers in the briefing against the real schema, validate it, and
    regenerate (same backbone) when repair cannot fix it. A cached briefing goes
    through the same repair/validation on load. The repaired text replaces the cache;
    the model's raw text is kept next to it (*.raw.txt) with a *.check.json log."""
    from ..config import SQL_SOURCES
    from .briefing_check import read_schema, repair_db_block, validate_db_block

    schema = read_schema(sql_conn, list(SQL_SOURCES.values()))
    cache = Path(cache_path) if cache_path else None
    log = {"attempts": []}
    rebuild = force_rebuild
    block, problems = "", ["not built"]
    for attempt in range(1, BRIEFING_MAX_ATTEMPTS + 1):
        from_cache = bool(cache and cache.exists() and not rebuild)
        raw = generate(rebuild)
        block, changes = repair_db_block(raw, schema, text_store_names)
        problems = validate_db_block(block, schema, text_store_names)
        log["attempts"].append({"attempt": attempt, "from_cache": from_cache,
                                "raw_chars": len(raw), "changes": changes,
                                "problems": problems})
        raw_path = cache.with_suffix(".raw.txt") if cache else None
        if raw_path and (not from_cache or (changes and not raw_path.exists())):
            raw_path.write_text(raw, encoding="utf-8")  # keep the model's own text
        if not problems:
            break
        rebuild = True
    log["valid"] = not problems
    unchanged_cache = log["attempts"][-1]["from_cache"] and not log["attempts"][-1]["changes"]
    if cache and not unchanged_cache:  # skip rewrites when workers just load a clean cache
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(block, encoding="utf-8")
        cache.with_suffix(".check.json").write_text(json.dumps(log, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    if problems:
        raise RuntimeError(f"schema briefing still invalid after {BRIEFING_MAX_ATTEMPTS} "
                           f"attempts: {problems}")
    return block
