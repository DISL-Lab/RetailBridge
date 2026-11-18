"""Deterministic repair and validation of the LLM-written schema briefing (db_block).

The briefing is written once per backbone by that backbone. Its prose (value
distributions, join hints, KPI recipes) is kept as written; only identifiers are
normalised against the real schema, so that a naming slip in the setup step cannot
break every later question.

repair_db_block(block, schema, text_stores) -> (block, changes)
  * Store headers. "[SQL] sql" / "sqlite" / "main" / "backend" umbrella headers are
    split into one "[SQL] <Table>" header per table block found beneath them;
    prefixes ("sqlite.Review", "product.Product") and quotes are removed; case and
    plural variants become the table name; "[TEXT] reviews" -> "[TEXT] reviews_text".
  * Columns. "Table.column" references and column lines inside a table block are
    checked against the schema; a unique close match is corrected, a line naming
    a column that does not exist is dropped.
  * Lines that mention the hidden KPI tables are dropped.
validate_db_block(block, schema, text_stores) -> list of problems ([] = valid)
"""
from __future__ import annotations

import difflib
import re
import sqlite3
from typing import Dict, List, Sequence, Tuple

GENERIC_SQL_NAMES = {"sql", "sqlite", "sqlite3", "main", "backend", "database", "db",
                     "sql_db", "main_db", "relational", "rdbms", "sql store", "sqlite db"}
HIDDEN = ("Seller_Metric", "Buyer_Metric")
MIN_CHARS = 2000


def read_schema(conn: sqlite3.Connection, tables: Sequence[str]) -> Dict[str, List[str]]:
    out = {}
    for t in tables:
        cur = conn.execute(f'PRAGMA table_info("{t}")')
        out[t] = [r[1] for r in cur.fetchall()]
    return out


def _match(name: str, options: Sequence[str]) -> str | None:
    """Exact, case/plural-insensitive, then unique close match."""
    if name in options:
        return name
    low = {o.lower(): o for o in options}
    n = name.lower()
    for cand in (n, n.rstrip("s"), n + "s"):
        if cand in low:
            return low[cand]
    close = difflib.get_close_matches(n, list(low), n=2, cutoff=0.85)
    if len(close) == 1:
        return low[close[0]]
    return None


def _table_of(name: str, schema: Dict[str, List[str]]) -> str | None:
    name = name.strip().strip("\"'`[]")
    if re.search(r"[.:]", name):
        name = re.split(r"[.:]", name)[-1]
    return _match(name, list(schema))


def _text_store_of(name: str, stores: Sequence[str]) -> str | None:
    n = name.strip().strip("\"'`[]").lower()
    if n in stores:
        return n
    aliases = {s: {s, s.replace("_text", ""), s.rstrip("s"), s.replace("_", " ")} for s in stores}
    aliases.get("sales_scripts", set()).update({"scripts", "sales_script", "script_text", "scripts_text"})
    for s, al in aliases.items():
        if n in al:
            return s
    close = difflib.get_close_matches(n, list(stores), n=2, cutoff=0.8)
    return close[0] if len(close) == 1 else None


HEADER = re.compile(r"^(\s*)\[(SQL|TEXT)\]\s+(.+?)\s*(?:(—|-{1,2}|:)\s*(.*))?$")
# A line that is only a table name, optionally followed by ":" / "—" / "(...)" text
SUBHEAD_LINE = re.compile(r"^\s*\"?([A-Za-z_][\w.:]*?)\"?\s*((?::|—|-{1,2}\s|\().*)?$")
COLUMN = re.compile(r"^(\s+)([A-Za-z_]\w*)(\s*(?:\(|∈|:|—|-\s|$).*)$")
QUALIFIED = re.compile(r"\b\"?([A-Z][A-Za-z_]*)\"?\.([a-z_][a-z0-9_]*)\b")


def repair_db_block(block: str, schema: Dict[str, List[str]],
                    text_stores: Sequence[str]) -> Tuple[str, List[str]]:
    changes: List[str] = []
    out: List[str] = []
    umbrella_indent = None      # indent of an "[SQL] sql" header being split
    current_table = None        # table whose column block we are in
    section = None              # "SQL" / "TEXT" / other bracketed section
    for line in block.split("\n"):
        stripped = line.strip()
        if any(h in line for h in HIDDEN):
            changes.append(f"drop (hidden table): {stripped[:80]}")
            continue
        m = HEADER.match(line)
        if m:
            indent, kind, name, sep, desc = m.groups()
            name_clean = name.strip().strip("\"'`")
            section = kind
            current_table = None
            umbrella_indent = None
            if kind == "SQL":
                if name_clean.lower() in GENERIC_SQL_NAMES:
                    # "[SQL] backend — Product store: ..." names its table in the description
                    lead = re.match(r"\s*\"?([A-Za-z_]\w*)\"?\b(.*)", desc or "")
                    tbl = {t.lower(): t for t in schema}.get(lead.group(1).lower()) if lead else None
                    if tbl:
                        current_table = tbl
                        changes.append(f"[SQL] {name_clean} — {tbl} … -> [SQL] {tbl}")
                        rest = lead.group(2).strip().lstrip(":—-").strip()
                        out.append(f"{indent}[SQL] {tbl}" + (f" — {rest}" if rest else ""))
                        continue
                    umbrella_indent = len(indent)
                    changes.append(f"split umbrella header '[SQL] {name_clean}' into per-table headers")
                    continue
                tbl = _table_of(name_clean, schema)
                if tbl:
                    current_table = tbl
                    if tbl != name_clean:
                        changes.append(f"[SQL] {name_clean} -> [SQL] {tbl}")
                    out.append(f"{indent}[SQL] {tbl}" + (f" — {desc}" if desc else ""))
                    continue
            else:
                st = _text_store_of(name_clean, text_stores)
                if st:
                    if st != name_clean:
                        changes.append(f"[TEXT] {name_clean} -> [TEXT] {st}")
                    out.append(f"{indent}[TEXT] {st}" + (f" — {desc}" if desc else ""))
                    continue
            out.append(line)
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section, current_table, umbrella_indent = None, None, None
            out.append(line)
            continue
        # Table sub-block under an umbrella header ("  Product" / "  Product:")
        if umbrella_indent is not None:
            # flat "Product.product_id (...)" lines: open a header when the table changes
            qm = re.match(r"^\s*\"?([A-Z][A-Za-z_]*)\"?\.[a-z_]", line)
            qt = {t.lower(): t for t in schema}.get(qm.group(1).lower()) if qm else None
            if qt:
                if qt != current_table:
                    current_table = qt
                    out.append(f"[SQL] {qt}")
                out.append(line)
                continue
            sm = SUBHEAD_LINE.match(re.sub(r"^(\s*)(?:Table|TABLE|table)\s+", r"\1", line))
            word = sm.group(1).split(".")[-1].split(":")[-1] if sm else ""
            tbl = {t.lower(): t for t in schema}.get(word.lower())
            if tbl:
                current_table = tbl
                rest = (sm.group(2) or "").strip().lstrip(":—-").strip()
                out.append(f"[SQL] {tbl}" + (f" — {rest}" if rest else ""))
                continue
        # Column lines inside a table block
        if current_table and section == "SQL":
            cm = COLUMN.match(line)
            if cm and len(cm.group(1)) >= 2:
                col = cm.group(2)
                cols = schema[current_table]
                if col not in cols and col.lower() not in {c.lower() for c in cols}:
                    fix = _match(col, cols)
                    if fix:
                        changes.append(f"{current_table}.{col} -> {fix}")
                        line = f"{cm.group(1)}{fix}{cm.group(3)}"
                    elif re.fullmatch(r"[a-z_][a-z0-9_]*", col):
                        changes.append(f"drop (no column {current_table}.{col}): {stripped[:80]}")
                        continue
        # Qualified Table.column references anywhere
        def _q(mq):
            tbl = _table_of(mq.group(1), schema)
            if not tbl:
                return mq.group(0)
            col = mq.group(2)
            if col in schema[tbl]:
                return f"{tbl}.{col}" if tbl != mq.group(1) else mq.group(0)
            fix = _match(col, schema[tbl])
            if fix:
                changes.append(f"{tbl}.{col} -> {tbl}.{fix}")
                return f"{tbl}.{fix}"
            return mq.group(0)
        line = QUALIFIED.sub(_q, line)
        out.append(line)
    return "\n".join(out), changes


def validate_db_block(block: str, schema: Dict[str, List[str]],
                      text_stores: Sequence[str]) -> List[str]:
    problems = []
    sql_names = [re.sub(r"\s.*", "", n).strip("\"'") for n in re.findall(r"^\s*\[SQL\]\s+(\S+)", block, re.M)]
    txt_names = [n.strip("\"'") for n in re.findall(r"^\s*\[TEXT\]\s+(\S+)", block, re.M)]
    if set(sql_names) != set(schema):
        problems.append(f"[SQL] stores {sorted(set(sql_names))} != {sorted(schema)}")
    if set(txt_names) != set(text_stores):
        problems.append(f"[TEXT] stores {sorted(set(txt_names))} != {sorted(text_stores)}")
    if len(block) < MIN_CHARS:
        problems.append(f"too short ({len(block)} chars), likely truncated")
    for h in HIDDEN:
        if h in block:
            problems.append(f"mentions hidden table {h}")
    for t, c in QUALIFIED.findall(block):
        if t in schema and c not in schema[t]:
            problems.append(f"unknown column {t}.{c}")
    return problems
