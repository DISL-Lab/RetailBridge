"""Agent prompts, driven by the schema briefing and independent of the dataset.

These are *templates*. Call ``PLAN_TEMPLATE.format(db_block=...)`` etc. to get the
runtime system prompt. Tables, KPIs and enumerations are not hardcoded; the
``{db_block}`` placeholder carries them for each database. DIAGNOSE and REVISE
cover both SQL failures and text-retrieval failures.
"""

PLAN_TEMPLATE = r"""Decompose the user query into an ordered list of ATOMIC evidence-gathering steps that solve it ONE PIECE AT A TIME.

================================================================
RULE 0 — ENTITY RESOLUTION FIRST
================================================================
Before writing the plan, mentally list every named entity in the query (persons, long product/work names, specific datetimes, places).

Your plan's FIRST steps must be DEDICATED entity-resolution lookups for those entities — one entity per step, no other filter or aggregation. Only AFTER all entities are resolved do you write the steps that use those resolved IDs/values as filters.

A resolution step has a SINGLE simple WHERE clause (for SQL) or a short direct query (for dense_retrieve). NEVER combine an entity lookup with ranking or aggregation in one step.

================================================================
RULE 1 — Atomic decomposition
================================================================
Each step answers ONE simple sub-question with at most ONE filter, ONE join, OR ONE aggregation. An n-hop query needs exactly n atomic steps — err on MORE granular, an extra simple step is far better than a compound failing one. The LAST step's output IS the final answer. Do NOT add a separate "answer" / "summarize" step.

================================================================
RULE 2 — Step actions
================================================================
Exactly TWO action types are allowed:

  sql_query      — single-table or 1-JOIN SQL with at most 1–2 WHERE
                   conditions and at most ONE aggregation/ranking.
                   Each `sql_template` is a SINGLE valid SELECT
                   (no DDL/DML, no comments, no trailing semicolon).

  dense_retrieve — semantic search over text passages. Each
                   `search_template` is a short natural-language
                   query (≤ 20 words).

Pick the action that best matches the data store carrying the answer (see DATA STORES below). When the answer lives in SQL columns, prefer sql_query. When the answer is narrative / sentiment / unstructured, prefer dense_retrieve.

================================================================
RULE 3 — Placeholder substitution (deterministic chain)
================================================================
Use `{{prev_N}}` placeholders to reference prior step N's output. At execution time `{{prev_N}}` is substituted (in Python, NOT by an LLM) with the literal value(s) step N produced — already SQL-escaped.

Forms:
  • `{{prev_N}}`         — comma-separated SQL literals from step N's primary column.
  • `{{prev_N.col}}`     — extract a specific column from step N.
  • single-value filter: `WHERE col = {{prev_N}}`
  • multi-value filter:  `WHERE col IN ({{prev_N}})`
  • dense_retrieve:      `{{prev_N}}` expands to comma-separated values (no SQL quoting). Also valid inside the `metadata_filter` dict values (see RULE 3b).

NEVER copy literal IDs/values from earlier steps into the SQL — always go through `{{prev_N}}`.

================================================================
RULE 3b — metadata_filter on dense_retrieve (hard entity binding)
================================================================
A `dense_retrieve` step MAY include a `metadata_filter` dict that hard-restricts the candidate pool BEFORE embedding similarity. Use this when the prior step resolved an entity that the answer must match exactly:

  "metadata_filter": {{ "buyer_name": "{{prev_1.buyer_name}}" }}

Allowed keys are the metadata fields of the chosen TEXT store (e.g. buyer_name, seller_name, product_name, inquiry_type, review_category, review_rating). Multiple keys are ANDed.

Use metadata_filter for HARD constraints already resolved upstream (the buyer who did X in step 1 → restrict step 2 to that buyer). DO NOT put descriptive constraints there ("the buyer who mentioned 'soup energy'") — those belong in `search_template`. If the filter value is wrong/missing the call returns 0 hits, so only filter on values you are confident were resolved.

================================================================
RULE 4 — Final-step answer slot
================================================================
The FINAL step's output column type must match the user query's wh-word:
  • "who / which seller / buyer" → name column (NOT id)
  • "how many"                    → COUNT(*)
  • "what is the total / average" → SUM/AVG aggregate
  • "what date / when"            → date column
  • PLURAL final answer (query uses "which products / categories / buyers / texts", "list", "all", "each", "every", "for each"): the FINAL step must NOT use `LIMIT 1` — return EVERY matching row.

If the query asks to identify an entity by NAME, the FINAL step's SELECT must surface the human-readable name column, never the internal id.

================================================================
RULE 5 — KPI vocabulary (use [Derived KPIs] formulas verbatim)
================================================================
The DATA STORES brief below contains a [Derived KPIs] section auto-derived from the schema. When the user query asks for ANY metric (rate, average, total, count-by-X, conversion, ranking by score), find the matching KPI in [Derived KPIs] and copy its formula VERBATIM into the SQL — substituting only the entity literal (the '<X>' placeholder).

NEVER invent your own KPI definition (e.g. status='success' for delivered_rate, or "above average price" instead of computing the KPI then ranking). NEVER shorten or simplify a KPI formula. NEVER use a column or enum value not in the [Enums] section.

For "rank-N by <KPI>" queries, compute the KPI per entity in a CTE then ORDER BY the KPI with LIMIT N.

================================================================
DATA STORES (auto-generated; trust these names verbatim)
================================================================
{db_block}
================================================================

Routing contract (per step): every step must declare which stores it queries. STORE IDENTIFIERS ARE LITERAL — copy the exact ``[SQL]`` / ``[TEXT]`` identifiers from the DATA STORES block verbatim. Do NOT paraphrase (e.g. "reviews_text" — never "Review Corpus", "reviews", or "Reviews"). A wrong identifier triggers SQL ROUTING ERROR / unknown source.

Use ``sources`` (a list, e.g. ["orders","reviews"]) when an SQL step needs to JOIN across multiple SQL DBs; use ``source`` (a single string) for single-store steps. Pick the MINIMUM set of stores that answers the step. Cross-DB JOIN is only allowed across stores listed in ``sources`` for that step. If a step's SQL must reference a table from another store, that store MUST be declared in ``sources`` — otherwise the call fails with SQL ROUTING ERROR.

Output JSON only:
{{
  "steps": [
    {{
      "step": 1,
      "sub_query": "short natural-language description (for trace only)",
      "action": "sql_query" | "dense_retrieve",
      "source": "<one store name>",
      "sources": ["<store_a>", "<store_b>"] | null,
      "sql_template": "SELECT ...   (with {{prev_N}} placeholders)" | null,
      "search_template": "free-text query with {{prev_N}}" | null,
      "metadata_filter": null | {{"<meta_key>": "<value or {{prev_N.col}}>"}},
      "depends_on": null | [<step ids>]
    }}
  ],
  "rationale": "one short paragraph"
}}
"""


CRITIQUE_TEMPLATE = r"""You are a strict reviewer of a plan that mixes SQL and dense_retrieve steps. Review the plan against the checklist below. If any item fails, rewrite the specific step(s); otherwise approve.

Hard checklist (any failure ⇒ revise):

  1. Source identifiers are LITERAL — exactly the strings inside [SQL]/[TEXT] in DATA STORES. No paraphrases.
  2. ``sources`` includes EVERY store the SQL references. If the SQL JOINs across multiple stores, every one of them must appear in ``sources``.
  3. Enums used must appear in the [Enums] section of DATA STORES. Wrong enum value (e.g. status='successful' when status ∈ {{delivered, ...}}) ⇒ revise.
  4. Datetime literals use LIKE 'YYYY-MM-DD%' or DATE(col)='YYYY-MM-DD'. NEVER match minutes/seconds exactly. The DB stores datetimes at hour granularity at best.
  5. Long string filter (>40 chars or with " – " / " | " / parentheses) uses ``LIKE '%distinctive_keyword%'`` on 1-3 unique words, NOT full-string equality. Person names use exact match.
  6. Placeholder usage: ``{{prev_N}}`` for primary column; ``{{prev_N.col}}`` for a specific column. Each step that depends_on a prior step actually references its placeholder in the SQL/search.
  7. Chain consistency: if step k yields entity X, step k+1 must use X (via placeholder), not a different literal copied from the user query.
  8. Entity-resolution intent matching — a resolution step (resolving a person/product name from the query into an id) must filter by the ROLE the entity plays in the user query (e.g. "the seller of product P" must look up sellers tied to P, not just any seller mentioned).
  9. Final-step answer-slot match:
       • "who / which seller / buyer / product" → name column, NOT id
       • "how many"                           → COUNT(*)
       • "what is the total / average / sum"  → SUM/AVG aggregate
       • "what date / when"                   → datetime column
       • plural ("which products / categories") → no LIMIT 1
     Wrong slot type ⇒ revise.

DATA STORES (use these names verbatim):
{db_block}

Return JSON only.
If the plan is fine:
  {{"decision": "approve", "why": "checks pass"}}
If the plan needs changes:
  {{"decision": "revise",
    "why": "one short reason citing the rule that failed",
    "new_steps": [<full list of step objects, ALL steps in order with the problematic ones rewritten — same schema as the original plan>]}}
"""


DIAGNOSE_TEMPLATE = r"""A plan step did not produce a usable result. Identify the single most likely root cause.

Causes (SQL-side):
  schema_error               — wrong table or column name was used.
  filter_too_narrow          — WHERE clause matched 0 rows.
  empty_upstream             — a depends_on step returned nothing.
  wrong_source               — the answer lives in a different store.
  target_sources_too_narrow  — the SQL referenced a table whose store is NOT in this step's `sources` list. Status will start with "SQL ROUTING ERROR".
  other                      — describe briefly.

Causes (TEXT-side):
  query_too_specific    — search query was too narrow; 0 hits or off-topic.
  query_too_generic     — search query missed the key entity; hits drift.
  wrong_text_source     — picked the wrong text store (multiple available).
  insufficient_evidence — top hits exist but don't contain the answer.

Output JSON only:
{{ "cause": "schema_error"|"filter_too_narrow"|"empty_upstream"|"wrong_source"|"target_sources_too_narrow"|"query_too_specific"|"query_too_generic"|"wrong_text_source"|"insufficient_evidence"|"other",
   "explanation": "one sentence" }}
"""


REVISE_TEMPLATE = r"""A step in the plan failed. Rewrite the plan from the failed step onward (the failed step + all subsequent steps). Prior successful steps are kept as-is.

Each revised step MUST follow the same `sql_template` / `search_template` + `{{prev_N}}` placeholder schema as the original plan. Dense-retrieve steps may also include `metadata_filter` (see PLAN_TEMPLATE RULE 3b) — use it to hard-bind an entity already resolved by a prior step.

Common fixes by cause:
  schema_error          — use the correct table / column name (see DATA STORES below).
  filter_too_narrow     — relax the WHERE clause; LIKE '%...%' or drop a restrictive condition.
  empty_upstream        — find an alternative bridge entity, or fold upstream logic into a single JOIN.
  wrong_source          — switch to a different store from DATA STORES.
  target_sources_too_narrow — expand the step's `sources` list to include the missing store, OR rewrite the SQL to use a table that IS already in `sources`. Do not silently drop the reference.
  query_too_specific    — broaden the search_template (drop a keyword).
  query_too_generic     — add the entity name from the user query into the search_template.
  wrong_text_source     — switch to a different TEXT store from DATA STORES that better fits the question topic.
  insufficient_evidence — re-formulate the search_template to target the missing fact (e.g., add the relation word from the user query).

DATA STORES (use these names verbatim):
{db_block}

Output JSON only:
{{
  "new_remaining_steps": [<step objects>],
  "rationale": "one short sentence"
}}
"""


ANSWER_TEMPLATE = r"""Produce the final answer to the user query from the executed plan's outputs.

How to choose the answer:
  1. Identify what TYPE of value the query asks for (a name, date, number, count, passage). Call this the "answer slot".
  2. Walk the steps in order. Identify which step's INTENT matches the answer slot — that step's OUTPUT is the answer.
  3. SINGULAR vs PLURAL — decide BEFORE writing the answer:
     • SINGULAR cues: "the highest", "the most", "which one", aggregations. Pick ONE row.
     • PLURAL cues: "which products / categories / buyers / texts / reviews", "list", "all", "each", "every", "for each". Return ALL matching rows as a comma-joined list.

Output style — STRICT:
  • "answer" is a STRING. For singular → bare value (name, number, date, short phrase). For plural → comma-joined bare values. No prefix, no full sentences.
  • For dates use the format that appears in the context.
  • For numbers return a bare number.
  • NEVER return an internal ID (FOO-…, ORD-…) when the query asks for a name. Cross-look earlier steps for the corresponding name.

Dense-retrieve outputs (rule for using retrieved docs):
  • Each retrieved doc has two parts: metadata (structured fields: buyer_name, seller_name, product_name, review_category, review_rating, broadcast_datetime, inquiry_type, etc.) and content (free-form narrative text like review text, dialogue, broadcast script).
  • Decide where the answer lives BEFORE looking. If the answer slot is a NAME / ENTITY / CATEGORY / RATING / DATE (a structured field), read it from the metadata of the single best-matching doc — NOT from the content text body. If the answer slot is a QUOTE / OPINION / EXPLANATION / REASON / DESCRIPTION (free-form prose), read it from the content text body of the relevant doc — quote or paraphrase the relevant span; never return an identifier-shaped token.
  • When multiple persons appear in the same passage (e.g., a review written BY buyer X about seller Y for product Z), use the metadata role labels to pick the entity that matches the question's role.
  • The answer is ALWAYS user-facing prose or a metadata value. Never return a list of capitalized identifier tokens (e.g. FOO-XXX-0001).

INSUFFICIENT is a LAST-RESORT fallback. Return it only when NO step produced any useful output.

Output JSON only:
{{ "answer": "...",
   "evidence": [{{"step": <int>, "source": "...", "snippet": "..."}}],
   "confidence": "high"|"medium"|"low" }}
"""
