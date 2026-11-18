"""RetailRAG pipeline: query -> plan -> execute -> revise -> answer.

The DB-adaptive pipeline used in the paper. The DB block is
composed once per (sql_db x vector_db x kb_text) combo and cached on disk; the
agent's PLAN/REVISE prompts are built from that block at construction time, with
no retail-specific rules baked in.
"""
from __future__ import annotations

from typing import Optional

from .config import DEFAULT_CONFIG, Config
from .databases import SQLDB, VectorDB
from .tools import RetrieveTool, SQLTool
from .agents.plan_revise_agent import trace_to_dict  # noqa: F401  (re-exported)


class PlanRevisePipeline:
    """RetailRAG's agent, wrapping the shared SQL/text backends.

    ``answer(query)`` returns a PlanReviseTrace; ``trace_to_dict`` (imported
    above) serializes it.
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        max_revisions: int = 10,
        per_step_revisions: int = 3,
        enable_plan_retry: bool = True,
        retry_trigger_mode: str = "basic",
        hide_user_log: bool = True,
        kb_text: str = "",
        cache_path: Optional[str] = None,
        force_rebuild_block: bool = False,
    ) -> None:
        from .config import VECTOR_SOURCES
        from .agents.db_block import build_retail_db_block
        from .agents.plan_revise_agent import PlanReviseAgent
        from .tools.sql_tool import make_adaptive_sql_prompt

        self.cfg = cfg or DEFAULT_CONFIG
        self.sql_db = SQLDB(hide_user_log=hide_user_log)
        self.vector_db = VectorDB(self.cfg.embedding)
        self.retrieve_tool = RetrieveTool(self.vector_db)

        if cache_path is None:
            cache_path = "data/index/db_block_retail.txt"
        self.db_block = build_retail_db_block(
            sql_conn=self.sql_db.conn,
            vector_db=self.vector_db,
            sources=list(VECTOR_SOURCES.keys()),
            kb_text=kb_text,
            cache_path=cache_path,
            force_rebuild=force_rebuild_block,
        )
        # Ablation (RETAILRAG_NO_BRIEFING=1): replace the KPI-enriched schema
        # briefing with a bare table/column listing, to isolate how much of
        # RetailRAG's performance comes from the briefing vs. routing/revision.
        import os as _os
        if _os.environ.get("RETAILRAG_NO_BRIEFING"):
            cur = self.sql_db.conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%' ORDER BY name")
            _tabs = [r[0] for r in cur.fetchall()]
            _lines = []
            for _t in _tabs:
                cur.execute(f'PRAGMA table_info("{_t}")')
                _cols = [r[1] for r in cur.fetchall()]
                _lines.append(f'  {_t}({"|".join(_cols)})')
            self.db_block = "\n".join(_lines)
            print("[ablation] RETAILRAG_NO_BRIEFING -> using bare schema instead of db_block", flush=True)
        self.sql_tool = SQLTool(
            self.sql_db,
            system_prompt=make_adaptive_sql_prompt(self.db_block),
        )
        self.agent = PlanReviseAgent(
            db_block=self.db_block,
            sql_tool=self.sql_tool,
            retrieve_tool=self.retrieve_tool,
            cfg=self.cfg,
            max_revisions=max_revisions,
            per_step_revisions=per_step_revisions,
            enable_plan_retry=enable_plan_retry,
            retry_trigger_mode=retry_trigger_mode,
        )

    def answer(self, query: str):
        return self.agent.answer(query)

    def close(self) -> None:
        try:
            self.sql_db.close()
        except Exception:
            pass
