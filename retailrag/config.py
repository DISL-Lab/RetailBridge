"""Centralized configuration for RetailRAG."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The evaluation database from the Hugging Face release (DISLab/RetailBridge,
# fetched by scripts/download_data.py). Free-text columns are held out of the SQL
# tables and the pre-computed KPI tables are not included, so the agent must reach
# narrative evidence through retrieval and derive KPIs from base tables.
# Override with RETAILRAG_DATA_DIR.
DATABASE_DIR = Path(
    os.environ.get("RETAILRAG_DATA_DIR", PROJECT_ROOT / "data" / "RetailBridge" / "database" / "raw")
)
BENCHMARK_FILE = DATABASE_DIR.parent.parent / "queries" / "test.jsonl"
RESULT_DIR = Path(os.environ.get("RETAILRAG_RESULT_DIR", PROJECT_ROOT / "results"))
INDEX_DIR = Path(os.environ.get("RETAILRAG_INDEX_DIR", PROJECT_ROOT / "data" / "index"))


# SQL: each sqlite file holds one or more tables. We attach all of them under
# a single connection so that cross-DB JOINs are possible.
#
# IMPORTANT — user_log is HIDDEN from the agent. Its tables (Buyer_Metric,
# Seller_Metric, Seller) are pre-computed KPIs the agent must instead derive
# from base tables (Order, Review, Script, ...) at plan time.
# ``HIDE_USER_LOG=True`` keeps the bare-name VIEWs out of SQLDB so any SQL the
# agent emits referring to those tables fails — pushing it to compute the
# metric from base data via tool composition.
SQL_FILES: Dict[str, Path] = {
    "product":  DATABASE_DIR / "product.sqlite",
    "script":   DATABASE_DIR / "script.sqlite",
    "order":    DATABASE_DIR / "order.sqlite",
    "inquiry":  DATABASE_DIR / "inquiry.sqlite",
    "review":   DATABASE_DIR / "review.sqlite",
    "user_log": DATABASE_DIR / "user_log.sqlite",
}

HIDE_USER_LOG: bool = True  # paper's task-difficulty knob

# Logical SQL "sources" surfaced to the planner. user_log-backed tables
# (buyer_metrics / seller_metrics / sellers) are NOT listed when
# HIDE_USER_LOG is set — the agent has to compute their KPIs from base.
SQL_SOURCES: Dict[str, str] = {
    "products":        "Product",
    "scripts":         "Script",
    "orders":          "Order",          # quoted in SQL because it's reserved
    "inquiries":       "Inquiry",
    "reviews":         "Review",
}
# Tables that exist in the SQLite files but should be invisible to the agent.
HIDDEN_TABLES = {"Buyer_Metric", "Seller_Metric", "Seller"}

# Vector DB sources (jsonl) used for dense retrieval.
VECTOR_SOURCES: Dict[str, Path] = {
    "sales_scripts":  DATABASE_DIR / "sales_scripts.jsonl",
    "inquiries_text": DATABASE_DIR / "inquiries.jsonl",
    "reviews_text":   DATABASE_DIR / "reviews.jsonl",
}


@dataclass
class LLMConfig:
    model_id: str = "openai/gpt-oss-120b"
    temperature: float = 0.3
    max_tokens: int = 8192
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    reasoning: Optional[Dict[str, Any]] = None  # OpenRouter `reasoning` field, e.g. {"effort": "low"}


# Fixed LLM judge for every RetailBridge evaluation, independent of the backbone.
JUDGE_MODEL_ID: str = "meta/muse-ember"
JUDGE_REASONING: Optional[Dict[str, Any]] = {"effort": "low"}
JUDGE_MAX_TOKENS: int = 8192


def judge_config() -> "LLMConfig":
    """LLMConfig of the fixed judge."""
    return LLMConfig(model_id=JUDGE_MODEL_ID, temperature=0.0,
                     max_tokens=JUDGE_MAX_TOKENS, reasoning=JUDGE_REASONING)


@dataclass
class EmbeddingConfig:
    # Backend: "openrouter" (no local GPU) or "sentence_transformers" (local).
    backend: str = "openrouter"

    # OpenRouter remote embedding (default).
    model_name: str = "qwen/qwen3-embedding-8b"
    # Per-batch upper bound. OpenRouter accepts lists; chunk to be safe under
    # request size / rate limits.
    batch_size: int = 32

    # The following only apply when backend == "sentence_transformers".
    st_model_name: str = "intfloat/multilingual-e5-base"
    st_max_seq_length: int = 384
    # Whether the local model expects the e5-style "query: " / "passage: " prefix.
    use_e5_prefix: bool = True
    device: str = "cuda"  # local mode only — falls back to "cpu" automatically


@dataclass
class PipelineConfig:
    # Top-k documents returned by dense retrieval.
    retrieve_top_k: int = 10
    # Maximum execution steps to guard against runaway plans.
    max_plan_steps: int = 8


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


DEFAULT_CONFIG = Config()


# Schema documentation injected into agent prompts so they know what columns
# exist on each logical source. Kept minimal so it fits inside system prompts.
SCHEMA_HINTS: Dict[str, List[str]] = {
    "products":       ["product_id", "product_name", "product_category", "original_price", "store", "features"],
    "scripts":        ["script_id", "seller_id", "seller_name", "product_id", "product_name", "broadcast_datetime", "discount_rate", "original_price", "discount_price", "discount_end_date", "shipping_fee"],
    "orders":         ["order_id", "script_id", "buyer_id", "buyer_name", "product_id", "product_name", "quantity", "total_price", "status", "order_date", "delivery_date", "post_inquiry_type", "seller_id", "seller_name"],
    "inquiries":      ["inquiry_id", "inquiry_type", "buyer_id", "buyer_name", "seller_id", "seller_name", "product_id", "product_name", "script_id"],
    "reviews":        ["review_id", "review_category", "review_rating", "buyer_id", "buyer_name", "seller_id", "seller_name", "product_id", "product_name", "script_id", "inquiry_id"],
    # JSONL text sources (vector indexed, not SQL-queryable). Each record has a
    # `content` field plus metadata (entity ids/names) that mirror the SQL keys.
    "sales_scripts":  ["id (script_id)", "content", "metadata"],
    "inquiries_text": ["inquiry_id", "content", "metadata"],
    "reviews_text":   ["review_id", "content", "metadata"],
    # NOTE: user_log-backed tables (Buyer_Metric / Seller_Metric / Seller)
    # are intentionally NOT listed here. The agent must derive KPIs from
    # base tables via tool composition (calc_refund_rate, calc_conversion_rate,
    # aggregate, plus SQL SUM/AVG/COUNT on Order/Review/Script).
}
