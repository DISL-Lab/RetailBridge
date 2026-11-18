"""Dense retrieval over the three jsonl text sources.

Documents are stored on disk as jsonl with at least an ``id``-like key,
``content``, and a ``metadata`` block carrying the entity IDs (script_id,
seller_id, buyer_id, product_id, ...) that we can filter on at query time.

Two embedding backends are supported, switchable via ``EmbeddingConfig``:
  * ``openrouter`` — calls qwen3-embedding-8b through the OpenRouter API.
    Default. No local GPU required.
  * ``sentence_transformers`` — local model, useful for ablations / offline
    runs where the API is unavailable.

Vectors are normalized at write-time so search reduces to a single matmul.
"""
from __future__ import annotations

import json
import math
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import DEFAULT_CONFIG, INDEX_DIR, VECTOR_SOURCES, EmbeddingConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


ID_KEYS = ("inquiry_id", "review_id", "id")


# Metadata fields that get prepended to ``content`` at embed-time so the
# vector encodes entity / category / rating context, not just the narrative
# body. The fix targets L3 text-only failures where embedding similarity
# matched surface entities (product, seller) but missed the chain-resolved
# entity (e.g. the specific buyer described by the prior hop). With entity
# names baked into the embedded text, retrieval becomes entity-aware.
#
# We deliberately INCLUDE only human-readable, low-cardinality fields:
#   • *_name        — pin buyers/sellers/products by name
#   • *_type        — pre/post inquiry distinction
#   • *_category    — broadcast/pre-cs/post-cs/product review distinction
#   • *_rating      — numeric rating exposed as a literal token
#   • *_datetime    — only when the source explicitly uses date filtering
# and EXCLUDE *_id columns (high cardinality, not in user queries).
_EMBED_META_KEYS = (
    "inquiry_type", "review_category", "review_rating",
    "buyer_name", "seller_name", "product_name", "broadcast_datetime",
)


def _build_embed_text(content: str, metadata: Dict[str, Any]) -> str:
    """Return content with a metadata header prepended (for embed-time only)."""
    if not metadata:
        return content
    pairs = []
    for k in _EMBED_META_KEYS:
        v = metadata.get(k)
        if v in (None, ""):
            continue
        pairs.append(f"{k}: {v}")
    if not pairs:
        return content
    header = " | ".join(pairs)
    return f"[{header}]\n{content}"


@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    domain: str


def _load_jsonl(path: Path) -> List[Document]:
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = next((obj[k] for k in ID_KEYS if k in obj), None)
            if doc_id is None:
                continue
            docs.append(
                Document(
                    doc_id=str(doc_id),
                    content=obj.get("content", ""),
                    metadata=obj.get("metadata", {}) or {},
                    domain=obj.get("domain", ""),
                )
            )
    return docs


# ---------------------------------------------------------------- backends


class _BaseEmbedder:
    def encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        raise NotImplementedError

    @property
    def tag(self) -> str:
        """Stable identifier baked into index filenames."""
        raise NotImplementedError


class _STEmbedder(_BaseEmbedder):
    """Local sentence-transformers embedder (optional, no API calls)."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for backend=sentence_transformers. "
                "Install with: pip install sentence-transformers"
            ) from e

        import torch

        device = self.cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        model = SentenceTransformer(self.cfg.st_model_name, device=device)
        model.max_seq_length = self.cfg.st_max_seq_length
        self._model = model
        return model

    def encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        model = self._load()
        if self.cfg.use_e5_prefix:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        vecs = model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    @property
    def tag(self) -> str:
        return "st__" + re.sub(r"[^A-Za-z0-9]+", "_", self.cfg.st_model_name)


def _normalize_arr(embeddings: List[List[float]]) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


class _OpenRouterEmbedder(_BaseEmbedder):
    """Remote embedding via OpenRouter (default).

    Splits inputs into chunks of ``cfg.batch_size`` and retries with
    exponential backoff on transient errors.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        from retailrag.openrouter_client import embed_openrouter  # type: ignore

        self._embed = embed_openrouter

    # Per-request char budget. qwen3-embedding-8b accepts batched text but
    # inquiries dialogues run ~6K chars (=~1.5K tokens) on average, so a
    # naive 32-batch can exceed the request token limit and retry forever.
    # Cap by char sum AND a hard per-call doc count.
    _CHAR_BUDGET = 12000
    _MAX_DOCS_PER_CALL = 16

    def encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        # qwen3-embedding-8b takes raw text; no prefix convention needed.
        # If a single doc itself blows the budget, we send it alone but
        # truncated to a safer length so the call still succeeds.
        bs = max(1, self.cfg.batch_size)

        # Greedy char-aware packing.
        batches: List[List[str]] = []
        cur: List[str] = []
        cur_chars = 0
        for t in texts:
            t_safe = t if len(t) <= self._CHAR_BUDGET else t[: self._CHAR_BUDGET]
            if cur and (cur_chars + len(t_safe) > self._CHAR_BUDGET
                        or len(cur) >= min(bs, self._MAX_DOCS_PER_CALL)):
                batches.append(cur)
                cur, cur_chars = [], 0
            cur.append(t_safe)
            cur_chars += len(t_safe)
        if cur:
            batches.append(cur)

        # Concurrent batch dispatch — OpenRouter tolerates many parallel
        # embed calls; sequential was the bottleneck (~30-60 min). With
        # 100-way parallelism this drops to ~1-3 min for 43K docs.
        if is_query or len(batches) <= 1:
            # Single batch (e.g., a query embed) — no concurrency benefit.
            return _normalize_arr([e for b in batches for e in self._call_with_retry(b)])

        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_workers = min(100, len(batches))
        results: List[Optional[List[List[float]]]] = [None] * len(batches)

        def _worker(idx: int, batch: List[str]):
            return idx, self._call_with_retry(batch)

        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_worker, i, b) for i, b in enumerate(batches)]
            for fut in as_completed(futs):
                idx, embs = fut.result()
                results[idx] = embs
                done += 1
                if done % 100 == 0 or done == len(batches):
                    print(f"    {done}/{len(batches)} batches done "
                          f"({sum(len(r) for r in results if r)} docs)",
                          flush=True)

        out: List[List[float]] = []
        for embs in results:
            if embs is None:
                raise RuntimeError("a batch failed; partial encoding")
            out.extend(embs)
        return _normalize_arr(out)

    def _call_with_retry(self, batch: List[str], max_retries: int = 3) -> List[List[float]]:
        last_err: Optional[Exception] = None
        cur = batch
        for attempt in range(max_retries + 1):
            try:
                return self._embed(model_id=self.cfg.model_name, texts=cur)
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    break
                # On any failure, halve the batch and retry. If we're already
                # at 1 doc, shrink the doc itself.
                if len(cur) > 1:
                    cur = cur[: max(1, len(cur) // 2)]
                else:
                    cur = [cur[0][: max(500, len(cur[0]) // 2)]]
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"OpenRouter embedding failed after {max_retries + 1} attempts (batch={len(batch)}): {last_err}"
        )

    @property
    def tag(self) -> str:
        return "or__" + re.sub(r"[^A-Za-z0-9]+", "_", self.cfg.model_name)


def _make_embedder(cfg: EmbeddingConfig) -> _BaseEmbedder:
    if cfg.backend == "openrouter":
        return _OpenRouterEmbedder(cfg)
    if cfg.backend == "sentence_transformers":
        return _STEmbedder(cfg)
    raise ValueError(f"unknown embedding backend: {cfg.backend}")


# ---------------------------------------------------------------- VectorDB


class VectorDB:
    """Per-source dense retriever backed by a numpy matrix on disk.

    Indexes (numpy matrices + Document lists) are cached at the CLASS level
    so multiple VectorDB instances inside one process share the same memory.
    A per-source numpy index is ~700 MB (43k docs × 4096-dim float32); without
    this cache, fan-out workers each hold their own copy → tens of GB. The
    cache is keyed by (backend_tag, source) so different embedding backends
    do not collide.

    All cache reads/writes are guarded by a process-level lock; numpy
    matmul + mask reads on the cached arrays are inherently thread-safe.
    """

    # (backend_tag, source) → {"vecs": np.ndarray, "docs": list[Document]}
    import threading as _threading
    _SHARED_SOURCES: Dict[tuple, Dict[str, Any]] = {}
    _CACHE_LOCK = _threading.Lock()
    # backend_tag → embedder instance (HTTP client reuse)
    _SHARED_EMBEDDERS: Dict[str, Any] = {}

    def __init__(self, embedding_cfg: Optional[EmbeddingConfig] = None) -> None:
        self.embedding_cfg = embedding_cfg or DEFAULT_CONFIG.embedding
        # Reuse a single embedder per (process, backend) so HTTP connection
        # pools and any in-memory weights are not duplicated per worker.
        with VectorDB._CACHE_LOCK:
            tag_probe = _make_embedder(self.embedding_cfg)
            tag = tag_probe.tag
            if tag in VectorDB._SHARED_EMBEDDERS:
                self._embedder = VectorDB._SHARED_EMBEDDERS[tag]
            else:
                self._embedder = tag_probe
                VectorDB._SHARED_EMBEDDERS[tag] = tag_probe
        # Per-instance view; the actual storage is the class-level cache.
        self._sources: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ paths
    def _index_paths(self, source: str) -> Dict[str, Path]:
        # Single index per source. Different backends are kept apart by
        # pointing INDEX_DIR somewhere else (e.g. data/index_st/) when
        # running ablations — keeps file names simple and avoids
        # path-suffix manipulation surprises with multi-dot stems.
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "vecs": INDEX_DIR / f"{source}.npy",
            "docs": INDEX_DIR / f"{source}.pkl",
        }

    # ------------------------------------------------------------------ build
    def build_index(self, source: str, force: bool = False, verbose: bool = False) -> int:
        if source not in VECTOR_SOURCES:
            raise KeyError(f"unknown vector source: {source}")
        paths = self._index_paths(source)
        if not force and paths["vecs"].exists() and paths["docs"].exists():
            return -1

        docs = _load_jsonl(VECTOR_SOURCES[source])
        if not docs:
            raise RuntimeError(f"no documents loaded from {VECTOR_SOURCES[source]}")
        # Embed `[metadata header]\ncontent` so entity / category / rating
        # information is part of the vector. The serialized Document keeps
        # `content` unchanged (the LLM still sees the raw narrative at
        # answer-synthesis time); only the encoded text differs.
        texts = [_build_embed_text(d.content, d.metadata) for d in docs]

        if verbose:
            print(f"  encoding {len(texts)} docs via {self._embedder.tag} ...", flush=True)
        vecs = self._embedder.encode(texts, is_query=False)

        np.save(paths["vecs"], vecs)
        with open(paths["docs"], "wb") as fh:
            pickle.dump(docs, fh)
        return len(docs)

    # ------------------------------------------------------------------ load
    def _load_source(self, source: str) -> Dict[str, Any]:
        # Process-shared cache, keyed by (backend, source). All instances
        # in the same process get the same numpy array reference.
        key = (self._embedder.tag, source)
        cache = VectorDB._SHARED_SOURCES
        cached = cache.get(key)
        if cached is not None:
            return cached
        with VectorDB._CACHE_LOCK:
            cached = cache.get(key)  # double-check after acquiring lock
            if cached is not None:
                return cached
            paths = self._index_paths(source)
            if not (paths["vecs"].exists() and paths["docs"].exists()):
                raise FileNotFoundError(
                    f"index for '{source}' (backend={self._embedder.tag}) "
                    f"not built. Run scripts/build_vector_index.py first."
                )
            vecs = np.load(paths["vecs"])
            with open(paths["docs"], "rb") as fh:
                docs = pickle.load(fh)
            bundle = {"vecs": vecs, "docs": docs}
            cache[key] = bundle
            return bundle

    # ------------------------------------------------------------------ search
    def search(
        self,
        source: str,
        query: str,
        top_k: int = 5,
        filter_ids: Optional[Iterable[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        bundle = self._load_source(source)
        vecs: np.ndarray = bundle["vecs"]
        docs: List[Document] = bundle["docs"]

        mask = _build_mask(docs, filter_ids, filter_metadata)
        if mask is not None and not mask.any():
            return []

        qvec = self._embedder.encode([query], is_query=True)[0]
        scores = vecs @ qvec
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)

        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, top_k)[:top_k]
            order = part[np.argsort(-scores[part])]

        results = []
        for idx in order:
            score = float(scores[idx])
            if not np.isfinite(score):
                continue
            d = docs[idx]
            results.append(
                {
                    "doc_id": d.doc_id,
                    "score": score,
                    "content": d.content,
                    "metadata": d.metadata,
                    "domain": d.domain,
                }
            )
            if len(results) >= top_k:
                break
        return results


_ENTITY_KEYS = ("seller_id", "buyer_id", "product_id", "script_id", "inquiry_id", "review_id")


def _build_mask(
    docs: List[Document],
    filter_ids: Optional[Iterable[str]],
    filter_metadata: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    if not filter_ids and not filter_metadata:
        return None

    ids = set(filter_ids or [])
    md = filter_metadata or {}

    n = len(docs)
    out = np.zeros(n, dtype=bool)
    for i, d in enumerate(docs):
        ok = True
        if ids:
            doc_match = d.doc_id in ids
            meta_match = any(str(d.metadata.get(k)) in ids for k in _ENTITY_KEYS)
            if not (doc_match or meta_match):
                ok = False
        if ok and md:
            for k, v in md.items():
                if str(d.metadata.get(k)) != str(v):
                    ok = False
                    break
        out[i] = ok
    return out
