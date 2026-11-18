#!/usr/bin/env python3
"""Evaluate RetailRAG on the RetailBridge benchmark.

    python scripts/download_data.py          # once: questions + database from Hugging Face
    python evaluate.py --llm-judge --out results/eval.json

RetailRAG is the DB-adaptive agent from the paper: it plans a query across
sources, executes SQL and dense retrieval, and revises when a step comes back
wrong. Answers are scored with exact-match / substring / F1 / numeric /
date and, with --llm-judge, a binary LLM judge. Results are written
incrementally to --out, so an interrupted run leaves a valid partial file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

from retailrag.pipeline import PlanRevisePipeline, trace_to_dict
from retailrag.evaluation import aggregate, llm_judge, score_item
from retailrag.benchmark_io import load_benchmark

_WRITE_LOCK = threading.Lock()



def _atomic_write_json(out_path: Path, payload: Any) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)


def _build_pipeline(args):
    kb_text = ""
    if args.kb_path and Path(args.kb_path).exists():
        kb_text = Path(args.kb_path).read_text(encoding="utf-8")
    return PlanRevisePipeline(
        max_revisions=args.max_revisions,
        per_step_revisions=args.per_step_revisions,
        enable_plan_retry=bool(getattr(args, "enable_plan_retry", False)),
        retry_trigger_mode=args.retry_trigger_mode,
        kb_text=kb_text,
        cache_path=args.db_block_cache,
        force_rebuild_block=bool(getattr(args, "rebuild_db_block", False)),
        hide_user_log=not args.show_user_log,
    )


def _judge_cfg():
    # The judge stays fixed regardless of the backbone (retailrag.config.judge_config),
    # so cross-model comparisons stay calibrated.
    from retailrag.config import judge_config
    return judge_config()


def _eval_one(rag, item: Dict[str, Any], llm_judge_on: bool) -> Dict[str, Any]:
    from retailrag.openrouter_client import (
        reset_call_count, get_call_count, reset_token_counts, get_token_counts)
    q = item.get("question") or ""
    gold = item.get("answer")
    reset_call_count()
    reset_token_counts()
    trace = rag.answer(q)
    api_calls = get_call_count()
    tokens = get_token_counts()  # captured BEFORE the judge call below
    d = trace_to_dict(trace)
    d["gold"] = gold
    d["meta"] = {k: item.get(k) for k in ("hops", "type", "sources_used")}
    d["api_calls"] = api_calls
    d["tokens"] = tokens
    score = score_item(d.get("answer"), gold)
    if llm_judge_on:
        score.update(llm_judge(q, gold, d.get("answer"), cfg=_judge_cfg()))
    d["score"] = score
    return d


def _run_eval(args, items: List[Dict[str, Any]], out_path: Path) -> Dict[str, Any]:
    if args.limit > 0:
        items = items[: args.limit]
    parallel = max(1, args.parallel or 1)

    if parallel == 1:
        rag = _build_pipeline(args)
        results: List[Dict[str, Any]] = []
        for i, item in enumerate(items):
            q = item.get("question") or ""
            print(f"[{i+1}/{len(items)}] {q[:120]}", flush=True)
            d = _eval_one(rag, item, args.llm_judge)
            print(
                f"    pred={str(d.get('answer'))[:100]!r}\n"
                f"    gold={str(item.get('answer'))[:100]!r}\n"
                f"    score={d['score']}",
                flush=True,
            )
            results.append(d)
            _atomic_write_json(out_path, {"results": results, "summary": aggregate(results)})
        rag.close()
    else:
        # Each worker gets its own pipeline — SQLDB / VectorDB are not safe to
        # share across threads.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        local = threading.local()

        # One schema briefing per run: build and check it once here, then every
        # worker loads the same cached file instead of writing its own.
        if args.rebuild_db_block or not (args.db_block_cache and Path(args.db_block_cache).exists()):
            _build_pipeline(args).close()
            args.rebuild_db_block = False

        def worker(idx, item):
            if not hasattr(local, "rag"):
                local.rag = _build_pipeline(args)
            return idx, _eval_one(local.rag, item, args.llm_judge)

        results: List[Any] = [None] * len(items)
        done = 0
        SAVE_EVERY = 50
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(worker, i, it): i for i, it in enumerate(items)}
            for fut in as_completed(futures):
                idx, d = fut.result()
                results[idx] = d
                done += 1
                q = items[idx].get("question") or ""
                print(f"[{idx+1}/{len(items)}] {q[:90]}\n"
                      f"    pred={str(d.get('answer'))[:80]!r}  score={d['score']}",
                      flush=True)
                if done % SAVE_EVERY == 0 or done == len(items):
                    with _WRITE_LOCK:
                        safe = [r for r in results if r is not None]
                        _atomic_write_json(out_path, {"results": safe, "summary": aggregate(safe)})
        results = [r for r in results if r is not None]

    summary = aggregate(results)
    print(f"\nwrote {out_path}")
    print("--- summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate RetailRAG on RetailBridge.")
    p.add_argument("--eval", default=None,
                   help="question file (default: data/RetailBridge/queries/test.jsonl, "
                        "fetched by scripts/download_data.py)")
    p.add_argument("--out", default="results/eval.json", help="output JSON path")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N items")
    p.add_argument("--parallel", type=int, default=1,
                   help="worker threads (OpenRouter is network-bound)")
    p.add_argument("--llm-judge", action="store_true", help="add binary LLM-judge metric")
    p.add_argument("--model", default=None,
                   help="OpenRouter model id for the agent backbone (judge stays fixed)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="backbone max output tokens incl. reasoning (default: config, 8192)")
    p.add_argument("--show-user-log", action="store_true",
                   help="expose pre-computed KPI tables to the agent (needs a user_log.sqlite "
                        "in the data directory, which the public release does not include; off "
                        "by default so KPIs must be derived from base tables)")

    # agent settings
    p.add_argument("--max-revisions", type=int, default=10)
    p.add_argument("--per-step-revisions", type=int, default=3)
    p.add_argument("--enable-plan-retry", action="store_true", help="best-of-2 plan retry")
    p.add_argument("--retry-trigger-mode", choices=["basic", "expanded"], default="basic")
    p.add_argument("--kb-path", default=None, help="optional domain-knowledge text file")
    p.add_argument("--db-block-cache", default=None,
                   help="schema-briefing cache path (default: data/index/db_block_<model>.txt)")
    p.add_argument("--rebuild-db-block", action="store_true", help="ignore cached DB block")

    args = p.parse_args()

    if args.model:
        from retailrag.config import DEFAULT_CONFIG
        DEFAULT_CONFIG.llm.model_id = args.model
        print(f"[evaluate] backbone model overridden: {args.model}", flush=True)
    if args.max_tokens:
        from retailrag.config import DEFAULT_CONFIG
        DEFAULT_CONFIG.llm.max_tokens = args.max_tokens

    if args.db_block_cache is None:
        # One briefing file per backbone: each model writes and uses its own.
        import re
        from retailrag.config import DEFAULT_CONFIG, INDEX_DIR
        name = re.sub(r"[^A-Za-z0-9._-]", "_", DEFAULT_CONFIG.llm.model_id.split("/")[-1])
        args.db_block_cache = str(INDEX_DIR / f"db_block_{name}.txt")
    if args.eval is None:
        from retailrag.config import BENCHMARK_FILE
        args.eval = str(BENCHMARK_FILE)
    if not Path(args.eval).exists():
        raise SystemExit(f"{args.eval} not found; run `python scripts/download_data.py` first")
    items = load_benchmark(args.eval)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_eval(args, items, out_path)


if __name__ == "__main__":
    main()
