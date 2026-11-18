"""OpenRouter client for the RetailRAG agent, the embedder and the LLM judge.

`call_openrouter(..., return_usage=True)` returns token usage as a dict, and the
evaluator reads a per-query budget from the thread-local call/token counters.
All callers share one cached, thread-safe OpenAI client.
"""
import os
import threading
import time
from typing import Any, List, Optional, Sequence

import httpx
from openai import OpenAI


def _resolve_api_key(api_key: str = "") -> str:
    """Return an explicit key, else OPENROUTER_API_KEY from the environment."""
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Missing OpenRouter API key. Set the OPENROUTER_API_KEY environment "
            "variable, or pass api_key= explicitly."
        )
    return key


# ── Thread-local per-query counters (used by evaluation) ────────────────────
# Each eval worker keeps its own counts so ThreadPoolExecutor threads don't
# interfere. Both counters are updated inside call_openrouter on every request.
_call_state = threading.local()


def reset_call_count() -> None:
    _call_state.count = 0


def get_call_count() -> int:
    return getattr(_call_state, "count", 0)


def _increment_call_count() -> None:
    _call_state.count = getattr(_call_state, "count", 0) + 1


def reset_token_counts() -> None:
    _call_state.prompt_tokens = 0
    _call_state.completion_tokens = 0


def get_token_counts() -> dict:
    p = getattr(_call_state, "prompt_tokens", 0)
    c = getattr(_call_state, "completion_tokens", 0)
    return {"prompt": p, "completion": c, "total": p + c}


def _add_token_usage(usage_dict: dict) -> None:
    _call_state.prompt_tokens = getattr(_call_state, "prompt_tokens", 0) + int(
        usage_dict.get("prompt_tokens") or 0)
    _call_state.completion_tokens = getattr(_call_state, "completion_tokens", 0) + int(
        usage_dict.get("completion_tokens") or 0)


# ── Process-wide client cache ───────────────────────────────────────────────
# Each OpenAI() owns an httpx connection pool; recreating one per call leaks
# pools and balloons memory over long eval runs. Key by (api_key, base_url,
# timeout) so different configs still get their own client. The client is
# thread-safe, so sharing across workers is fine.
_CLIENT_LOCK = threading.Lock()
_CLIENT_CACHE: dict = {}


def _get_client(api_key: str, base_url: str, timeout: float):
    key = (api_key, base_url, timeout)
    cli = _CLIENT_CACHE.get(key)
    if cli is not None:
        return cli
    with _CLIENT_LOCK:
        cli = _CLIENT_CACHE.get(key)
        if cli is None:
            cli = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=httpx.Timeout(connect=10.0, read=timeout, write=10.0,
                                      pool=timeout),
            )
            _CLIENT_CACHE[key] = cli
        return cli


def _usage_to_dict(usage) -> dict:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def call_openrouter(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_retries: int = 2,
    timeout: float = 180.0,
    return_usage: bool = False,
    reasoning: Optional[dict] = None,
) -> Any:
    """Chat completion via OpenRouter.

    return_usage=True returns (content, usage_dict) where usage_dict has
    prompt_tokens / completion_tokens / total_tokens. Otherwise returns content
    only. The thread-local call and token counters are updated either way.
    """
    api_key = _resolve_api_key(api_key)
    client = _get_client(api_key, "https://openrouter.ai/api/v1", timeout)

    # Optional reasoning-effort control via env var. Reasoning models on
    # OpenRouter accept this; non-reasoning models ignore it. Keeps per-call
    # latency and cost predictable.
    extra_body = {}
    eff = os.getenv("LLM_REASONING_EFFORT")
    if reasoning is not None:
        # An explicit per-call setting (e.g. the fixed judge) wins over env vars.
        extra_body["reasoning"] = reasoning
    elif eff:
        extra_body["reasoning"] = {"effort": eff}
    elif os.getenv("LLM_REASONING_EXCLUDE"):
        extra_body["reasoning"] = {"exclude": True}

    last_err = ""
    for attempt in range(max_retries + 1):
        _increment_call_count()
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
            usage_dict = _usage_to_dict(getattr(response, "usage", None))
            _add_token_usage(usage_dict)
            if not response.choices:
                last_err = "empty choices (choices=None or [])"
                continue
            content = response.choices[0].message.content
            if content is not None:
                return (content, usage_dict) if return_usage else content
            last_err = f"empty content (finish_reason={response.choices[0].finish_reason})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max_retries:
            time.sleep(2 * (attempt + 1))  # backoff: 2s, 4s
    raise RuntimeError(
        f"{model_id} failed after {max_retries + 1} attempts — last_err: {last_err}"
    )


def embed_openrouter(
    model_id: str,
    texts: Sequence[str],
    api_key: str = "",
) -> List[List[float]]:
    """Create text embeddings via the OpenRouter OpenAI-compatible endpoint.

    Args:
        model_id: e.g., "qwen/qwen3-embedding-8b"
        texts: input strings
        api_key: optional; falls back to OPENROUTER_API_KEY

    Returns:
        Embeddings in the same order as inputs.
    """
    if not texts:
        return []
    api_key = _resolve_api_key(api_key)
    client = _get_client(api_key, "https://openrouter.ai/api/v1", 180.0)
    response = client.embeddings.create(
        model=model_id,
        input=list(texts),
    )
    return [item.embedding for item in response.data]
