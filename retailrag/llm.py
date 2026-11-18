"""Thin wrapper around the OpenRouter client."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retailrag.openrouter_client import call_openrouter  # type: ignore

from .config import DEFAULT_CONFIG, LLMConfig


def call_llm(
    system_prompt: str,
    user_prompt: str,
    cfg: Optional[LLMConfig] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Call the configured backbone LLM through OpenRouter and return raw text."""
    cfg = cfg or DEFAULT_CONFIG.llm
    temp = cfg.temperature if temperature is None else temperature
    mtok = cfg.max_tokens if max_tokens is None else max_tokens
    return call_openrouter(
        model_id=cfg.model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_key=cfg.api_key,
        temperature=temp,
        max_tokens=mtok,
        reasoning=getattr(cfg, "reasoning", None),
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM output.

    Looks for fenced ```json ... ``` blocks first, then falls back to the
    largest balanced {...} substring. Raises ValueError on failure so the
    caller can decide whether to retry.
    """
    if not text:
        raise ValueError("empty LLM response")

    fenced = _FENCE_RE.findall(text)
    candidates = list(fenced)

    # Fallback: outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"could not parse JSON from LLM output: {last_err}\n---\n{text[:1000]}")


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    cfg: Optional[LLMConfig] = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call the LLM and parse its output as JSON, with light retry."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        raw = call_llm(system_prompt, user_prompt, cfg=cfg, **kwargs)
        try:
            return extract_json(raw)
        except Exception as e:
            last_err = e
            # On retry, nudge the model with a stricter instruction.
            user_prompt = user_prompt + "\n\nIMPORTANT: respond with a single valid JSON object, no prose."
    raise RuntimeError(f"LLM JSON parse failed after {max_retries + 1} attempts: {last_err}")
