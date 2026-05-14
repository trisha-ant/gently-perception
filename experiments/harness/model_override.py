"""Override model/thinking at the call_claude boundary [P2].

The legacy ``perception._base.call_claude`` hardcodes model and thinking.
This shim replaces every bound ``call_claude`` with one that hits the API
using the model/thinking from ``RunConfig`` — so the new runner's
``--model``/``--thinking`` flags actually take effect without editing any
variant code, and Phase A replay-equivalence (which depends on the legacy
call path being unchanged) stays intact.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from typing import Any

import anthropic

from experiments.analysis.record_replay import (
    _extract_content,
    _extract_system,
    _patch_targets,
)

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


@contextmanager
def install_model_override(model: str, thinking: str | None):
    """Replace every ``call_claude`` binding with one that calls the API
    using ``model`` and ``thinking`` instead of the hardcoded defaults.

    ``thinking`` is ``"adaptive"`` or ``None``. We never send
    ``temperature``/``top_p``/``top_k`` (opus-4-7 returns 400 for them) and
    never send ``output_config.effort`` (the legacy ``xhigh`` is 4.7-only;
    omitting it gives the API default ``high`` which works on both 4.6 and
    4.7). Multi-turn (``call_claude_conversation``) is collapsed to a single
    user turn — fine for the 8 leaderboard variants, none of which use it.
    """
    client = _get_client()

    async def _call(system: str, content: list[dict], *, max_tokens: int) -> str:
        kw: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": content}],
        }
        if thinking == "adaptive":
            kw["thinking"] = {"type": "adaptive"}
        resp = await asyncio.to_thread(client.messages.create, **kw)
        for block in resp.content:
            if block.type == "text":
                return block.text
        return ""

    originals: list[tuple[Any, str, Any]] = []
    for mod, name in _patch_targets():
        originals.append((mod, name, getattr(mod, name)))

        async def wrapped(*args, __name=name, **kwargs) -> str:
            system = _extract_system(args, kwargs)
            content = _extract_content(args, kwargs, __name)
            max_tokens = kwargs.get("max_tokens", 4096)
            return await _call(system, content, max_tokens=max_tokens)

        setattr(mod, name, wrapped)

    logger.info(f"model override active: model={model} thinking={thinking}")
    try:
        yield
    finally:
        for mod, name, orig in originals:
            setattr(mod, name, orig)
