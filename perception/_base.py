"""
Shared utilities for modular perception functions.

Provides the common output dataclass, API helper, and prompt-building utilities
that all perception function variants use.

This file is fixed infrastructure (like prepare.py in autoresearch).
The agent should not modify it — only the perception function files.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# Default model for single-call perception functions
DEFAULT_MODEL = "claude-opus-4-7"

# Developmental stages in order (C. elegans)
STAGES = ["early", "bean", "comma", "1.5fold", "2fold", "pretzel", "hatching", "hatched"]


@dataclass
class PerceptionOutput:
    """Standard output from any perception function."""

    stage: str  # "early", "bean", "comma", etc.
    reasoning: str  # Free-text explanation

    # Metadata for analysis
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    verification_triggered: bool = False
    phase_count: int = 1
    raw_response: str = ""  # Full API response text for debugging


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Return a module-level Anthropic client (created once)."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


async def call_claude(
    system: str,
    content: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    Thin async wrapper around the Anthropic messages API.

    Uses prompt caching on the system prompt (1h TTL).

    Returns the text of the first text block in the response.
    """
    client = _get_client()

    system_blocks = [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]

    response = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=[{"role": "user", "content": content}],
    )

    # Log cache metrics
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    if cache_read > 0 or cache_create > 0:
        logger.info(f"Cache: read={cache_read:,}, created={cache_create:,}")

    for block in response.content:
        if block.type == "text":
            return block.text

    return ""


async def call_claude_conversation(
    system: str,
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    Multi-turn conversation wrapper around the Anthropic messages API.

    Like call_claude, but accepts a full messages list instead of a single
    user content block, enabling follow-up turns.

    Returns the text of the first text block in the response.
    """
    client = _get_client()

    system_blocks = [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]

    response = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=messages,
    )

    # Log cache metrics
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    if cache_read > 0 or cache_create > 0:
        logger.info(f"Cache: read={cache_read:,}, created={cache_create:,}")

    for block in response.content:
        if block.type == "text":
            return block.text

    return ""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_stage_json(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from a VLM response.

    Tries code-fence first, then balanced-brace extraction, then whole-string.
    Returns the parsed dict, or an empty dict on failure.
    """
    # Strategy 1: JSON code block
    m = re.search(r"```json?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: balanced braces
    start = text.find("{")
    if start >= 0:
        depth = 0
        end = start
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Strategy 3: whole string
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    logger.warning("Failed to parse JSON from response")
    return {}


def response_to_output(raw: str) -> PerceptionOutput:
    """
    Parse a raw VLM text response into a PerceptionOutput.

    Falls back to stage="early" on parse failure.
    """
    data = parse_stage_json(raw)
    if not data:
        return PerceptionOutput(
            stage="early",
            reasoning="Parse error (no JSON found)",
            raw_response=raw,
        )

    stage = data.get("stage", "early")
    if stage not in STAGES:
        stage = "early"

    return PerceptionOutput(
        stage=stage,
        reasoning=data.get("reasoning", ""),
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Prompt-building helpers
# ---------------------------------------------------------------------------


def build_reference_content(
    references: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """
    Build Anthropic content blocks for reference images.

    Parameters
    ----------
    references : dict
        stage_name -> list of base64 JPEG images

    Returns
    -------
    list of content blocks (text + image dicts) with cache_control on the
    last block.
    """
    content: list[dict[str, Any]] = []
    content.append({"type": "text", "text": "REFERENCE EXAMPLES FOR EACH STAGE:"})

    for stage in STAGES:
        images = references.get(stage, [])
        if not images:
            continue
        content.append({"type": "text", "text": f"\n{stage.upper()}"})
        for img_b64 in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                }
            )

    # Mark final block for caching
    if content:
        content[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}

    return content


def build_history_text(history: list[dict]) -> str:
    """
    Format temporal context from history dicts.

    Parameters
    ----------
    history : list of dict
        Each dict has keys: timepoint, stage

    Returns
    -------
    Formatted string, or empty string if no history.
    """
    if not history:
        return ""

    lines = ["PREVIOUS OBSERVATIONS:"]
    for obs in history[-3:]:
        tp = obs.get("timepoint", "?")
        stage = obs.get("stage", "?")
        lines.append(f"- T{tp}: {stage}")

    return "\n".join(lines)
