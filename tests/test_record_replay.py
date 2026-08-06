"""Unit tests for the record/replay shim — no API, no volumes."""

from __future__ import annotations

import asyncio

import pytest

from experiments.analysis.record_replay import (
    CacheMiss,
    Recorder,
    install_record,
    install_replay,
    request_key,
)


def test_request_key_stable_and_order_sensitive():
    sys = "you are a classifier"
    c1 = [{"type": "text", "text": "hello"}]
    c2 = [{"type": "text", "text": "world"}]
    assert request_key(sys, c1) == request_key(sys, c1)
    assert request_key(sys, c1) != request_key(sys, c2)
    assert request_key(sys, c1 + c2) != request_key(sys, c2 + c1)


def test_request_key_hashes_image_bytes_not_literal():
    img = {"type": "image",
           "source": {"type": "base64", "media_type": "image/jpeg",
                      "data": "A" * 10000}}
    k = request_key("s", [img])
    assert "AAAA" not in k  # key is a sha, not the payload
    # Same bytes → same key; different bytes → different key
    assert k == request_key("s", [dict(img)])
    img2 = {**img, "source": {**img["source"], "data": "B" * 10000}}
    assert k != request_key("s", [img2])


def test_request_key_ignores_cache_control():
    a = [{"type": "text", "text": "x"}]
    b = [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]
    assert request_key("s", a) == request_key("s", b)


def test_recorder_roundtrip(tmp_path):
    rec = Recorder(tmp_path / "r.jsonl")
    rec.put("k1", "response-one", meta={"n_blocks": 3})
    rec.put("k1", "DUPLICATE")  # ignored
    assert rec.get("k1") == "response-one"

    # Reload from disk
    rec2 = Recorder(tmp_path / "r.jsonl")
    assert rec2.has("k1")
    assert rec2.get("k1") == "response-one"
    with pytest.raises(CacheMiss):
        rec2.get("missing")


def test_install_replay_intercepts_call_claude(tmp_path):
    """Replay shim must short-circuit perception._base.call_claude with the
    recorded response — and raise CacheMiss for unrecorded requests."""
    import perception._base as pb

    sys_prompt = "classify"
    content = [{"type": "text", "text": "frame T0"}]
    key = request_key(sys_prompt, content)

    rec = Recorder(tmp_path / "r.jsonl")
    rec.put(key, '{"stage": "comma", "reasoning": "stub"}')

    async def go():
        with install_replay(rec):
            out = await pb.call_claude(sys_prompt, content)
            assert out == '{"stage": "comma", "reasoning": "stub"}'
            with pytest.raises(CacheMiss):
                await pb.call_claude(sys_prompt, [{"type": "text", "text": "T1"}])
        # Restored after context exit
        assert pb.call_claude.__name__ == "call_claude"

    asyncio.run(go())


def test_install_record_wraps_and_restores(tmp_path, monkeypatch):
    """Record shim must call through to the real function and persist the
    response under the correct key."""
    import perception._base as pb

    async def fake_live(system, content, **_kw):
        return "LIVE:" + content[0]["text"]

    monkeypatch.setattr(pb, "call_claude", fake_live)

    rec = Recorder(tmp_path / "r.jsonl")

    async def go():
        with install_record(rec):
            out = await pb.call_claude("s", [{"type": "text", "text": "hello"}])
            assert out == "LIVE:hello"
        # Restored to the (monkeypatched) original
        assert pb.call_claude is fake_live

    asyncio.run(go())
    k = request_key("s", [{"type": "text", "text": "hello"}])
    assert rec.get(k) == "LIVE:hello"
