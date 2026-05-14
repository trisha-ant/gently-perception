"""Model override shim: verify it injects model/thinking and omits params
that 400 on newer model versions."""

from __future__ import annotations

import asyncio

import pytest


def test_override_injects_model_and_thinking(monkeypatch):
    import experiments.harness.model_override as mo

    captured: dict = {}

    class _FakeResp:
        class _Block:
            type = "text"
            text = '{"stage":"comma","reasoning":"x"}'
        content = [_Block()]

    def fake_create(**kw):
        captured.update(kw)
        return _FakeResp()

    class _FakeMessages:
        create = staticmethod(fake_create)

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(mo, "_client", _FakeClient())

    # Load a variant so its call_claude binding is in sys.modules.
    from perception import hybrid as _  # noqa: F401
    import perception.hybrid as ph

    async def go():
        with mo.install_model_override("claude-opus-4-6", "adaptive"):
            return await ph.call_claude(system="s",
                                        content=[{"type": "text", "text": "x"}])

    out = asyncio.run(go())
    assert out == '{"stage":"comma","reasoning":"x"}'
    assert captured["model"] == "claude-opus-4-6"
    assert captured["thinking"] == {"type": "adaptive"}
    # Must NOT send sampling params (opus-4-7 400s on them)
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "output_config" not in captured  # xhigh would fail on 4.6
    # Prompt caching on system block
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_override_thinking_none_omits_block(monkeypatch):
    import experiments.harness.model_override as mo

    captured: dict = {}

    class _FakeResp:
        content = [type("B", (), {"type": "text", "text": "ok"})()]

    monkeypatch.setattr(
        mo, "_client",
        type("C", (), {"messages": type("M", (), {
            "create": staticmethod(lambda **kw: (captured.update(kw), _FakeResp())[1])
        })()})()
    )

    import perception.minimal as pm

    async def go():
        with mo.install_model_override("claude-opus-4-7", None):
            return await pm.call_claude(system="s",
                                        content=[{"type": "text", "text": "x"}])

    asyncio.run(go())
    assert captured["model"] == "claude-opus-4-7"
    assert "thinking" not in captured


def test_override_restores_on_exit(monkeypatch):
    import experiments.harness.model_override as mo
    import perception._base as pb

    orig = pb.call_claude
    monkeypatch.setattr(mo, "_client",
                        type("C", (), {"messages": type("M", (), {
                            "create": staticmethod(lambda **kw: type(
                                "R", (), {"content": []})())
                        })()})())

    with mo.install_model_override("m", None):
        assert pb.call_claude is not orig
    assert pb.call_claude is orig
