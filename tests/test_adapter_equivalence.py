"""The adapter must produce byte-identical API requests to the legacy path.

This is the load-bearing invariant for record-replay equivalence (Phase A):
if old and new emit the same (system, content) to ``call_claude``, replaying
old recordings through the new loop yields identical predictions by
construction.
"""

from __future__ import annotations

import asyncio

import pytest

from gently_perception.types import FrameInput, Obs
from experiments.analysis.record_replay import request_key
from experiments.harness.discover import discover_variants
from perception import get_functions


CASES = ["hybrid", "scientific", "temporal", "minimal", "fillpct",
         "multimeasure"]


@pytest.mark.parametrize("variant", CASES)
def test_adapter_emits_identical_request(variant, monkeypatch):
    # Load both registries first so all bindings exist before patching.
    legacy_fns = get_functions()
    new_fns = discover_variants()

    captured: list[str] = []

    async def capture(system, content, **_kw):
        captured.append(request_key(system, content))
        return '{"stage": "comma", "reasoning": "stub"}'

    from experiments.analysis.record_replay import _patch_targets
    for mod, attr in _patch_targets():
        monkeypatch.setattr(mod, attr, capture)

    # Shared inputs
    img = "QUJDRA=="  # b"ABCD"
    refs = {"early": ["ZWFybHk="], "comma": ["Y29tbWE="]}
    history = [{"timepoint": 0, "stage": "early"},
               {"timepoint": 1, "stage": "early"},
               {"timepoint": 2, "stage": "comma"}]
    timepoint = 3

    # Old path
    legacy = legacy_fns[variant]
    asyncio.run(legacy(image_b64=img, references=refs, history=list(history),
                       timepoint=timepoint))
    assert len(captured) >= 1, f"{variant}: capture never fired (patch missed)"
    old_keys = captured.copy()
    captured.clear()

    # New path via adapter
    new = new_fns[variant]
    fi = FrameInput(image_b64=img,
                    history=tuple(Obs(h["timepoint"], h["stage"]) for h in history),
                    timepoint=timepoint, references=refs)
    asyncio.run(new(fi))

    assert captured == old_keys, (
        f"{variant}: adapter produced different request(s).\n"
        f"old={old_keys}\nnew={captured}"
    )
