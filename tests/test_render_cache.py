"""Render cache: warm run must hit cache and return byte-identical frames."""

from __future__ import annotations

from dataclasses import dataclass

from gently_perception.render import CachedFrameSource, b64_bytes_equal


@dataclass
class _Frame:
    embryo_id: str
    timepoint: int
    ground_truth_stage: str | None
    image_b64: str
    top_image_b64: str | None = None
    side_image_b64: str | None = None
    midplane_b64: str | None = None
    zslices_b64: list | None = None


class _CountingSource:
    """Upstream source that records how many frames it actually rendered.
    Mimics ``OfflineTestset``: exposes embryo_ids, iter_embryo, get_timepoint_count.
    """

    def __init__(self, frames_by_embryo):
        self._d = frames_by_embryo
        self.render_calls = 0
        self.embryo_ids = list(frames_by_embryo)

    def get_timepoint_count(self, eid):
        return len(self._d[eid])

    def iter_embryo(self, eid):
        for fr in self._d[eid]:
            self.render_calls += 1
            yield fr

    def iter_all(self):
        for eid in self.embryo_ids:
            yield eid, self.iter_embryo(eid)


def _src(n=4):
    import base64
    return _CountingSource({"e1": [
        _Frame("e1", t, "1.5fold",
               image_b64=base64.b64encode(f"img-{t}".encode()).decode(),
               midplane_b64=base64.b64encode(f"mid-{t}".encode()).decode())
        for t in range(n)
    ]})


def test_cold_run_populates_cache(tmp_path):
    src = _src()
    cached = CachedFrameSource(src, cache_dir=tmp_path, session_id="s")
    out = [f for _, it in cached.iter_all() for f in it]
    assert len(out) == 4
    assert cached.stats() == {"hits": 0, "misses": 4}
    assert src.render_calls == 4
    assert len(list(tmp_path.glob("*.json"))) == 4


def test_warm_run_hits_cache_and_is_byte_identical(tmp_path):
    # Cold pass
    src1 = _src()
    cold = CachedFrameSource(src1, cache_dir=tmp_path, session_id="s")
    cold_frames = [f for _, it in cold.iter_all() for f in it]

    # Warm pass — fresh upstream that would re-render if asked
    src2 = _src()
    warm = CachedFrameSource(src2, cache_dir=tmp_path, session_id="s")
    warm_frames = [f for _, it in warm.iter_all() for f in it]

    assert warm.stats()["hits"] == 4
    assert src2.render_calls == 0  # upstream never touched on full-hit
    for c, w in zip(cold_frames, warm_frames, strict=True):
        assert c.timepoint == w.timepoint
        assert b64_bytes_equal(c.image_b64, w.image_b64)
        assert c.midplane_b64 == w.midplane_b64
        assert c.ground_truth_stage == w.ground_truth_stage


def test_different_session_ids_do_not_collide(tmp_path):
    src_a = _src(2)
    CachedFrameSource(src_a, cache_dir=tmp_path, session_id="A").iter_all()
    list(_drain(CachedFrameSource(src_a, cache_dir=tmp_path, session_id="A")))

    src_b = _src(2)
    cb = CachedFrameSource(src_b, cache_dir=tmp_path, session_id="B")
    list(_drain(cb))
    assert cb.stats()["misses"] == 2  # separate keyspace


def _drain(cached):
    for _, it in cached.iter_all():
        yield from it
