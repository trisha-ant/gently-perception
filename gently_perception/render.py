"""Disk-cached frame rendering [P5 — dev-loop speedup].

The legacy ``benchmark/testset.py`` decodes a multi-MB TIFF and renders five
images per timepoint on every iteration. This module wraps any
``FrameSource`` with a content-addressed disk cache so the second and
subsequent runs skip the TIFF entirely.

The cache is purely additive: the wrapped source is asked for each frame
exactly once; cached frames are byte-identical to what the source produced,
so record-replay equivalence is preserved.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "frames"

_B64_FIELDS = ("image_b64", "top_image_b64", "side_image_b64", "midplane_b64")
_LIST_FIELDS = ("zslices_b64",)


@dataclass
class CachedFrame:
    """Minimal frame carrying the same b64 fields as ``benchmark.testset.TestCase``."""

    embryo_id: str
    timepoint: int
    ground_truth_stage: str | None
    image_b64: str
    top_image_b64: str | None = None
    side_image_b64: str | None = None
    midplane_b64: str | None = None
    zslices_b64: list[str] | None = None
    volume = None  # not cached — too large


def _key(session_id: str, embryo_id: str, timepoint: int) -> str:
    raw = f"{session_id}|{embryo_id}|{timepoint}"
    return hashlib.sha1(raw.encode()).hexdigest()


class CachedFrameSource:
    """Wrap a FrameSource (e.g. ``OfflineTestset``) with a disk cache.

    Parameters
    ----------
    source:
        The underlying source. Must expose ``iter_all()`` and either a
        ``session_path`` attribute or accept ``session_id`` explicitly.
    cache_dir:
        Where to store cached frames. One ``{sha1}.json`` per frame (b64
        strings stored verbatim so cached bytes equal source bytes).
    """

    def __init__(self, source, *, cache_dir: Path | None = None,
                 session_id: str | None = None):
        self._source = source
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if session_id is None:
            sp = getattr(source, "session_path", None)
            session_id = str(sp.resolve()) if sp else "default"
        self._session_id = session_id
        self.hits = 0
        self.misses = 0

    @property
    def embryo_ids(self):
        return self._source.embryo_ids

    def _path(self, embryo_id: str, timepoint: int) -> Path:
        return self.cache_dir / f"{_key(self._session_id, embryo_id, timepoint)}.json"

    def _load(self, embryo_id: str, timepoint: int) -> CachedFrame | None:
        p = self._path(embryo_id, timepoint)
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return CachedFrame(**d)

    def _store(self, fr) -> CachedFrame:
        cf = CachedFrame(
            embryo_id=fr.embryo_id,
            timepoint=fr.timepoint,
            ground_truth_stage=fr.ground_truth_stage,
            image_b64=fr.image_b64,
            **{f: getattr(fr, f, None) for f in _B64_FIELDS[1:]},
            **{f: getattr(fr, f, None) for f in _LIST_FIELDS},
        )
        p = self._path(cf.embryo_id, cf.timepoint)
        p.write_text(json.dumps(cf.__dict__))
        return cf

    def _cached_prefix_len(self, embryo_id: str) -> int:
        tp = 0
        while self._path(embryo_id, tp).exists():
            tp += 1
        return tp

    def _iter_embryo_cached(self, embryo_id: str,
                            make_src_iter) -> Iterator[CachedFrame]:
        prefix = self._cached_prefix_len(embryo_id)
        count: int | None = None
        if hasattr(self._source, "get_timepoint_count"):
            count = self._source.get_timepoint_count(embryo_id)

        # Full hit: yield from cache only — upstream generator is never created,
        # so no TIFF decode or render happens.
        if count is not None and prefix >= count:
            for tp in range(count):
                cf = self._load(embryo_id, tp)
                assert cf is not None
                self.hits += 1
                yield cf
            return

        # Partial/cold: yield cached prefix, then drive upstream for the rest.
        for tp in range(prefix):
            cf = self._load(embryo_id, tp)
            assert cf is not None
            self.hits += 1
            yield cf
        for fr in make_src_iter():
            if fr.timepoint < prefix:
                continue
            self.misses += 1
            yield self._store(fr)

    def iter_all(self) -> Iterable[tuple[str, Iterator[CachedFrame]]]:
        # Avoid materialising upstream iterators eagerly: capture a thunk
        # per embryo so warm runs can skip creation entirely.
        embryo_ids = list(getattr(self._source, "embryo_ids", []))
        if embryo_ids and hasattr(self._source, "iter_embryo"):
            for eid in embryo_ids:
                yield eid, self._iter_embryo_cached(
                    eid, lambda e=eid: self._source.iter_embryo(e))
        else:
            for eid, src_iter in self._source.iter_all():
                yield eid, self._iter_embryo_cached(eid, lambda it=src_iter: it)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


def b64_bytes_equal(a: str, b: str) -> bool:
    """Compare two base64 strings by decoded bytes (whitespace-insensitive)."""
    return base64.b64decode(a) == base64.b64decode(b)
