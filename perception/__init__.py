"""
Modular perception functions for VLM-based embryo stage classification.

Each function has the same async signature::

    async def perceive(
        image_b64: str,
        references: dict[str, list[str]],
        history: list[dict],
        timepoint: int,
    ) -> PerceptionOutput

The FUNCTIONS registry maps variant names to their perceive() callables.
This is the code the agent modifies — equivalent to train.py in autoresearch.
"""

from ._base import PerceptionOutput  # noqa: F401

# Lazy registry — populated on first access via get_functions()
_FUNCTIONS: dict | None = None


def get_functions() -> dict:
    """Return the registry mapping variant name -> perceive callable."""
    global _FUNCTIONS
    if _FUNCTIONS is not None:
        return _FUNCTIONS

    from .minimal import perceive_minimal
    from .descriptive import perceive_descriptive
    from .minimal_multishot import perceive_minimal_multishot
    from .descriptive_multishot import perceive_descriptive_multishot
    from .contrastive import perceive_contrastive
    from .temporal import perceive_temporal
    from .temporal_v2 import perceive_temporal_v2
    from .temporal_v3 import perceive_temporal_v3
    from .scientific import perceive_scientific
    from .hybrid import perceive_hybrid
    from .unified import perceive_unified
    from .ensemble import perceive_ensemble
    from .compare import perceive_compare
    from .changegate import perceive_changegate
    from .duration_aware import perceive_duration_aware
    from .zslice import perceive_zslice
    from .zslice_asym import perceive_zslice_asym
    from .zslice_multi import perceive_zslice_multi
    from .fillpct import perceive_fillpct
    from .pairwise import perceive_pairwise
    from .hybrid_fillpct import perceive_hybrid_fillpct

    _FUNCTIONS = {
        "minimal": perceive_minimal,
        "descriptive": perceive_descriptive,
        "minimal_multishot": perceive_minimal_multishot,
        "descriptive_multishot": perceive_descriptive_multishot,
        "contrastive": perceive_contrastive,
        "temporal": perceive_temporal,
        "temporal_v2": perceive_temporal_v2,
        "temporal_v3": perceive_temporal_v3,
        "scientific": perceive_scientific,
        "hybrid": perceive_hybrid,
        "unified": perceive_unified,
        "ensemble": perceive_ensemble,
        "compare": perceive_compare,
        "changegate": perceive_changegate,
        "duration_aware": perceive_duration_aware,
        "zslice": perceive_zslice,
        "zslice_asym": perceive_zslice_asym,
        "zslice_multi": perceive_zslice_multi,
        "fillpct": perceive_fillpct,
        "pairwise": perceive_pairwise,
        "hybrid_fillpct": perceive_hybrid_fillpct,
    }
    return _FUNCTIONS
