"""Machine learning: walk-forward combination of alpha signals."""

from __future__ import annotations

from .combiner import (
    CombinerConfig,
    CombinerResult,
    build_design_matrix,
    build_target,
    combine_signals,
)
from .splits import (
    Fold,
    PurgedWalkForward,
    assert_no_leakage,
    describe_folds,
)

__all__ = [
    "CombinerConfig",
    "CombinerResult",
    "Fold",
    "PurgedWalkForward",
    "assert_no_leakage",
    "build_design_matrix",
    "build_target",
    "combine_signals",
    "describe_folds",
]
