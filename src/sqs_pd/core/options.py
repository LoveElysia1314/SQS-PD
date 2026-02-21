"""Public options objects.

These dataclasses provide a stable way to pass configuration into the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..foundation.constants import (
    DEFAULT_ITERATIONS,
    OCCUPANCY_TOLERANCE,
)


@dataclass(slots=True)
class SQSOptions:
    """Options that control SQS generation."""

    supercell: Optional[Tuple[int, int, int]] = None
    iterations: int = DEFAULT_ITERATIONS
    output_file: Optional[str] = None

    # Diagnostics / artifacts
    return_all: bool = False
    include_analysis_details: bool = False
    log_file: Optional[str] = None

    # Logging
    console_log: bool = False
    debug: bool = False


@dataclass(slots=True)
class DisorderAnalysisOptions:
    """Options for CIF disorder analysis."""

    group_by: str = "label"
    tol: float = OCCUPANCY_TOLERANCE
