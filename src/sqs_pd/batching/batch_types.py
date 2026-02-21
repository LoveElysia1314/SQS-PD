"""Dataclasses used by batch SQS workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CIFAnalysisResult:
    cif_name: str
    cif_path: Path
    occupancies: List
    candidates: List[Dict]
    disorder_types: List[str]
    valid_supercell_count: int
    num_disordered_sites: int
    per_site_denominators: List[List[int]]
    per_site_fraction_strs: List[List[str]]
    per_site_den_gcds: List[int]
    denominators_gcd: int
    total_site_entropy: float
    site_entropies: List[float]


@dataclass
class SQSTaskSpec:
    cif_name: str
    size: int
    supercell: Tuple[int, int, int]
    rss: float
    max_error: Optional[float] = None

    def to_key(self) -> Tuple:
        l, w, h = self.supercell
        return (self.cif_name, self.size, l, w, h)


@dataclass
class SQSResult:
    cif_name: str
    size: int
    l: int
    w: int
    h: int
    success: bool
    objective: Optional[float]
    time_s: float
    message: str = ""


@dataclass
class ProgressTracker:
    total_cifs: int
    completed_cifs: int = 0
    partial_cifs: List[str] = field(default_factory=list)
    pending_cifs: int = 0
    total_specs: int = 0
    completed_specs: int = 0

    def current_progress(self) -> str:
        return (
            f"CIFs: {self.completed_cifs}/{self.total_cifs} completed, "
            f"{len(self.partial_cifs)} partial, {self.pending_cifs} pending | "
            f"Specs: {self.completed_specs}/{self.total_specs}"
        )
