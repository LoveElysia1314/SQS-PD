"""Shared ranking feature utilities.

Reusable helpers for geometry, denominator-LCM and entropy features used by
both model inference and batch dataset generation.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Dict, List

import numpy as np

from ..foundation.fraction_utils import (
    compute_lcm,
    denominators_from_projected_occupancies,
)


def compute_geometry_features(l: int, w: int, h: int) -> Dict[str, float]:
    """Compute shared geometry features for a supercell shape."""
    volume = l * w * h
    geo_mean = volume ** (1.0 / 3.0)
    arith_mean = (l + w + h) / 3.0
    sphericity = geo_mean / arith_mean if arith_mean > 0 else 0.0
    face_sqrt_mean = (sqrt(l * w) + sqrt(w * h) + sqrt(l * h)) / 3.0
    face_sphericity = geo_mean / face_sqrt_mean if face_sqrt_mean > 0 else 0.0

    return {
        "volume": float(volume),
        "sphericity": float(sphericity),
        "face_sphericity": float(face_sphericity),
        "mean_dim": float(arith_mean),
    }


def compute_site_entropies(occupancies: List[Any], eps: float = 1e-12) -> List[float]:
    """Compute per-site Shannon entropy from nested occupancy probabilities."""
    if not occupancies:
        return []

    site_entropies: List[float] = []
    for site in occupancies:
        probs = list(site) if isinstance(site, (list, tuple)) else [site]
        try:
            values = [float(p) for p in probs]
        except Exception:
            continue

        site_sum = sum(values)
        if site_sum < 1.0 - 1e-6:
            values.append(1.0 - site_sum)

        arr = np.array(values, dtype=float)
        if arr.sum() <= 0:
            continue
        arr = arr / arr.sum()

        entropy = -float((arr * np.log(np.clip(arr, eps, 1.0))).sum())
        site_entropies.append(entropy)

    return site_entropies


def compute_total_entropy(occupancies: List[Any], eps: float = 1e-12) -> float:
    """Compute total Shannon entropy across disordered sites."""
    return float(sum(compute_site_entropies(occupancies, eps=eps)))


def extract_lcmm_from_occupancies(
    occupancies: List[Any],
    supercell_size: int,
) -> int:
    """Extract lcmm from occupancies projected to a specific supercell size."""
    denominators = denominators_from_projected_occupancies(
        occupancies,
        supercell_size=supercell_size,
        flatten_nested=True,
    )
    return compute_lcm(denominators)
