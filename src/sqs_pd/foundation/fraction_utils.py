"""Fraction and denominator utility helpers.

This module centralizes occupancy -> fraction/denominator conversion logic so
multiple workflows can share identical behavior.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Any, Dict, Iterable, List


def compute_lcm(values: Iterable[int]) -> int:
    """Compute LCM for integer iterable; non-positive values are treated as 1."""
    result = 1
    for raw in values:
        value = max(1, int(raw))
        result = result * value // gcd(result, value)
    return max(1, int(result))


def fraction_from_occupancy(
    value: Any,
    max_den: int = 100,
    fallback_round_digits: int = 6,
) -> Fraction:
    """Convert occupancy-like value to a bounded-denominator Fraction."""
    try:
        return Fraction(value).limit_denominator(max_den)
    except Exception:
        return Fraction(round(float(value), fallback_round_digits)).limit_denominator(
            max_den
        )


def flatten_occupancies(occupancies: List[Any]) -> List[Any]:
    """Flatten nested occupancy representation into a single list."""
    flat: List[Any] = []
    for item in occupancies:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def denominators_from_occupancies(
    occupancies: List[Any],
    max_den: int = 100,
    flatten_nested: bool = False,
) -> List[int]:
    """Extract denominator list from occupancy values."""
    source = flatten_occupancies(occupancies) if flatten_nested else occupancies
    denominators: List[int] = []
    for occ in source:
        frac = fraction_from_occupancy(occ, max_den=max_den)
        denominators.append(int(frac.denominator))
    return denominators


def denominators_from_projected_occupancies(
    occupancies: List[Any],
    supercell_size: int,
    flatten_nested: bool = False,
) -> List[int]:
    """Extract reduced denominators from nearest fractions projected to a size.

    For each occupancy ``occ``, the nearest fraction under the given supercell size is
    ``round(occ * size) / size``. The returned denominator is the reduced denominator
    of ``Fraction(round(occ * size), size)``.
    """
    size = int(supercell_size)
    if size <= 0:
        raise ValueError(f"supercell_size must be positive, got {supercell_size}")

    source = flatten_occupancies(occupancies) if flatten_nested else occupancies
    denominators: List[int] = []
    for occ in source:
        numerator = int(round(float(occ) * size))
        frac = Fraction(numerator, size)
        denominators.append(int(frac.denominator))
    return denominators


def summarize_denominators(
    occupancies: List[Any],
    supercell_size: int,
) -> Dict[str, Any]:
    """Return denominator summary stats using projected nearest fractions at size."""
    dens = denominators_from_projected_occupancies(
        occupancies,
        supercell_size=supercell_size,
        flatten_nested=True,
    )

    if not dens:
        return {
            "lcmm": 1,
            "min_den": None,
            "max_den": None,
            "mean_den": None,
            "median_den": None,
            "unique_den_count": 0,
        }

    dens_sorted = sorted(dens)
    return {
        "lcmm": compute_lcm(dens),
        "min_den": min(dens),
        "max_den": max(dens),
        "mean_den": sum(dens) / len(dens),
        "median_den": dens_sorted[len(dens) // 2],
        "unique_den_count": len(set(dens)),
    }


def per_site_fraction_stats(
    occupancies: List[Any], max_den: int = 100
) -> Dict[str, Any]:
    """Compute denominator/fraction stats for per-site occupancies."""

    def _gcd_list(values: List[int]) -> int:
        if not values:
            return 1
        value = values[0]
        for item in values[1:]:
            value = gcd(value, item)
        return max(1, int(value))

    per_site_fraction_strs: List[List[str]] = []
    per_site_denominators: List[List[int]] = []
    per_site_den_gcds: List[int] = []
    all_denominators: List[int] = []

    for site in occupancies:
        site_items = site if isinstance(site, (list, tuple)) else [site]
        site_fractions = [
            fraction_from_occupancy(occ, max_den=max_den) for occ in site_items
        ]
        site_denominators = [int(frac.denominator) for frac in site_fractions]
        site_fraction_strs = [
            f"{frac.numerator}/{frac.denominator}" for frac in site_fractions
        ]

        per_site_fraction_strs.append(site_fraction_strs)
        per_site_denominators.append(site_denominators)
        per_site_den_gcds.append(_gcd_list(site_denominators))
        all_denominators.extend(site_denominators)

    return {
        "num_disordered_sites": len(occupancies),
        "per_site_fraction_strs": per_site_fraction_strs,
        "per_site_denominators": per_site_denominators,
        "per_site_den_gcds": per_site_den_gcds,
        "denominators_gcd": _gcd_list(all_denominators) if all_denominators else 1,
    }
