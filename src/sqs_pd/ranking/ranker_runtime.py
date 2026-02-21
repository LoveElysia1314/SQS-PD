"""Shared ranker runtime helpers.

Centralized logic for model artifact loading and candidate normalization/filtering.
This is intended to be reused by inference and batch flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import lightgbm as lgb


@dataclass(slots=True)
class RankerRuntime:
    ranker: lgb.Booster
    selected_features: List[str]
    model_file: Path
    config_file: Path


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def normalize_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize candidate rows and sort by (rss, size)."""
    normalized: List[Dict[str, Any]] = []
    for candidate in candidates:
        rss = _safe_float(candidate.get("rss"))
        supercell = candidate.get("supercell")
        size = candidate.get("size")

        if rss is None or supercell is None or size is None:
            continue

        try:
            sc_tuple = tuple(int(x) for x in supercell)
            if len(sc_tuple) != 3:
                continue
            normalized.append(
                {
                    "size": int(size),
                    "supercell": sc_tuple,
                    "rss": float(rss),
                    "max_error": _safe_float(candidate.get("max_error")),
                }
            )
        except Exception:
            continue

    normalized.sort(key=lambda item: (item["rss"], item["size"]))
    return normalized


def filter_min_rss_candidates(
    candidates: List[Dict[str, Any]],
    rss_tolerance: float = 1e-9,
) -> List[Dict[str, Any]]:
    """Keep only candidates whose RSS equals global minimum within tolerance."""
    normalized = normalize_candidates(candidates)
    if not normalized:
        return []

    min_rss = min(item["rss"] for item in normalized)
    return [
        item
        for item in normalized
        if abs(float(item["rss"]) - float(min_rss)) <= rss_tolerance
    ]


def load_ranker_runtime(
    preferred_model: Path,
    preferred_config: Path,
    fallback_model: Path,
    fallback_config: Path,
    missing_model_hint: Optional[str] = None,
    missing_config_hint: Optional[str] = None,
) -> RankerRuntime:
    """Load model + inference config from preferred/fallback paths."""
    model_file = preferred_model if preferred_model.exists() else fallback_model
    config_file = preferred_config if preferred_config.exists() else fallback_config

    if not model_file.exists():
        hint = f" {missing_model_hint}" if missing_model_hint else ""
        raise FileNotFoundError(f"Ranker model not found: {model_file}.{hint}".rstrip())

    if not config_file.exists():
        hint = f" {missing_config_hint}" if missing_config_hint else ""
        raise FileNotFoundError(
            f"Ranker inference config not found: {config_file}.{hint}".rstrip()
        )

    ranker = lgb.Booster(model_file=str(model_file))
    with config_file.open("r", encoding="utf-8") as file:
        cfg = json.load(file)

    selected_features = list(cfg.get("selected_features") or [])
    if not selected_features:
        raise ValueError(
            f"Invalid inference config: selected_features is empty ({config_file})"
        )

    return RankerRuntime(
        ranker=ranker,
        selected_features=selected_features,
        model_file=model_file,
        config_file=config_file,
    )
