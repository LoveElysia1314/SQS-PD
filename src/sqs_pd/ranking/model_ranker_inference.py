"""Model-based supercell ranking inference.

Provides reusable inference API for ranking legal supercell candidates from a CIF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..runtime.dry_run import analyze_cif_and_recommend_supercell
from ..batching.pipeline_layout import resolve_project_path
from .ranker_runtime import (
    filter_min_rss_candidates,
    load_ranker_runtime,
    normalize_candidates,
)
from .ranker_features import (
    compute_geometry_features,
    compute_total_entropy,
    extract_lcmm_from_occupancies,
)

DEFAULT_RANKER_MODEL_PATH = "artifacts/models/default_ml_ranker/ranker_model.txt"
DEFAULT_RANKER_CONFIG_PATH = (
    "artifacts/models/default_ml_ranker/ranker_inference_config.json"
)


@dataclass(slots=True)
class RankedSupercell:
    rank: int
    supercell: Tuple[int, int, int]
    size: int
    model_score: float
    rss: Optional[float] = None
    max_error: Optional[float] = None


def _compute_total_entropy(occupancies: List[Any]) -> float:
    return compute_total_entropy(occupancies)


def _compute_geometry_features(l: int, w: int, h: int) -> Dict[str, float]:
    return compute_geometry_features(l, w, h)


def _normalize_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return normalize_candidates(candidates)


def _extract_lcmm_from_occupancies(
    occupancies: List[Any],
    supercell_size: int,
) -> int:
    return extract_lcmm_from_occupancies(
        occupancies,
        supercell_size=supercell_size,
    )


def _load_ranker_and_config(
    model_path: Optional[str | Path] = None,
    model_config_path: Optional[str | Path] = None,
) -> tuple[Any, List[str], Path, Path]:
    preferred_model = resolve_project_path(model_path or DEFAULT_RANKER_MODEL_PATH)
    preferred_cfg = resolve_project_path(
        model_config_path or DEFAULT_RANKER_CONFIG_PATH
    )

    fallback_model = resolve_project_path("artifacts/models/ml_ranker/ranker_model.txt")
    fallback_cfg = resolve_project_path(
        "artifacts/models/ml_ranker/ranker_inference_config.json"
    )

    runtime = load_ranker_runtime(
        preferred_model=preferred_model,
        preferred_config=preferred_cfg,
        fallback_model=fallback_model,
        fallback_config=fallback_cfg,
        missing_config_hint=(
            "Please retrain ranker to generate ranker_inference_config.json."
        ),
    )

    return (
        runtime.ranker,
        runtime.selected_features,
        runtime.model_file,
        runtime.config_file,
    )


def recommend_supercells_by_model(
    cif_file: str | Path,
    top_k: Optional[int] = None,
    max_error: float = 0.0005,
    model_path: Optional[str | Path] = None,
    model_config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Rank legal supercell candidates by trained model score.

    Returns dry-run analysis plus model-ranked legal candidate list.
    """

    analysis = analyze_cif_and_recommend_supercell(
        cif_file=cif_file, max_error=max_error
    )
    if not analysis.get("optimization_success", False):
        return {
            "success": False,
            "cif_file": str(cif_file),
            "message": analysis.get("message", "Optimization failed"),
            "analysis": analysis,
            "ranked_candidates": [],
        }

    candidates_all = _normalize_candidates(analysis.get("all_candidates", []))
    candidates = filter_min_rss_candidates(candidates_all)
    if not candidates:
        return {
            "success": True,
            "cif_file": str(cif_file),
            "message": "No min-rss candidates for ranking",
            "analysis": analysis,
            "ranked_candidates": [],
            "top_k": 0,
            "num_candidates": 0,
        }

    occupancies = analysis.get("occupancies", [])
    num_disordered_sites = int(analysis.get("num_disordered_sites", len(occupancies)))
    valid_supercell_count = int(analysis.get("num_candidates", len(candidates_all)))
    total_site_entropy = _compute_total_entropy(occupancies)
    best_size = int(candidates[0]["size"])
    lcmm = _extract_lcmm_from_occupancies(occupancies, supercell_size=best_size)

    ranker, selected_features, used_model, used_cfg = _load_ranker_and_config(
        model_path=model_path,
        model_config_path=model_config_path,
    )

    feature_matrix = []
    for cand in candidates:
        l, w, h = cand["supercell"]
        geom = _compute_geometry_features(l, w, h)
        feature_map = {
            **geom,
            "lcmm": float(lcmm),
            "num_disordered_sites": float(num_disordered_sites),
            "valid_supercell_count": float(valid_supercell_count),
            "total_site_entropy": float(total_site_entropy),
        }
        feature_matrix.append(
            [float(feature_map.get(name, 0.0)) for name in selected_features]
        )

    X = np.array(feature_matrix, dtype=float)

    scores = ranker.predict(X)

    ranked_rows: List[RankedSupercell] = []
    for cand, score in zip(candidates, scores):
        ranked_rows.append(
            RankedSupercell(
                rank=0,
                supercell=cand["supercell"],
                size=cand["size"],
                model_score=float(score),
                rss=cand.get("rss"),
                max_error=cand.get("max_error"),
            )
        )

    ranked_rows.sort(key=lambda x: x.model_score, reverse=True)
    for idx, row in enumerate(ranked_rows, 1):
        row.rank = idx

    if top_k is not None:
        top_k = max(1, int(top_k))
        ranked_rows = ranked_rows[:top_k]

    return {
        "success": True,
        "cif_file": str(cif_file),
        "message": "Model ranking completed",
        "analysis": analysis,
        "num_candidates": len(candidates),
        "num_candidates_all": len(candidates_all),
        "constraint": "min-rss-only",
        "top_k": len(ranked_rows),
        "ranked_candidates": [
            {
                "rank": item.rank,
                "supercell": item.supercell,
                "size": item.size,
                "model_score": item.model_score,
                "rss": item.rss,
                "max_error": item.max_error,
            }
            for item in ranked_rows
        ],
        "recommended_supercell": ranked_rows[0].supercell if ranked_rows else None,
        "model_file": str(used_model),
        "model_config": str(used_cfg),
        "selected_features": selected_features,
    }
