"""Candidate selection strategies for batch SQS runs."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .batch_common import DEFAULT_RANKER_CONFIG_PATH, DEFAULT_RANKER_MODEL_PATH
from .batch_types import CIFAnalysisResult
from .pipeline_layout import resolve_project_path
from ..ranking.ranker_features import (
    compute_geometry_features,
    extract_lcmm_from_occupancies,
)
from ..ranking.ranker_runtime import (
    RankerRuntime,
    filter_min_rss_candidates,
    load_ranker_runtime,
)


class SupercellSelector:
    """Select candidates in min-rss mode or model-topk mode."""

    def __init__(
        self,
        mode: str = "min-rss",
        top_k: int = 5,
        model_path: Optional[str] = None,
        model_config: Optional[str] = None,
    ):
        self.mode = mode
        self.top_k = max(1, int(top_k))
        self.model_path = resolve_project_path(model_path or DEFAULT_RANKER_MODEL_PATH)
        self.model_config = resolve_project_path(
            model_config or DEFAULT_RANKER_CONFIG_PATH
        )
        self.runtime: Optional[RankerRuntime] = None

        if self.mode == "model-topk":
            self._load_ranker()

    def _load_ranker(self):
        fallback_model = resolve_project_path(
            "artifacts/models/ml_ranker/ranker_model.txt"
        )
        fallback_cfg = resolve_project_path(
            "artifacts/models/ml_ranker/ranker_inference_config.json"
        )

        self.runtime = load_ranker_runtime(
            preferred_model=self.model_path,
            preferred_config=self.model_config,
            fallback_model=fallback_model,
            fallback_config=fallback_cfg,
            missing_model_hint="Train and promote a model first, or pass --model-path.",
            missing_config_hint="Please retrain model to generate ranker_inference_config.json.",
        )

    @staticmethod
    def _filter_min_rss_candidates(candidates: List[Dict]) -> List[Dict]:
        return filter_min_rss_candidates(candidates)

    def _build_feature_vector(
        self, analysis: CIFAnalysisResult, candidate: Dict
    ) -> Dict[str, float]:
        l, w, h = candidate["supercell"]
        geom = compute_geometry_features(int(l), int(w), int(h))
        best_size = int(analysis.candidates[0]["size"])
        lcmm = extract_lcmm_from_occupancies(
            analysis.occupancies,
            supercell_size=best_size,
        )

        return {
            "volume": geom["volume"],
            "sphericity": geom["sphericity"],
            "face_sphericity": geom["face_sphericity"],
            "mean_dim": geom["mean_dim"],
            "lcmm": lcmm,
            "num_disordered_sites": analysis.num_disordered_sites,
            "valid_supercell_count": analysis.valid_supercell_count,
            "total_site_entropy": analysis.total_site_entropy,
        }

    def _rank_topk_candidates(
        self, analysis: CIFAnalysisResult, candidates: List[Dict]
    ) -> List[Dict]:
        if not candidates:
            return []
        if self.runtime is None:
            raise RuntimeError("Ranker not loaded for model-topk mode")

        matrix = []
        for cand in candidates:
            feat_map = self._build_feature_vector(analysis, cand)
            matrix.append(
                [
                    float(feat_map.get(feature, 0.0))
                    for feature in self.runtime.selected_features
                ]
            )

        X = np.array(matrix, dtype=float)

        scores = self.runtime.ranker.predict(X)
        ranked = []
        for cand, score in zip(candidates, scores):
            c = dict(cand)
            c["model_score"] = float(score)
            ranked.append(c)

        ranked.sort(key=lambda x: x["model_score"], reverse=True)
        return ranked[: self.top_k]

    def select_candidates(self, analysis: CIFAnalysisResult) -> List[Dict]:
        candidates = analysis.candidates
        if self.mode == "model-topk":
            constrained = self._filter_min_rss_candidates(candidates)
            return self._rank_topk_candidates(analysis, constrained)
        return self._filter_min_rss_candidates(candidates)
