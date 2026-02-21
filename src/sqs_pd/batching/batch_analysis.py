"""Disorder analysis and feature extraction for batch SQS workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..interface.api import batch_recommend_supercells
from .batch_types import CIFAnalysisResult
from ..foundation.fraction_utils import per_site_fraction_stats, summarize_denominators
from ..ranking.ranker_features import compute_site_entropies
from ..ranking.ranker_runtime import normalize_candidates
from ..runtime.io_utils import write_json_file


class DisorderAnalyzer:
    def __init__(self):
        self.cif_analyses: Dict[str, CIFAnalysisResult] = {}

    def ingest_analysis_result(
        self,
        cif_path: Path,
        analysis: Dict[str, Any],
    ) -> Optional[CIFAnalysisResult]:
        occupancies = analysis.get("occupancies", [])

        if not occupancies or not analysis.get("optimization_success"):
            return None

        candidates = self._normalize_candidates(analysis.get("all_candidates", []))
        valid_supercell_count = len(candidates)

        per_site_info = self._compute_per_site_stats(occupancies)
        total_entropy, site_entropies = self._compute_entropy(occupancies)

        result = CIFAnalysisResult(
            cif_name=cif_path.name,
            cif_path=cif_path,
            occupancies=occupancies,
            candidates=candidates,
            disorder_types=analysis.get("disorder_types", []),
            valid_supercell_count=valid_supercell_count,
            num_disordered_sites=per_site_info["num_disordered_sites"],
            per_site_denominators=per_site_info["per_site_denominators"],
            per_site_fraction_strs=per_site_info["per_site_fraction_strs"],
            per_site_den_gcds=per_site_info["per_site_den_gcds"],
            denominators_gcd=per_site_info["denominators_gcd"],
            total_site_entropy=total_entropy,
            site_entropies=site_entropies,
        )

        self.cif_analyses[cif_path.name] = result
        return result

    def analyze_batch(self, cif_files: List[Path], verbose: bool = False) -> None:
        analysis_results = batch_recommend_supercells(cif_files, verbose=verbose)
        for cif_path, analysis in zip(cif_files, analysis_results):
            self.ingest_analysis_result(cif_path, analysis)

    def _normalize_candidates(self, candidates: List[Dict]) -> List[Dict]:
        return normalize_candidates(candidates)

    def _compute_per_site_stats(self, occupancies: List, max_den: int = 100) -> Dict:
        return per_site_fraction_stats(occupancies, max_den=max_den)

    def _compute_entropy(self, occupancies: List) -> Tuple[float, List[float]]:
        site_entropies = compute_site_entropies(occupancies)
        return float(sum(site_entropies)), site_entropies

    def compute_denom_stats(
        self,
        occupancies: List,
        supercell_size: int,
    ) -> Dict:
        return summarize_denominators(
            occupancies,
            supercell_size=supercell_size,
        )

    def get_cif_info(self, cif_name: str) -> Optional[CIFAnalysisResult]:
        return self.cif_analyses.get(cif_name)

    def save_summary(self, output_path: Path, total_files: int):
        summary = {
            "total_files": total_files,
            "disordered_count": len(self.cif_analyses),
            "disordered_cifs": {
                name: {
                    "cif_path": str(info.cif_path),
                    "occupancies": info.occupancies,
                    "num_candidates": len(info.candidates),
                    "candidates": info.candidates,
                    "disorder_types": info.disorder_types,
                    "valid_supercell_count": info.valid_supercell_count,
                    "num_disordered_sites": info.num_disordered_sites,
                    "per_site_fraction_strs": info.per_site_fraction_strs,
                    "per_site_denominators": info.per_site_denominators,
                    "per_site_den_gcds": info.per_site_den_gcds,
                    "denominators_gcd": info.denominators_gcd,
                    "total_site_entropy": info.total_site_entropy,
                    "site_entropies": info.site_entropies,
                }
                for name, info in self.cif_analyses.items()
            },
        }

        write_json_file(output_path, summary)

    def save_features_csv(self, output_path: Path):
        import csv

        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "cif_file",
                    "num_disordered_sites",
                    "denominators_gcd",
                    "valid_supercell_count",
                    "num_candidates_after_filter",
                    "per_site_denominators_str",
                    "total_site_entropy",
                ],
            )
            writer.writeheader()

            for name, info in self.cif_analyses.items():
                per_site_den_str = "|".join(
                    ",".join(str(x) for x in site)
                    for site in info.per_site_denominators
                )
                writer.writerow(
                    {
                        "cif_file": name,
                        "num_disordered_sites": info.num_disordered_sites,
                        "denominators_gcd": info.denominators_gcd,
                        "valid_supercell_count": info.valid_supercell_count,
                        "num_candidates_after_filter": len(info.candidates),
                        "per_site_denominators_str": per_site_den_str,
                        "total_site_entropy": info.total_site_entropy,
                    }
                )
