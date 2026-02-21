"""Shared constants and tiny helpers for batch SQS modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

DEFAULT_RANKER_MODEL_PATH = "artifacts/models/default_ml_ranker/ranker_model.txt"
DEFAULT_RANKER_CONFIG_PATH = (
    "artifacts/models/default_ml_ranker/ranker_inference_config.json"
)


def str_to_bool(value: str) -> bool:
    return str(value).lower() in ("true", "1", "yes")


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def run_batch_analysis(
    cif_files: List[Union[str, Path]],
    analyze_single: Callable[[Union[str, Path], float, bool], Dict[str, Any]],
    max_error: float = 0.0005,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run single-file analyzer across many CIF files with stable error policy."""
    results: List[Dict[str, Any]] = []

    for index, cif_file in enumerate(cif_files, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"分析 [{index}/{len(cif_files)}]: {cif_file}")
            print(f"{'='*60}\n")

        try:
            result = analyze_single(cif_file, max_error=max_error, verbose=verbose)
            results.append(result)
        except Exception as exc:
            error_result = {
                "cif_file": str(cif_file),
                "optimization_success": False,
                "message": f"Error: {str(exc)}",
                "error": str(exc),
            }
            results.append(error_result)

            if verbose:
                print(f"❌ 分析失败: {exc}\n")

    return results
