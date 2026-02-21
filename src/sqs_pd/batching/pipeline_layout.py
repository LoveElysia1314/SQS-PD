"""Pipeline path/layout helpers for batch dataset workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PIPELINE_DIR = "artifacts/pipeline"
DEFAULT_SQS_RESULTS_REL = f"{DEFAULT_PIPELINE_DIR}/sqs_all_results.csv"
DEFAULT_ML_DATASET_REL = f"{DEFAULT_PIPELINE_DIR}/ml_dataset.csv"
DEFAULT_SUMMARY_JSON_REL = f"{DEFAULT_PIPELINE_DIR}/disorder_summary.json"


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve path relative to project root unless absolute."""
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def derive_disorder_features_path(summary_json_path: Path) -> Path:
    """Derive disorder features csv path from summary json location."""
    return summary_json_path.parent / "disorder_features.csv"


@dataclass(slots=True)
class PipelineOutputPaths:
    """Resolved batch pipeline output paths."""

    sqs_csv: Path
    ml_csv: Path
    summary_json: Path
    disorder_features_csv: Path


def resolve_pipeline_output_paths(
    out_csv: str | Path,
    out_ml: str | Path,
    out_json: str | Path,
) -> PipelineOutputPaths:
    """Resolve user-configured output paths and derived artifacts."""
    resolved_json = resolve_project_path(out_json)
    return PipelineOutputPaths(
        sqs_csv=resolve_project_path(out_csv),
        ml_csv=resolve_project_path(out_ml),
        summary_json=resolved_json,
        disorder_features_csv=derive_disorder_features_path(resolved_json),
    )
