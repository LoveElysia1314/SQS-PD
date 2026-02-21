"""Shared file output utilities for consistent artifact handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


def resolve_output_profile(
    output_profile: Optional[str],
    save_json: bool,
    save_txt_report: bool,
) -> Tuple[bool, bool, bool]:
    """Resolve output profile into (write_cif, save_json, save_txt_report)."""
    if not output_profile:
        return True, save_json, save_txt_report

    profile = output_profile.strip().lower()
    if profile in {"silent", "none"}:
        return False, False, False
    if profile in {"cif", "cif-only", "cif_only"}:
        return True, False, False
    if profile in {"logs", "log"}:
        return True, True, False
    if profile in {"full", "all"}:
        return True, True, True
    raise ValueError(
        f"Unknown output_profile: {output_profile}. Use silent/cif/logs/full."
    )


def normalize_artifact_policy(
    artifact_policy: Optional[str],
    default: str = "best",
) -> str:
    """Normalize artifact policy into canonical values: none/best/all."""
    policy = (artifact_policy or default).strip().lower()
    if policy not in {"none", "best", "all"}:
        raise ValueError("artifact_policy must be one of: none, best, all")
    return policy


def ensure_parent_dir(path: Union[str, Path]) -> Path:
    """Create parent directory for a file path and return normalized Path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory path and return normalized Path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _render_folder_name(
    template: Optional[str],
    default_name: str,
    **context: Any,
) -> str:
    """Render folder name template with graceful fallback to raw template text."""
    folder_name = template or default_name
    try:
        return folder_name.format(**context)
    except Exception:
        return folder_name


def write_json_file(
    path: Union[str, Path],
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: Any = str,
) -> Path:
    """Write JSON payload to file with automatic parent directory creation."""
    output_path = ensure_parent_dir(path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=ensure_ascii, default=default)
    return output_path


def write_text_file(path: Union[str, Path], text: str, encoding: str = "utf-8") -> Path:
    """Write plain text to file with automatic parent directory creation."""
    output_path = ensure_parent_dir(path)
    with open(output_path, "w", encoding=encoding) as f:
        f.write(text)
    return output_path


def derive_sqsgenerator_config_path(log_file: Union[str, Path]) -> Path:
    """Derive sibling sqsgenerator config path from a log json file path."""
    log_path = Path(log_file)
    return log_path.with_name(f"{log_path.stem}_sqsgenerator.json")


def build_topk_output_root(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    output_folder_name: Optional[str],
    top_k: int,
) -> Path:
    """Build normalized output root used by top-k model execution API."""
    stem = Path(input_file).stem
    default_folder = f"{stem}__model_top{top_k}"
    folder_name = _render_folder_name(
        output_folder_name,
        default_folder,
        stem=stem,
        top_k=top_k,
    )

    return Path(output_dir) / folder_name


def build_topk_candidate_prefix(
    output_root: Union[str, Path],
    rank: int,
    supercell: Tuple[int, int, int],
) -> Path:
    """Build canonical file prefix for a top-k candidate artifact group."""
    l, w, h = supercell
    return Path(output_root) / f"model_rank_{rank}_{l}x{w}x{h}"


def build_topk_best_artifact_paths(
    output_root: Union[str, Path],
    base_name: str = "best_by_actual_objective",
) -> Dict[str, Path]:
    """Build canonical artifact paths for top-k best-by-objective export."""
    root = Path(output_root)
    cif = root / f"{base_name}.cif"
    log_json = root / f"{base_name}.json"
    sqsgenerator_config_json = derive_sqsgenerator_config_path(log_json)
    return {
        "cif": cif,
        "log_json": log_json,
        "sqsgenerator_config_json": sqsgenerator_config_json,
    }


def format_supercell_tag(supercell: Optional[Tuple[int, int, int]]) -> str:
    """Convert supercell tuple to a compact folder/file tag."""
    if not supercell:
        return "auto"
    return "x".join(str(x) for x in supercell)


def build_output_prefix(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    output_folder_name: Optional[str],
    supercell: Optional[Tuple[int, int, int]],
) -> Path:
    """Build normalized output prefix path used by report-oriented API."""
    stem = Path(input_file).stem
    supercell_tag = format_supercell_tag(supercell)
    default_folder = f"{stem}__{supercell_tag}"
    folder_name = _render_folder_name(
        output_folder_name,
        default_folder,
        stem=stem,
        supercell=supercell_tag,
    )

    folder_path = Path(output_dir) / folder_name
    return folder_path / folder_name
