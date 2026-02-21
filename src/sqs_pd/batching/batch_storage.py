"""CSV and dataset I/O for batch SQS workflows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .batch_types import SQSResult
from .batch_common import safe_int, str_to_bool
from ..ranking.ranker_features import compute_geometry_features
from ..runtime.io_utils import ensure_parent_dir


class CSVHandler:
    SQS_FIELDNAMES = [
        "cif_file",
        "size",
        "l",
        "w",
        "h",
        "success",
        "objective",
        "time_s",
        "message",
    ]

    ML_FIELDNAMES = [
        "cif_file",
        "size",
        "l",
        "w",
        "h",
        "objective",
        "success",
        "time_s",
        "message",
        "volume",
        "sphericity",
        "face_sphericity",
        "mean_dim",
        "lcmm",
        "min_den",
        "max_den",
        "mean_den",
        "median_den",
        "unique_den_count",
        "num_disordered_sites",
        "per_site_denominators_str",
        "valid_supercell_count",
        "total_site_entropy",
    ]

    def __init__(self, sqs_csv: Path, ml_csv: Path):
        self.sqs_csv = sqs_csv
        self.ml_csv = ml_csv

    def initialize_sqs_csv(self):
        if not self.sqs_csv.exists():
            sqs_csv_path = ensure_parent_dir(self.sqs_csv)
            with sqs_csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.SQS_FIELDNAMES)
                writer.writeheader()

    def read_computed_specs(self) -> Set[Tuple]:
        computed = set()
        if not self.sqs_csv.exists():
            return computed

        with self.sqs_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = (
                        row.get("cif_file", ""),
                        safe_int(row.get("size")),
                        safe_int(row.get("l")),
                        safe_int(row.get("w")),
                        safe_int(row.get("h")),
                    )
                    computed.add(key)
                except Exception:
                    continue

        return computed

    def read_computed_cifs(self) -> Set[str]:
        computed_cifs: Set[str] = set()
        if not self.sqs_csv.exists():
            return computed_cifs

        with self.sqs_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cif_name = (row.get("cif_file") or "").strip()
                if cif_name:
                    computed_cifs.add(cif_name)

        return computed_cifs

    def append_sqs_result(self, result: SQSResult):
        row = {
            "cif_file": result.cif_name,
            "size": result.size,
            "l": result.l,
            "w": result.w,
            "h": result.h,
            "success": result.success,
            "objective": result.objective,
            "time_s": round(result.time_s, 3),
            "message": result.message,
        }

        sqs_csv_path = ensure_parent_dir(self.sqs_csv)
        with sqs_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SQS_FIELDNAMES)
            writer.writerow(row)

    def read_all_sqs_results(self) -> List[Dict]:
        if not self.sqs_csv.exists():
            return []

        results = []
        with self.sqs_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)

        return results

    def write_ml_dataset(self, rows: List[Dict], disorder_analyzer):
        ml_rows = []

        for sqs_row in rows:
            cif_name = sqs_row.get("cif_file", "")
            size = safe_int(sqs_row.get("size"))
            l = safe_int(sqs_row.get("l"))
            w = safe_int(sqs_row.get("w"))
            h = safe_int(sqs_row.get("h"))

            cif_info = disorder_analyzer.get_cif_info(cif_name)
            geom = compute_geometry_features(l, w, h)
            best_size = (
                int(cif_info.candidates[0]["size"])
                if (cif_info and cif_info.candidates)
                else int(size)
            )
            denom_info = disorder_analyzer.compute_denom_stats(
                cif_info.occupancies if cif_info else [],
                supercell_size=best_size,
            )

            ml_row = {
                "cif_file": cif_name,
                "size": size,
                "l": l,
                "w": w,
                "h": h,
                "objective": sqs_row.get("objective", ""),
                "success": str_to_bool(sqs_row.get("success", "False")),
                "time_s": sqs_row.get("time_s", ""),
                "message": sqs_row.get("message", ""),
                **geom,
                "lcmm": denom_info["lcmm"],
                "min_den": denom_info["min_den"],
                "max_den": denom_info["max_den"],
                "mean_den": denom_info["mean_den"],
                "median_den": denom_info["median_den"],
                "unique_den_count": denom_info["unique_den_count"],
                "num_disordered_sites": (
                    cif_info.num_disordered_sites if cif_info else 0
                ),
                "per_site_denominators_str": (
                    "|".join(
                        ",".join(str(x) for x in site)
                        for site in cif_info.per_site_denominators
                    )
                    if cif_info
                    else ""
                ),
                "valid_supercell_count": (
                    cif_info.valid_supercell_count if cif_info else 0
                ),
                "total_site_entropy": cif_info.total_site_entropy if cif_info else 0.0,
            }
            ml_rows.append(ml_row)

        ml_csv_path = ensure_parent_dir(self.ml_csv)
        with ml_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.ML_FIELDNAMES)
            writer.writeheader()
            writer.writerows(ml_rows)
