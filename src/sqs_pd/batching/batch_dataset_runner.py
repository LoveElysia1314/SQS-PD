"""Batch SQS entrypoint.

This module now focuses on CLI argument parsing and orchestration wiring.
Business logic is split across dedicated modules:
- batch_types.py
- batch_selector.py
- batch_analysis.py
- batch_storage.py
- batch_service.py
"""

from __future__ import annotations

import argparse

from .batch_common import DEFAULT_RANKER_CONFIG_PATH, DEFAULT_RANKER_MODEL_PATH
from .batch_service import SQSBatchRunner
from .batch_types import CIFAnalysisResult, ProgressTracker, SQSResult, SQSTaskSpec
from .batch_selector import SupercellSelector
from .batch_analysis import DisorderAnalyzer
from .batch_storage import CSVHandler
from .pipeline_layout import (
    DEFAULT_ML_DATASET_REL,
    DEFAULT_SQS_RESULTS_REL,
    DEFAULT_SUMMARY_JSON_REL,
)


def build_batch_runner_parser(
    default_data_dir: str = "data/input",
    default_out_csv: str = DEFAULT_SQS_RESULTS_REL,
    default_out_ml: str = DEFAULT_ML_DATASET_REL,
    default_out_json: str = DEFAULT_SUMMARY_JSON_REL,
    default_iterations: int = 10000,
):
    """创建批量运行参数解析器（通用库入口）"""
    parser = argparse.ArgumentParser(
        description="Batch SQS runner (split-module architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default=default_data_dir, help="CIF folder"
    )
    parser.add_argument("--out-csv", type=str, default=default_out_csv, help="SQS CSV")
    parser.add_argument(
        "--out-ml",
        type=str,
        default=default_out_ml,
        help="ML dataset CSV",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=default_out_json,
        help="Summary JSON",
    )
    parser.add_argument(
        "--iterations", type=int, default=default_iterations, help="SQS iterations"
    )
    parser.add_argument(
        "--max-cifs", type=int, default=None, help="Max CIFs to process"
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["min-rss", "model-topk"],
        default="min-rss",
        help="Candidate selection mode: min-rss (legacy) or model-topk",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="When selection-mode=model-topk, run SQS on top-k candidates",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=("Custom ranker model path " f"(default: {DEFAULT_RANKER_MODEL_PATH})"),
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help=(
            "Custom inference config path " f"(default: {DEFAULT_RANKER_CONFIG_PATH})"
        ),
    )
    parser.add_argument(
        "--keep-policy",
        type=str,
        choices=["all", "best-only"],
        default="all",
        help="For each CIF after executing selected specs, keep all rows or only best objective",
    )

    return parser


def run_batch_sqs_all_candidates(args):
    """执行批量 SQS + 训练集生成流程（通用库入口）"""
    runner = SQSBatchRunner(args)
    runner.run()


def main(argv=None):
    """模块脚本入口（通用默认值，不依赖硬编码绝对路径）"""
    parser = build_batch_runner_parser()
    args = parser.parse_args(argv)
    run_batch_sqs_all_candidates(args)


__all__ = [
    "CIFAnalysisResult",
    "SQSTaskSpec",
    "SQSResult",
    "ProgressTracker",
    "SupercellSelector",
    "CSVHandler",
    "DisorderAnalyzer",
    "SQSBatchRunner",
    "build_batch_runner_parser",
    "run_batch_sqs_all_candidates",
    "main",
]


if __name__ == "__main__":
    main()
