"""测试05: 批量入口与 pipeline 路径一致性。

学习目标:
1. 确保 CLI 默认参数与常量一致
2. 确保输出路径统一归档到 artifacts/pipeline
"""

from pathlib import Path

from conftest import print_section, print_concept
from sqs_pd.batching.batch_dataset_runner import build_batch_runner_parser
from sqs_pd.batching.pipeline_layout import (
    DEFAULT_ML_DATASET_REL,
    DEFAULT_SQS_RESULTS_REL,
    DEFAULT_SUMMARY_JSON_REL,
    resolve_pipeline_output_paths,
)


def test_batch_entry_defaults_consistent():
    print_section("批量入口默认值")
    print_concept("parser 默认参数应与 pipeline_layout 常量一致")

    default_cif_dir = r"D:\drzqr\Documents\GitHub\SODNet\datasets\SuperCon\cif"
    parser = build_batch_runner_parser(
        default_data_dir=default_cif_dir,
        default_out_csv=DEFAULT_SQS_RESULTS_REL,
        default_out_ml=DEFAULT_ML_DATASET_REL,
        default_out_json=DEFAULT_SUMMARY_JSON_REL,
        default_iterations=10000,
    )
    args = parser.parse_args([])

    assert args.out_csv == DEFAULT_SQS_RESULTS_REL
    assert args.out_ml == DEFAULT_ML_DATASET_REL
    assert args.out_json == DEFAULT_SUMMARY_JSON_REL
    assert args.data_dir == default_cif_dir


def test_pipeline_paths_resolve_to_artifacts_pipeline():
    print_section("pipeline 输出路径")
    print_concept("解析后的绝对路径应集中在 artifacts/pipeline")

    outputs = resolve_pipeline_output_paths(
        DEFAULT_SQS_RESULTS_REL,
        DEFAULT_ML_DATASET_REL,
        DEFAULT_SUMMARY_JSON_REL,
    )
    root = Path(__file__).resolve().parents[1] / "artifacts" / "pipeline"

    assert outputs.sqs_csv.parent == root
    assert outputs.ml_csv.parent == root
    assert outputs.summary_json.parent == root
    assert outputs.disorder_features_csv.parent == root
