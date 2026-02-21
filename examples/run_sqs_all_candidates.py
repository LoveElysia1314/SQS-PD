"""批量 SQS + 训练集生成入口。

说明：
- 默认输出路径为 artifacts/pipeline/*
- 具体业务逻辑位于 `sqs_pd.batching.batch_dataset_runner`
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqs_pd.batching.batch_dataset_runner import (
    build_batch_runner_parser,
    run_batch_sqs_all_candidates,
)
from sqs_pd.batching.pipeline_layout import (
    DEFAULT_SQS_RESULTS_REL,
    DEFAULT_ML_DATASET_REL,
    DEFAULT_SUMMARY_JSON_REL,
)


def main(argv=None):
    default_cif_dir = r"D:\drzqr\Documents\GitHub\SODNet\datasets\SuperCon\cif"

    print(
        "[路径策略] 当前脚本默认输出已切换到 artifacts/pipeline/*。"
        "\n[路径策略] 建议全链路统一使用 artifacts 目录。"
    )

    parser = build_batch_runner_parser(
        default_data_dir=default_cif_dir,
        default_out_csv=DEFAULT_SQS_RESULTS_REL,
        default_out_ml=DEFAULT_ML_DATASET_REL,
        default_out_json=DEFAULT_SUMMARY_JSON_REL,
        default_iterations=10000,
    )
    args = parser.parse_args(argv)
    run_batch_sqs_all_candidates(args)


if __name__ == "__main__":
    main()
