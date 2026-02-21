"""监督学习排序模型训练入口。

说明：
- 默认读取 artifacts/pipeline/ml_dataset.csv，输出到 artifacts/models/ml_ranker
- 具体训练逻辑位于 `sqs_pd.ranking.ml_ranker_trainer`
"""

from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqs_pd.ranking.ml_ranker_trainer import train_ranker


def _resolve_dataset_path() -> Path | None:
    """仅使用 artifacts 路径。"""
    path = ROOT / "artifacts" / "pipeline" / "ml_dataset.csv"
    return path if path.exists() else None


def main():
    parser = argparse.ArgumentParser(
        description="Train ranker and optionally promote default model"
    )
    parser.add_argument(
        "--set-default",
        action="store_true",
        help="Promote trained model to artifacts/models/default_ml_ranker",
    )
    args = parser.parse_args()

    print(
        "[路径策略] 当前脚本默认输出到 artifacts/models/ml_ranker。"
        "\n[路径策略] 数据集仅从 artifacts/pipeline/ml_dataset.csv 读取。"
    )

    dataset_path = _resolve_dataset_path()
    if dataset_path is None:
        print("ML dataset not found. Run examples/run_sqs_all_candidates.py first.")
        return

    train_ranker(
        csv_path=dataset_path,
        out_dir=ROOT / "artifacts" / "models" / "ml_ranker",
        set_as_default=args.set_default,
        default_model_dir=ROOT / "artifacts" / "models" / "default_ml_ranker",
    )


if __name__ == "__main__":
    main()
