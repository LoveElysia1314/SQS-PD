#!/usr/bin/env python
"""
命令行工具：快速分析 CIF 并推荐超胞规格

用法:
    python -m sqs_pd.interface.cli analyze demo_sd.cif
    python -m sqs_pd.interface.cli analyze demo_sd.cif demo_pd.cif --verbose
    python -m sqs_pd.interface.cli analyze data/input/*.cif
"""

import argparse
import sys
from pathlib import Path

from ..runtime.dry_run import (
    analyze_cif_and_recommend_supercell,
    batch_analyze_cifs,
    print_analysis_summary,
    format_batch_analysis_summary,
)
from .. import __version__


def analyze_command(args):
    """分析命令"""
    cif_files = args.cif_files

    if not cif_files:
        print("❌ 错误：请提供至少一个 CIF 文件")
        return 1

    # 验证文件存在
    valid_files = []
    for f in cif_files:
        path = Path(f)
        if not path.exists():
            print(f"⚠️  警告：文件不存在: {f}")
        else:
            valid_files.append(path)

    if not valid_files:
        print("❌ 错误：没有有效的 CIF 文件")
        return 1

    # 单文件或批量
    if len(valid_files) == 1:
        result = analyze_cif_and_recommend_supercell(
            valid_files[0], max_error=args.max_error, verbose=args.verbose
        )

        if not args.verbose:
            print_analysis_summary(result)

        return 0 if result["optimization_success"] else 1
    else:
        results = batch_analyze_cifs(
            valid_files, max_error=args.max_error, verbose=args.verbose
        )

        # 打印汇总
        if not args.verbose:
            print(format_batch_analysis_summary(results))

        # 如果有任何失败，返回错误码
        any_failed = any(not r.get("optimization_success", False) for r in results)
        return 1 if any_failed else 0


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        prog="sqs-pd",
        description="SQS-PD: 特殊准随机结构生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"sqs-pd {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # analyze 子命令
    analyze_parser = subparsers.add_parser(
        "analyze", help="分析 CIF 文件并推荐超胞规格"
    )
    analyze_parser.add_argument("cif_files", nargs="+", help="CIF 文件路径（支持多个）")
    analyze_parser.add_argument(
        "--max-error", type=float, default=0.0005, help="最大允许误差（默认 0.0005）"
    )
    analyze_parser.add_argument(
        "-v", "--verbose", action="store_true", help="显示详细信息"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "analyze":
        return analyze_command(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
