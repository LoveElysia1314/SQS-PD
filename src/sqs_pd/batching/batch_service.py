"""Batch SQS orchestration service.

Contains task planning, execution, and output post-processing while delegating
analysis/selection/storage to dedicated modules.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import List, Tuple

from ..interface.api import generate_sqs
from .batch_analysis import DisorderAnalyzer
from .batch_common import (
    DEFAULT_RANKER_CONFIG_PATH,
    DEFAULT_RANKER_MODEL_PATH,
    safe_float,
    safe_int,
)
from .batch_selector import SupercellSelector
from .batch_storage import CSVHandler
from .batch_types import ProgressTracker, SQSResult, SQSTaskSpec
from ..core.options import SQSOptions
from .pipeline_layout import resolve_pipeline_output_paths, resolve_project_path
from ..runtime.io_utils import ensure_parent_dir


class TaskManager:
    def __init__(
        self,
        disorder_analyzer: DisorderAnalyzer,
        csv_handler: CSVHandler,
        data_dir: Path,
        selector: SupercellSelector,
        keep_policy: str,
    ):
        self.disorder_analyzer = disorder_analyzer
        self.csv_handler = csv_handler
        self.data_dir = data_dir
        self.selector = selector
        self.keep_policy = keep_policy
        self.progress = ProgressTracker(total_cifs=0)

    def build_task_list(self) -> List[Tuple[str, List[SQSTaskSpec]]]:
        computed_specs = self.csv_handler.read_computed_specs()
        computed_cifs = self.csv_handler.read_computed_cifs()
        task_list = []

        total_cifs = len(self.disorder_analyzer.cif_analyses)
        self.progress = ProgressTracker(total_cifs=total_cifs)

        for cif_name, analysis in sorted(self.disorder_analyzer.cif_analyses.items()):
            selected_candidates = self.selector.select_candidates(analysis)

            if self.keep_policy == "best-only" and cif_name in computed_cifs:
                self.progress.completed_cifs += 1
                self.progress.completed_specs += len(selected_candidates)
                continue

            remaining = []

            for cand in selected_candidates:
                spec = SQSTaskSpec(
                    cif_name=cif_name,
                    size=cand["size"],
                    supercell=cand["supercell"],
                    rss=cand["rss"],
                    max_error=cand.get("max_error"),
                )

                if spec.to_key() not in computed_specs:
                    remaining.append(spec)

            if remaining:
                task_list.append((cif_name, remaining))
                self.progress.pending_cifs += 1
                self.progress.total_specs += len(remaining)

                if len(remaining) < len(selected_candidates):
                    self.progress.partial_cifs.append(cif_name)
            elif len(selected_candidates) > 0:
                self.progress.completed_cifs += 1
                self.progress.completed_specs += len(selected_candidates)

        return task_list

    def execute_task(
        self, spec: SQSTaskSpec, iterations: int, verbose: bool = False
    ) -> SQSResult:
        cif_path = self.data_dir / spec.cif_name
        l, w, h = spec.supercell

        opts = SQSOptions(
            supercell=(l, w, h), iterations=iterations, console_log=verbose
        )
        start = time.time()

        try:
            result = generate_sqs(str(cif_path), options=opts)
            elapsed = time.time() - start

            return SQSResult(
                cif_name=spec.cif_name,
                size=spec.size,
                l=l,
                w=w,
                h=h,
                success=result.success,
                objective=result.objective,
                time_s=elapsed,
                message=result.error or "",
            )

        except Exception as e:
            elapsed = time.time() - start
            return SQSResult(
                cif_name=spec.cif_name,
                size=spec.size,
                l=l,
                w=w,
                h=h,
                success=False,
                objective=None,
                time_s=elapsed,
                message=str(e),
            )


class SQSBatchRunner:
    def __init__(self, args):
        self.args = args
        self.data_dir = resolve_project_path(args.data_dir)
        outputs = resolve_pipeline_output_paths(
            args.out_csv, args.out_ml, args.out_json
        )
        self.out_csv = outputs.sqs_csv
        self.out_ml = outputs.ml_csv
        self.out_json = outputs.summary_json
        self.out_features = outputs.disorder_features_csv
        self.iterations = args.iterations
        self.max_cifs = args.max_cifs
        self.selection_mode = args.selection_mode
        self.top_k = args.top_k
        self.model_path = args.model_path
        self.model_config = args.model_config
        self.keep_policy = args.keep_policy

        self.disorder_analyzer = DisorderAnalyzer()
        self.csv_handler = CSVHandler(self.out_csv, self.out_ml)
        self.selector = SupercellSelector(
            mode=self.selection_mode,
            top_k=self.top_k,
            model_path=self.model_path,
            model_config=self.model_config,
        )
        self.task_manager = TaskManager(
            self.disorder_analyzer,
            self.csv_handler,
            self.data_dir,
            self.selector,
            self.keep_policy,
        )

    def run(self):
        self._validate_environment()
        self._prepare_directories()
        self._print_config()
        self._stage1_analyze_cifs()
        task_list = self._stage2_build_tasks()
        self._stage3_execute_tasks(task_list)
        self._stage4_finalize_and_organize()

    def _validate_environment(self):
        if not self.data_dir.exists():
            print(f"\n❌ 错误：CIF 文件夹不存在: {self.data_dir}")
            print(f"\n可以使用 --data-dir 指定正确的路径")
            sys.exit(1)

    def _prepare_directories(self):
        for path in [self.out_csv, self.out_ml, self.out_json, self.out_features]:
            ensure_parent_dir(path)

    def _print_config(self):
        print(f"\n使用的参数：")
        print(f"  CIF 文件夹: {self.data_dir}")
        print(f"  SQS 结果: {self.out_csv}")
        print(f"  ML 数据集: {self.out_ml}")
        print(f"  无序晶体摘要: {self.out_json}")
        print(f"  迭代次数: {self.iterations}")
        print(f"  候选选择模式: {self.selection_mode}")
        if self.selection_mode == "model-topk":
            print(f"  Top-K: {self.top_k}")
            resolved_model = resolve_project_path(
                self.model_path or DEFAULT_RANKER_MODEL_PATH
            )
            resolved_cfg = resolve_project_path(
                self.model_config or DEFAULT_RANKER_CONFIG_PATH
            )
            print(f"  模型文件: {resolved_model}")
            print(f"  推理配置: {resolved_cfg}")
        print(f"  保留策略: {self.keep_policy}")
        if self.max_cifs:
            print(f"  最大 CIF 数: {self.max_cifs}")

    def _stage1_analyze_cifs(self):
        print("\n" + "=" * 70)
        print("阶段 1：分析所有 CIF，生成无序晶体列表与候选超胞规格")
        print("=" * 70)

        cif_files = sorted(self.data_dir.glob("*.cif"))
        if self.max_cifs:
            cif_files = cif_files[: self.max_cifs]

        total_files = len(cif_files)
        print(f"Found {total_files} CIF files")

        self.disorder_analyzer.analyze_batch(cif_files, verbose=False)

        self.disorder_analyzer.save_summary(self.out_json, total_files)
        self.disorder_analyzer.save_features_csv(self.out_features)

        print(f"\nWrote disorder summary JSON: {self.out_json}")
        print(f"  Total CIFs: {total_files}")
        print(f"  Disordered CIFs: {len(self.disorder_analyzer.cif_analyses)}")

    def _stage2_build_tasks(self) -> List[Tuple[str, List[SQSTaskSpec]]]:
        print("\n" + "=" * 70)
        print("阶段 2：读取现有 CSV，排除已计算项，构建任务列表")
        print("=" * 70)

        task_list = self.task_manager.build_task_list()
        progress = self.task_manager.progress

        print(f"Task list: {len(task_list)} CIFs with specs to compute")
        print(f"Progress: {progress.current_progress()}")

        if progress.partial_cifs:
            print(f"  Partial: {', '.join(progress.partial_cifs[:5])}")

        for cif_name, specs in task_list:
            print(f"  - {cif_name}: {len(specs)} specs")

        if self.out_csv.exists() and self.out_csv.stat().st_size > 0:
            try:
                sqs_rows = self.csv_handler.read_all_sqs_results()
                self.csv_handler.write_ml_dataset(sqs_rows, self.disorder_analyzer)
                print(f"→ Generated/updated ML dataset: {self.out_ml}")
            except Exception as e:
                print(f"⚠ Failed to generate ML dataset: {e}")

        return task_list

    def _stage3_execute_tasks(self, task_list: List[Tuple[str, List[SQSTaskSpec]]]):
        print("\n" + "=" * 70)
        print("阶段 3：执行 SQS 计算")
        print("=" * 70)

        self.csv_handler.initialize_sqs_csv()
        progress = self.task_manager.progress

        for cif_idx, (cif_name, specs) in enumerate(task_list, 1):
            total_progress = progress.completed_cifs + cif_idx
            print(
                f"\n[{total_progress}/{progress.total_cifs}] {cif_name} ({len(specs)} specs)"
            )

            completed_specs = 0
            cif_results: List[SQSResult] = []

            for spec in specs:
                result = self.task_manager.execute_task(spec, self.iterations)
                cif_results.append(result)
                completed_specs += 1
                progress.completed_specs += 1

            results_to_keep = self._apply_keep_policy(cif_results)
            for kept in results_to_keep:
                self.csv_handler.append_sqs_result(kept)

            if completed_specs > 0:
                print(
                    f"  → Completed {completed_specs} specs for {cif_name}, kept {len(results_to_keep)}"
                )

                try:
                    sqs_rows = self.csv_handler.read_all_sqs_results()
                    self.csv_handler.write_ml_dataset(sqs_rows, self.disorder_analyzer)
                    print(f"  → Updated ML dataset")
                except Exception as e:
                    print(f"  ⚠ ML dataset update failed: {e}")

    def _apply_keep_policy(self, results: List[SQSResult]) -> List[SQSResult]:
        if self.keep_policy == "all":
            return results

        if not results:
            return []

        success_results = [r for r in results if r.success and r.objective is not None]
        if success_results:
            best = min(success_results, key=lambda x: float(x.objective))
            return [best]

        return [results[0]]

    def _stage4_finalize_and_organize(self):
        print("\n" + "=" * 70)
        print("阶段 4：完成并整理所有输出文件")
        print("=" * 70)

        print("\n→ 正在整理 SQS results CSV...")
        self._sort_and_rewrite_sqs_csv()
        print(f"✓ SQS results 已按文件名和 size 升序排列: {self.out_csv}")

        print("\n→ 正在强制全量生成 ML dataset...")
        try:
            sqs_rows = self.csv_handler.read_all_sqs_results()
            self.csv_handler.write_ml_dataset(sqs_rows, self.disorder_analyzer)
            print(f"✓ ML dataset 已生成: {self.out_ml}")
        except Exception as e:
            print(f"⚠ ML dataset 生成失败: {e}")

        print("\n" + "=" * 70)
        print("✓ 所有任务完成")
        print("=" * 70)
        print(f"SQS results (已整理): {self.out_csv}")
        print(f"ML dataset (已生成): {self.out_ml}")
        print(f"Summary JSON: {self.out_json}")

    def _sort_and_rewrite_sqs_csv(self):
        if not self.out_csv.exists():
            return

        rows = self.csv_handler.read_all_sqs_results()

        groups = defaultdict(list)
        for row in rows:
            key = (row.get("cif_file", ""), row.get("size", ""))
            groups[key].append(row)

        deduplicated_rows = []
        for (_, _), group_rows in sorted(groups.items()):
            if len(group_rows) == 1:
                deduplicated_rows.append(group_rows[0])
            else:
                success_true = [
                    r
                    for r in group_rows
                    if str(r.get("success", "False")).lower() == "true"
                ]
                success_false = [
                    r
                    for r in group_rows
                    if str(r.get("success", "False")).lower() != "true"
                ]

                if success_true:
                    if len(success_true) == 1:
                        deduplicated_rows.append(success_true[0])
                    else:
                        merged_row = self._merge_rows(success_true)
                        deduplicated_rows.append(merged_row)
                elif success_false:
                    if len(success_false) == 1:
                        deduplicated_rows.append(success_false[0])
                    else:
                        merged_row = self._merge_rows(success_false)
                        deduplicated_rows.append(merged_row)

        sorted_rows = sorted(
            deduplicated_rows,
            key=lambda r: (
                r.get("cif_file", ""),
                safe_int(r.get("size", 0)),
            ),
        )

        fieldnames = self.csv_handler.SQS_FIELDNAMES
        out_csv_path = ensure_parent_dir(self.out_csv)
        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_rows)

    def _merge_rows(self, rows):
        if not rows:
            return {}

        merged = rows[0].copy()

        objectives = [safe_float(r.get("objective", 0)) for r in rows]
        times = [safe_float(r.get("time_s", 0)) for r in rows]

        if objectives:
            avg_objective = sum(objectives) / len(objectives)
            merged["objective"] = str(avg_objective)

        if times:
            avg_time = sum(times) / len(times)
            merged["time_s"] = str(avg_time)

        return merged
