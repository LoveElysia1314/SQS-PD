# 开发者模块地图

## 公开入口层

- `src/sqs_pd/__init__.py`：统一导出
- `src/sqs_pd/interface/api.py`：对外 API
- `src/sqs_pd/interface/cli.py`：命令行入口

### `interface/api.py` 中 top-k 私有辅助函数边界

- `_run_topk_candidate_sqs`：仅负责逐候选执行 SQS，并收集 `rows/artifacts`
- `_assign_actual_ranks`：仅负责基于量化 objective 计算并列排名
- `_write_topk_comparison_csv`：仅负责导出 model-vs-actual 对比 CSV
- `_export_topk_best_artifacts`：仅负责 `best` 策略下的最终落盘导出
- `_build_topk_failed_result/_build_topk_success_result`：仅负责返回结构组装

建议：后续新增 top-k 行为时，优先扩展对应私有函数，不直接膨胀 `run_topk_sqs_with_model` 主编排函数。

## 编排与执行层

- `src/sqs_pd/runtime/sqs_orchestrator.py`：主流程协调（prepare/build/optimize/post-process）
- `src/sqs_pd/core/options.py`：配置对象定义
- `src/sqs_pd/runtime/io_utils.py`：输出与文件写入工具

## 结构语义层

- `src/sqs_pd/analysis/cif_disorder_analyzer.py`：CIF 表级解析与位点分类
- `src/sqs_pd/analysis/disorder_analyzer.py`：Structure 级分析
- `src/sqs_pd/analysis/analysis_utils.py`：分类工具函数

## 超胞与配置层

- `src/sqs_pd/core/supercell_optimizer.py`：候选合法性与优化选择
- `src/sqs_pd/core/supercell_calculator.py`：LCM 与形状基础工具（不承担默认 auto 选胞）
- `src/sqs_pd/core/config_builder.py`：split 模式配置生成

## 批量与 ML 层

- `src/sqs_pd/batching/batch_dataset_runner.py`：薄入口（CLI 参数解析 + facade 导出）
- `src/sqs_pd/batching/batch_service.py`：批处理编排（阶段调度、任务执行、结果整理）
- `src/sqs_pd/batching/batch_analysis.py`：批量分析与特征聚合
- `src/sqs_pd/batching/batch_selector.py`：候选选择（min-rss / model-topk）
- `src/sqs_pd/batching/batch_storage.py`：SQS/ML CSV 读写
- `src/sqs_pd/batching/batch_types.py`：批处理 dataclass
- `src/sqs_pd/batching/batch_common.py`：批处理常量、小工具与批量分析公共循环（API/dry-run/batch 共用）
- `src/sqs_pd/ranking/ml_ranker_trainer.py`：模型训练与评估
- `src/sqs_pd/ranking/model_ranker_inference.py`：推理排序
- `src/sqs_pd/batching/pipeline_layout.py`：统一路径策略

## 当前整合建议（保持单文件体量可控）

- 建议保留拆分后的 batch 五层结构，不再回并为单文件。
- 可继续整合的方向：
	- 若后续 `batch_types.py` 稳定，可并入 `batch_service.py` 顶部，减少文件跳转
- 暂不建议再合并 `batch_analysis.py` / `batch_selector.py` / `batch_storage.py`，它们边界清晰且后续扩展频繁。

## 测试映射（建议）

- 超胞基础与 LCMM 投影：`tests/test_01_supercell_basics.py`
- sqsgenerator 集成与关键语义：`tests/test_02_sqsgenerator_api.py`
- API 工作流与自动超胞策略：`tests/test_03_critical_semantics.py`
- CIF 分析与 dry-run 优化：`tests/test_04_project_api.py`
- 批量入口与 pipeline 路径一致性：`tests/test_05_cif_disorder_analysis.py`

## 最小回归命令

```bash
pytest tests/test_01_supercell_basics.py -v
pytest tests/test_02_sqsgenerator_api.py -v
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
pytest tests/test_05_cif_disorder_analysis.py -v
```
