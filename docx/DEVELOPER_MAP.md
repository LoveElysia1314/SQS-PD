# 开发者模块地图

## 术语统一约定（全仓）

| 术语 | 定义 |
|------|------|
| 合法超胞候选集 | dry-run 在误差约束下得到的全部可行候选 |
| 最小 RSS 候选子集（硬约束） | 从合法超胞候选集中按最小 RSS 过滤后的候选子集 |
| 模型排序 | 仅在"最小 RSS 候选子集（硬约束）"内执行 |
| top-k 实跑 | 对排序后的前 k 个候选逐个执行 SQS |
| `output_profile` | 输出配置档：`silent / cif / logs / full`（含别名） |
| `artifact_policy` | 产物保留策略：`none / best / all` |

---

## 文档治理规则

- **单一事实来源**：同一主题只在一处主讲
- **代码优先**：文档描述必须能在 `src/sqs_pd` 对应模块中找到实现
- **去历史化**：不记录重构过程，只保留当前稳定结论
- **经验产品化**：保留试错经验，以"规则/反例/检查清单"形式呈现

---

## 公开入口层

- `src/sqs_pd/__init__.py` — 统一导出
- `src/sqs_pd/interface/api.py` — 对外 API
- `src/sqs_pd/interface/cli.py` — 命令行入口

### `interface/api.py` 中 top-k 私有辅助函数边界

- `_run_topk_candidate_sqs`：逐候选执行 SQS，收集 rows / artifacts
- `_assign_actual_ranks`：基于量化 objective 计算并列排名
- `_write_topk_comparison_csv`：导出 model-vs-actual 对比 CSV
- `_export_topk_best_artifacts`：`best` 策略下的最终落盘导出
- `_build_topk_failed_result` / `_build_topk_success_result`：返回结构组装

> 后续新增 top-k 行为时，优先扩展对应私有函数，不直接膨胀 `run_topk_sqs_with_model` 主编排函数。

---

## 编排与执行层

- `src/sqs_pd/runtime/sqs_orchestrator.py` — 主流程协调（prepare/build/optimize/post-process）
- `src/sqs_pd/core/options.py` — 配置对象定义
- `src/sqs_pd/runtime/io_utils.py` — 输出与文件写入工具

---

## 结构语义层

- `src/sqs_pd/analysis/cif_disorder_analyzer.py` — CIF 表级解析与位点分类
- `src/sqs_pd/analysis/disorder_analyzer.py` — Structure 级分析
- `src/sqs_pd/analysis/analysis_utils.py` — 分类工具函数

---

## 超胞与配置层

- `src/sqs_pd/core/supercell_optimizer.py` — 候选合法性与优化选择
- `src/sqs_pd/core/supercell_calculator.py` — LCM 与形状基础工具
- `src/sqs_pd/core/config_builder.py` — split 模式配置生成

---

## 批量与 ML 层

- `src/sqs_pd/batching/batch_dataset_runner.py` — 薄入口（CLI 参数解析 + facade 导出）
- `src/sqs_pd/batching/batch_service.py` — 批处理编排（四阶段调度）
- `src/sqs_pd/batching/batch_analysis.py` — 批量分析与特征聚合
- `src/sqs_pd/batching/batch_selector.py` — 候选选择（min-rss / model-topk）
- `src/sqs_pd/batching/batch_storage.py` — SQS / ML CSV 读写
- `src/sqs_pd/batching/batch_types.py` — 批处理 dataclass
- `src/sqs_pd/batching/batch_common.py` — 批处理常量、小工具与公共循环
- `src/sqs_pd/ranking/ml_ranker_trainer.py` — 模型训练与评估
- `src/sqs_pd/ranking/model_ranker_inference.py` — 推理排序
- `src/sqs_pd/batching/pipeline_layout.py` — 统一路径策略

---

## 核心概念代码索引

| 概念 | 主要实现 |
|------|----------|
| 无序类型识别 | `analysis/disorder_analyzer.py`、`analysis/cif_disorder_analyzer.py` |
| 合法超胞候选枚举 | `core/supercell_optimizer.py`、`runtime/dry_run.py` |
| split 模式配置构建 | `core/config_builder.py` |
| sqsgenerator 调用与结果解析 | `runtime/sqs_orchestrator.py` |
| auto 选胞策略 | `interface/api.py` — `_resolve_auto_supercell_for_cif` |
| lcmm 特征计算 | `foundation/fraction_utils.py`、`ranking/ranker_features.py` |

---

## 测试映射

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_01_supercell_basics.py` | 超胞基础与 LCM 投影 |
| `tests/test_02_sqsgenerator_api.py` | sqsgenerator 集成与关键语义 |
| `tests/test_03_critical_semantics.py` | API 工作流与自动超胞策略 |
| `tests/test_04_project_api.py` | CIF 分析与 dry-run 优化 |
| `tests/test_05_cif_disorder_analysis.py` | 批量入口与 pipeline 路径一致性 |

## 最小回归命令

```bash
pytest tests/test_01_supercell_basics.py -v
pytest tests/test_02_sqsgenerator_api.py -v
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
pytest tests/test_05_cif_disorder_analysis.py -v
```
