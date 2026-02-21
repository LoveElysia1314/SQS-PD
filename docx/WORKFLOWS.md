# 工作流指南

本项目推荐三条主流程：

1. [单次 SQS 生成](#a-单次-sqs-生成)
2. [dry-run 合法超胞候选集分析](#b-dry-run-候选分析不执行优化)
3. [批量候选 + ML 排序](#c-批量候选--ml-排序)

---

## A. 单次 SQS 生成

### 流程

1. 读取结构（CIF / Structure）
2. 无序分析（SD / PD / SPD）
3. 生成 sqsgenerator 配置（split 模式）
4. 运行优化
5. 后处理并返回 `SQSResult`

### 最小调用

```python
from sqs_pd import generate_sqs, SQSOptions

result = generate_sqs(
    "data/input/demo_sd.cif",
    options=SQSOptions(iterations=5000),
)
if result.success:
    print("objective:", result.objective)
    print("supercell:", result.supercell_used)
```

### 带完整产物输出

```python
from sqs_pd import generate_sqs_with_report

result = generate_sqs_with_report(
    "data/input/demo_sd.cif",
    output_dir="data/output",
    output_folder_name="batch_{stem}__{supercell}",
    output_profile="full",  # silent / cif / logs / full
)
```

`output_profile` 别名说明：`none`（=silent）、`cif-only`（=cif）、`log`（=logs）、`all`（=full）

### 代码对应

- `src/sqs_pd/interface/api.py` — `generate_sqs`、`generate_sqs_with_report`
- `src/sqs_pd/runtime/sqs_orchestrator.py` — 主流程编排

---

## B. dry-run 候选分析（不执行优化）

### 目标

快速得到：
- 无序类型
- 合法超胞候选集数量
- 推荐超胞（当前策略）

### 单文件调用

```python
from sqs_pd import dry_run_recommend_supercell

info = dry_run_recommend_supercell("data/input/demo_pd.cif", max_error=0.0005)
print(info["recommended_supercell"], info["rss"], info["num_candidates"])
```

### 批量调用

```python
from sqs_pd import batch_recommend_supercells

rows = batch_recommend_supercells([
    "data/input/demo_sd.cif",
    "data/input/demo_pd.cif",
])
print(len(rows), rows[0]["optimization_success"])
```

### 代码对应

- `src/sqs_pd/runtime/dry_run.py`
- `src/sqs_pd/interface/api.py` — `dry_run_recommend_supercell`、`batch_recommend_supercells`
- `src/sqs_pd/batching/batch_common.py`
- `src/sqs_pd/core/supercell_optimizer.py`

---

## C. 批量候选 + ML 排序

### 目标

- 面向 CIF 数据集运行候选超胞
- 生成 `sqs_all_results.csv` 与 `ml_dataset.csv`
- 训练/推理排序模型并回馈候选选择（仅在最小 RSS 候选子集（硬约束）内排序）

### 约束规则

- 对任意 CIF：最小 RSS 候选子集是硬约束。
- 模型排序（含 `model-topk`）只在该子集内执行，不允许跨子集推荐。

### 批处理入口

```python
from sqs_pd.batching.batch_dataset_runner import build_batch_runner_parser, run_batch_sqs_all_candidates

parser = build_batch_runner_parser(default_data_dir="data/input")
args = parser.parse_args([])
run_batch_sqs_all_candidates(args)
```

### 单文件 top-k 实跑（推荐）

```python
from sqs_pd import run_topk_sqs_with_model

result = run_topk_sqs_with_model(
    "data/input/demo_sd.cif",
    top_k=5,
    artifact_policy="best",  # none / best / all
    output_dir="data/output",
    output_folder_name="topk_{stem}_k{top_k}",
)
print(result["success"], result["best"])
```

产物策略说明：
- `none`：仅返回内存结果，不落地候选文件
- `best`：只导出真实最优一组（`best_by_actual_objective.*`）
- `all`：导出 top-k 全部候选（`model_rank_{rank}_{l}x{w}x{h}.*`）

### 代码对应

- `src/sqs_pd/batching/batch_dataset_runner.py` — 薄入口
- `src/sqs_pd/batching/batch_service.py` — 批处理编排（四阶段）
- `src/sqs_pd/batching/batch_analysis.py` — 批量分析与特征聚合
- `src/sqs_pd/batching/batch_selector.py` — 候选选择（min-rss / model-topk）
- `src/sqs_pd/batching/batch_storage.py` — SQS / ML CSV 读写
- `src/sqs_pd/ranking/ml_ranker_trainer.py` — 模型训练与评估
- `src/sqs_pd/ranking/model_ranker_inference.py` — 推理排序
- `src/sqs_pd/batching/pipeline_layout.py` — 统一路径策略

---

## 常见误用

- **误把 dry-run 当作优化结果**：dry-run 不返回优化结构，只返回候选推荐。
- **手动改动 split 配置中的占位符语义**：容易破坏子晶格映射，导致 ParseError。
- **批处理续算时改变去重键定义**：会导致重复计算或错误跳过。

---

## 最小验证

```bash
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
python -m sqs_pd.interface.cli analyze data/input/demo_sd.cif
```
