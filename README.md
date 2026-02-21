# SQS-PD

SQS-PD 是一个面向**部分无序晶体结构**的 SQS（Special Quasirandom Structure）生成工具，提供从 CIF 文件到优化 SQS 结构的一体化工作流。

适用场景：
- 从实验数据库（如 ICSD、COD）获取的含无序位点 CIF 文件
- 需要为 DFT/MD 计算准备合理超胞的材料科学研究
- 超胞候选选择策略研究与批量数据流水线

---

## 功能特性

- **自动无序识别**：自动解析替代位无序（SD）、部分占位无序（PD）和混合无序（SPD），支持多标签组合
- **超胞候选推荐**：基于占据数误差约束枚举全部合法超胞候选，按 RSS 最小或 ML 模型排序择优
- **SQS 生成**：调用 sqsgenerator，采用 split 子晶格模式生成优化的特殊准随机结构
- **批量流水线**：对 CIF 数据集批量执行无序分析 + SQS 计算 + ML 数据集自动构建
- **ML 排序模型**：对超胞候选做组内监督排序，以最小化计算成本同时保持最优结构命中率

---

## 工作原理

### 无序类型识别

位点层分类规则：

| 类型 | 条件 |
|------|------|
| `ordered` | 单元素，总占据 ≈ 1 |
| `SD` | 多元素，总占据 ≈ 1（替代位无序） |
| `PD` | 单元素，总占据 < 1（含空位） |
| `SPD` | 多元素，总占据 < 1（混合无序） |

一个结构可包含多种无序类型的组合（如 `SD+PD`）。

### 超胞候选策略

当前实现分为两层：

1. 由占据数近似误差约束（`max_error`，默认 0.0005）筛选全部**合法超胞候选集**
2. 在合法候选集内按目标择优：RSS 最小，或 ML 模型排序

**`auto` 自动选胞策略（默认）**：
1. 优先使用 ML 模型 top-1 推荐
2. 模型不可用时，回退到合法候选集中 RSS 最小且 size 最小的规格

### SQS 生成流程

```
读取 CIF → 无序分析 → 超胞选择 → 构建 sqsgenerator 配置 → 运行优化 → 后处理 → SQSResult
```

SQS 优化采用 `split` 子晶格模式：
- 用占位符标签标记可优化位点（不直接使用元素符号）
- `composition` 仅描述无序位点的元素分配
- 空位在配置中统一用 `"0"` 表示

### ML 排序模型

基于 LightGBM 的组内排序模型，输入特征来自超胞几何与无序统计（`volume`、`sphericity`、`total_site_entropy`、`lcmm`、`num_disordered_sites`、`valid_supercell_count`），目标是对每个 CIF 的合法超胞候选做组内排序，降低后续 SQS 计算总开销。

当前模型（训练于 551 个 CIF、17254 个候选）**独立测试集表现**：Hit@3 = 0.93，NDCG@3 = 0.82，MRR = 0.81。详见 [ML 指南](docx/ML_GUIDE.md)。

---

## 安装

```bash
pip install pymatgen sqsgenerator
pip install -e .
```

---

## 快速开始

### 1. 单文件 SQS 生成（最小调用）

```python
from sqs_pd import generate_sqs

result = generate_sqs("data/input/demo_sd.cif")
if result.success:
    print("objective:", result.objective)
    print("supercell:", result.supercell_used)
else:
    print("error:", result.error)
```

### 2. 只做分析，不跑优化（dry-run）

```python
from sqs_pd import dry_run_recommend_supercell

info = dry_run_recommend_supercell("data/input/demo_pd.cif")
print(info["disorder_types"])
print(info["recommended_supercell"])
print(info["num_candidates"])
```

### 3. 带报告产物输出

```python
from sqs_pd import generate_sqs_with_report

result = generate_sqs_with_report(
    "data/input/demo_sd.cif",
    output_dir="data/output",
    output_profile="full",  # silent / cif / logs / full
)
print(result.success)
```

### 4. 模型排序推荐（top-k）

```python
from sqs_pd import recommend_supercells_with_model

ranked = recommend_supercells_with_model("data/input/demo_sd.cif", top_k=5)
print(ranked["constraint"])          # min-rss-only
print(ranked["num_candidates"], ranked["num_candidates_all"])
print(ranked["recommended_supercell"])
for row in ranked["ranked_candidates"]:
    print(row["rank"], row["supercell"], row["model_score"])
```

> `ranked_candidates` 只来自"最小 RSS 候选子集（硬约束）"；模型不跨候选子集推荐。

### 5. top-k 推荐并实跑

```python
from sqs_pd import run_topk_sqs_with_model

result = run_topk_sqs_with_model(
    "data/input/demo_sd.cif",
    top_k=3,
    artifact_policy="best",  # none / best / all
    output_dir="data/output",
)
print(result["success"], result["best"])
```

### 6. 批量 dry-run 推荐

```python
from sqs_pd import batch_recommend_supercells

rows = batch_recommend_supercells([
    "data/input/demo_sd.cif",
    "data/input/demo_pd.cif",
])
print(len(rows), rows[0]["optimization_success"])
```

---

## 主要术语

| 术语 | 含义 |
|------|------|
| 合法超胞候选集 | dry-run 在误差约束下得到的全部可行候选 |
| 最小 RSS 候选子集（硬约束） | 从合法超胞候选集中按最小 RSS 过滤后的候选子集 |
| 模型排序 | 仅在"最小 RSS 候选子集（硬约束）"内执行 |
| top-k 实跑 | 对排序后的前 k 个候选逐个执行 SQS |
| `output_profile` | 输出配置档：`silent / cif / logs / full`（含别名） |
| `artifact_policy` | 产物保留策略：`none / best / all` |

---

## 文档

详细文档位于 [`docx/`](docx/)：

| 文档 | 内容 |
|------|------|
| [工作流指南](docx/WORKFLOWS.md) | 三条主流程的完整示例与代码对应 |
| [核心概念](docx/CORE_CONCEPTS.md) | 无序类型、超胞策略、split 模式、执行链 |
| [API 参考](docx/API_REFERENCE.md) | 全部公开函数、参数与返回值说明 |
| [故障排查与实践规则](docx/TROUBLESHOOTING.md) | 实践规则 + 常见错误排查手册 |
| [ML 指南](docx/ML_GUIDE.md) | 训练、推理、成果分析与指标解读 |
| [开发者模块地图](docx/DEVELOPER_MAP.md) | 模块结构、术语约定、文档治理规则 |
| [附录：sqsgenerator 诊断手册](docx/APPENDIX_SQSGENERATOR_DIAGNOSTICS.md) | ParseError 诊断 + 深度调试脚本 |

---

## 运行测试

```bash
pytest tests/test_01_supercell_basics.py -v
pytest tests/test_02_sqsgenerator_api.py -v
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
pytest tests/test_05_cif_disorder_analysis.py -v
```

---

## 许可证

MIT
