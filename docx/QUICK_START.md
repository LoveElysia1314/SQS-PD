# 快速开始

## 1) 安装

```bash
pip install pymatgen sqsgenerator
pip install -e .
```

## 2) 单文件最小调用

```python
from sqs_pd import generate_sqs

result = generate_sqs("data/input/demo_sd.cif")
if result.success:
    print("objective:", result.objective)
    print("supercell:", result.supercell_used)
else:
    print("error:", result.error)
```

## 3) 只做分析，不跑优化（dry-run）

```python
from sqs_pd import dry_run_recommend_supercell

info = dry_run_recommend_supercell("data/input/demo_pd.cif")
print(info["disorder_types"])
print(info["recommended_supercell"])
print(info["num_candidates"])
```

## 4) 带报告产物输出

```python
from sqs_pd import generate_sqs_with_report

result = generate_sqs_with_report(
    "data/input/demo_sd.cif",
    output_dir="data/output",
    output_profile="full",  # 主值: silent/cif/logs/full；别名: none/cif-only/log/all
)
print(result.success)
```

## 5) 模型排序推荐（top-k）

```python
from sqs_pd import recommend_supercells_with_model

ranked = recommend_supercells_with_model("data/input/demo_sd.cif", top_k=5)
print(ranked["constraint"])          # min-rss-only
print(ranked["num_candidates"], ranked["num_candidates_all"])
print(ranked["recommended_supercell"])
for row in ranked["ranked_candidates"]:
    print(row["rank"], row["supercell"], row["model_score"])
```

说明：`ranked_candidates` 只来自“最小 RSS 候选子集（硬约束）”；模型不会跨出该子集推荐。

## 6) top-k 推荐并实跑（可控产物策略）

```python
from sqs_pd import run_topk_sqs_with_model

result = run_topk_sqs_with_model(
    "data/input/demo_sd.cif",
    top_k=3,
    artifact_policy="best",  # none/best/all
    output_dir="data/output",
    output_folder_name="run_{stem}_k{top_k}",
)
print(result["success"], result["best"])
```

## 7) 最小验证命令

```bash
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
python -m sqs_pd.interface.cli analyze data/input/demo_sd.cif
```

## 代码对应

- 公开 API：`src/sqs_pd/interface/api.py`
- 主流程编排：`src/sqs_pd/runtime/sqs_orchestrator.py`
- CLI：`src/sqs_pd/interface/cli.py`
