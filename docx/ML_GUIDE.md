# ML 指南（排序模型）

## 1. 目标

对每个 CIF 的合法超胞候选做组内排序，优先推荐更可能得到较优 objective 的规格。

## 2. 数据来源

默认训练数据：`artifacts/pipeline/ml_dataset.csv`

关键字段（训练侧）：
- `objective`
- `cif_file`
- 几何与统计特征（volume/sphericity/...）

`lcmm` 口径（已与当前代码同步）：
- 不再使用旧的“浮点直接 `limit_denominator(max_den)`”路径；
- 统一基于“最小 RSS 且最小 size”的超胞规格，将占据数投影到 `round(occ*size)/size`；
- 对投影分数逐项约分后，取分母的 LCM 作为 `lcmm`。

对应代码：
- `src/sqs_pd/foundation/fraction_utils.py`
- `src/sqs_pd/ranking/ranker_features.py`
- `src/sqs_pd/batching/batch_storage.py`
- `src/sqs_pd/ranking/model_ranker_inference.py`

代码对应：
- `src/sqs_pd/ranking/ml_ranker_trainer.py`
- `src/sqs_pd/batching/pipeline_layout.py`

## 3. 训练

```python
from sqs_pd.ranking.ml_ranker_trainer import train_ranker

train_ranker(
    csv_path="artifacts/pipeline/ml_dataset.csv",
    out_dir="artifacts/models/ml_ranker",
)
```

可选：固化为默认生产模型（训练模块内提供能力）。

## 4. 推理

```python
from sqs_pd import recommend_supercells_with_model

result = recommend_supercells_with_model("data/input/demo_sd.cif", top_k=5)
print(result["recommended_supercell"])
```

推理会自动：
1. dry-run 获取合法超胞候选集
2. 过滤为“最小 RSS 候选子集（硬约束）”
3. 加载模型与 inference config
4. 仅对该子集输出排序结果

代码对应：
- `src/sqs_pd/ranking/model_ranker_inference.py`

## 5. 与批处理联动

批处理中可切换候选选择模式（最小 RSS / 模型 top-k）。

约束说明：
- `model-topk` 并不是在全部合法超胞候选集上直接取 top-k；
- 它先应用“最小 RSS 候选子集（硬约束）”，再在该子集内做模型排序与截断 top-k。

代码对应：
- `src/sqs_pd/batching/batch_dataset_runner.py`（入口）
- `src/sqs_pd/batching/batch_service.py`
- `src/sqs_pd/batching/batch_selector.py`
- `src/sqs_pd/interface/api.py::batch_recommend_supercells`

## 6. 最小验证

```bash
python examples/train_sqs_ranker.py
pytest tests/test_04_project_api.py -v
```

## 深度阅读

- [附录 B：ML 排序指标解读](APPENDIX_ML_METRICS_INTERPRETATION.md)
