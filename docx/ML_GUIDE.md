# ML 指南（排序模型）

## 1. 目标

对每个 CIF 的合法超胞候选做组内排序，优先推荐更可能得到较优 objective 的规格，降低后续 SQS 计算总成本。

---

## 2. 数据来源

默认训练数据：`artifacts/pipeline/ml_dataset.csv`

关键字段（训练侧）：
- `objective`：SQS 优化目标函数值（越小越好）
- `cif_file`：CIF 文件名（用于组内分组）
- 几何与统计特征（volume / sphericity / ...）

`lcmm` 口径（已与当前代码同步）：
- 不再使用旧的"浮点直接 `limit_denominator(max_den)`"路径
- 统一基于"最小 RSS 且最小 size"的超胞规格，将占据数投影到 `round(occ*size)/size`
- 对投影分数逐项约分后，取分母的 LCM 作为 `lcmm`

代码对应：
- `src/sqs_pd/foundation/fraction_utils.py`
- `src/sqs_pd/ranking/ranker_features.py`
- `src/sqs_pd/batching/batch_storage.py`
- `src/sqs_pd/ranking/ml_ranker_trainer.py`
- `src/sqs_pd/batching/pipeline_layout.py`

---

## 3. 训练

```python
from sqs_pd.ranking.ml_ranker_trainer import train_ranker

train_ranker(
    csv_path="artifacts/pipeline/ml_dataset.csv",
    out_dir="artifacts/models/ml_ranker",
)
```

或直接运行训练脚本：

```bash
python examples/train_sqs_ranker.py
# 可选：--set-default  固化为默认生产模型
```

训练产物位于 `artifacts/models/ml_ranker/`：
- `ranker_model.txt` — 模型文件
- `ranker_inference_config.json` — 推理配置
- `ranker_report.json` — CV 与测试集指标
- `feature_importance.csv` — 特征重要性
- `per_cif_test_metrics.csv` — 每个 CIF 的测试指标
- `ranker_diagnostics.png` — 可视化诊断图

---

## 4. 推理

```python
from sqs_pd import recommend_supercells_with_model

result = recommend_supercells_with_model("data/input/demo_sd.cif", top_k=5)
print(result["recommended_supercell"])
for row in result["ranked_candidates"]:
    print(row["rank"], row["supercell"], row["model_score"])
```

推理自动执行：
1. dry-run 获取合法超胞候选集
2. 过滤为"最小 RSS 候选子集（硬约束）"
3. 加载模型与 inference config
4. 仅对该子集输出排序结果

代码对应：`src/sqs_pd/ranking/model_ranker_inference.py`

---

## 5. 与批处理联动

批处理中可切换候选选择模式：

| 模式 | 含义 |
|------|------|
| `min-rss` | 只取最小 RSS 候选子集，不排序 |
| `model-topk` | 先过滤最小 RSS 候选子集，再做模型排序并截断 top-k |

> `model-topk` 不在全部合法超胞候选集上直接取 top-k；硬约束（最小 RSS 子集）始终先行。

代码对应：
- `src/sqs_pd/batching/batch_dataset_runner.py` — 入口
- `src/sqs_pd/batching/batch_service.py`
- `src/sqs_pd/batching/batch_selector.py`

---

## 6. 最小验证

```bash
python examples/train_sqs_ranker.py
pytest tests/test_04_project_api.py -v
```

---

## 7. 当前数据集训练成果分析（2026-02-26）

本节基于当前仓库真实产物与一次完整训练输出：

- 数据集来源：`artifacts/pipeline/ml_dataset.csv`
- 模型输出目录：`artifacts/models/ml_ranker`
- 产物：`ranker_report.json`、`feature_importance.csv`、`per_cif_test_metrics.csv`、`ranker_diagnostics.png`

### 7.1 数据规模与候选复杂度

- 候选总量：17254（来自 551 个 CIF）
- 平均每 CIF 候选数：31.3（中位数 26）
- 最小 / 最大候选数：1 / 70
- 结构特点：候选规模分布离散，长尾明显（36、46、70 候选数的 CIF 占比较高）

这是典型的"组内候选数量不均衡"排序问题，采用按 CIF 分组切分与组内排序指标是必要的。

### 7.2 交叉验证（5-fold）稳定性

CV 汇总（均值 ± 标准差）：

| 指标 | 均值 | 标准差 |
|------|------|--------|
| NDCG@1 | 0.7551 | ±0.0250 |
| NDCG@3 | 0.8051 | ±0.0131 |
| NDCG@5 | 0.8253 | ±0.0150 |
| MRR | 0.7964 | ±0.0278 |
| Hit@1 | 0.6374 | ±0.0374 |
| Hit@3 | 0.9564 | ±0.0249 |
| Hit@5 | 0.9830 | ±0.0123 |

- 前 3 名质量稳定（`NDCG@3` 波动小，std=0.013）
- `Hit@3 ≈ 95.6%`：Top-3 后续实跑可覆盖绝大多数 CIF 的最优候选
- `Hit@1 ≈ 63.7%`：只跑模型第一名仍有约三成漏检风险

### 7.3 独立测试集表现

| 指标 | 测试集 |
|------|--------|
| NDCG@1 | 0.7969 |
| NDCG@3 | 0.8179 |
| NDCG@5 | 0.8362 |
| MRR | 0.8103 |
| Hit@1 | 0.6861 |
| Hit@3 | 0.9343 |
| Hit@5 | 0.9562 |

测试集与 CV 同量级，未见明显退化，泛化表现正常。

### 7.4 特征重要性结论

当前保留的 6 个特征（按重要性排序）：

1. `total_site_entropy`（2803）
2. `volume`（2546）
3. `sphericity`（2136）
4. `valid_supercell_count`（1753）
5. `num_disordered_sites`（1740）
6. `lcmm`（1022）

结构复杂度（熵、无序位点数）与几何尺度（体积、球形度）是主要信号；`lcmm` 提供有效补充，支持继续保留。

### 7.5 工程建议

1. **默认推理**：`top_k=3`（命中率与计算成本的最佳平衡点）
2. **高稳妥场景**：`top_k=5`（进一步降低漏最优风险）
3. **单推荐**（`top_k=1`）：接受约三成漏检，或对关键样本启用回退复算
4. **数据集更新后**：重跑训练并对比 `ranker_report.json` 的 CV/Test 指标漂移

---

## 8. 指标解读与工程阈值

### 8.1 适用场景说明

本项目模型是"组内排序"，每个 `cif_file` 一组候选。指标应按 CIF 分组解释，不能做跨 CIF 的随机切分评估（会虚高评估泛化性能）。

### 8.2 指标家族与业务含义

**NDCG@k**
- 含义：前 k 名是否把高相关候选排前面（位置加权）
- 在本项目中：衡量 top-k 推荐的整体排序质量

**MRR（Mean Reciprocal Rank）**
- 含义：真实最优候选倒数排名的平均值
- 直觉：MRR 接近 1 表示经常第 1 名就命中最优

**Hit@k**
- 含义：真实最优是否落在前 k 名
- 业务最直观：可直接作为"预算 k 次实跑能覆盖最优"的成功率
- 例：`Hit@3 = 0.93` → 93% 的 CIF 在前三中命中最优

**MAP@k（Mean Average Precision）**
- 含义：前 k 的平均精确率，兼顾命中数量与位置
- 补充 NDCG/Hit，适合多相关候选场景

### 8.3 推荐工程阈值

| 指标 | 阈值 | 场景 |
|------|------|------|
| `Hit@1 ≥ 0.55` | 可用于"单推荐"场景 | 成本极紧 |
| `Hit@3 ≥ 0.85` | 可用于"Top-3 后续精算" | 推荐默认策略 |
| `NDCG@3 ≥ 0.75` | 前排排序质量较稳 | 通用 |
| `MRR ≥ 0.70` | 平均找到最优位置较靠前 | 通用 |

当 `Hit@3` 高但 `Hit@1` 一般时，优先采用"Top-3 送后续计算"策略。

### 8.4 成本-收益分析模板

假设每个 CIF 平均候选数为 $N$，仅计算 top-k：

$$\text{计算节省比} \approx 1 - \frac{k}{N}$$

若 $N=40$、$k=3$，节省约 92.5%。

决策建议：
- 更看重稳妥 → `k=5`
- 更看重成本 → `k=3`

### 8.5 常见误读

1. **只看 `Hit@1` 否定模型**：忽略 top-k 工作流的真实价值；应联合看 `Hit@3`。
2. **不分组随机切分**：会高估泛化性能，必须按 CIF 分组切分。
3. **只看单个指标**：建议至少联合看 `Hit@k + NDCG@k + MRR`。
