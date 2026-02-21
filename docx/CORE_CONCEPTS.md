# 核心概念

## 1. 无序类型（SD / PD / SPD / ordered）

位点层规则（简化）：

- `ordered`：单元素且总占据约等于 1
- `SD`：多元素且总占据约等于 1（替代位无序）
- `PD`：单元素且总占据小于 1（含空位语义）
- `SPD`：多元素且总占据小于 1（混合无序）

结构层可为多标签组合（例如 `SD+PD`）。

---

## 2. 超胞候选思想（约束 + 优化）

当前实现不是"只讲 LCM"，而是两层思路：

1. 由占据数近似误差约束筛选合法超胞候选集（`max_error`，默认 0.0005）
2. 在合法超胞候选集中按目标（默认 RSS 最小）择优

因此需同时关注：
- **约束来源**：占据数与超胞规模匹配误差
- **优化目标**：RSS、候选数量、可选模型排序

当前 `auto` 选胞策略：
- 先模型 `top-1`；
- 模型路径异常/不可用时，回退到合法超胞候选集中 RSS 最小，并在并列时取最小 size。

> 旧的"LCM + 最小规模阈值"默认选胞逻辑已移除；LCM 仍可作为统计/特征概念存在，但不再作为默认 auto 选胞策略。

---

## 3. split 模式与子晶格映射

配置核心：
- 使用 `sublattice_mode = "split"`
- 结构中用占位符标签标记可优化位点
- `composition` 仅描述无序位点的元素分配

关键语义：
- 占位符是"子晶格标签"，不是化学结论。
- vacancy 在配置中统一用 `"0"`。

---

## 4. 端到端执行链

```
API → orchestrator → config_builder → sqsgenerator → post-process → SQSResult
```

各层职责：
- `interface/api.py`：公开接口，自动超胞策略，产物管理
- `runtime/sqs_orchestrator.py`：流程协调（prepare/build/optimize/post-process）
- `core/config_builder.py`：构建 split 模式配置
- `runtime/io_utils.py`：输出路径与文件写入

---

## 最小验证

```bash
pytest tests/test_01_supercell_basics.py -v
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
```
