# 故障排查与实践规则

---

## 第一部分：实践规则

以下规则来自实际调试中高价值的试错经验，以可复用形式记录。

### 规则 1：`sites` 优先使用字符串标签

- **错误表现**：`ParseError`、位点分配数量不匹配
- **正确做法**：`composition[*].sites` 使用占位符标签字符串
- **最小验证**：`pytest tests/test_03_critical_semantics.py -v`

### 规则 2：占位符是"子晶格标签"，不是元素物理含义

- **错误表现**：自动元素规范化后，子晶格独立性被破坏
- **正确做法**：在配置构建中保持占位符策略一致
- **参考**：`src/sqs_pd/core/config_builder.py`

### 规则 3：vacancy 统一用 `"0"`

- **错误表现**：配置解析异常、后处理异常
- **正确做法**：空位只用 `"0"`，不混用其他符号

### 规则 4：候选超胞先"合法"再"择优"

- **错误表现**：拍脑袋指定超胞导致误差不可控
- **正确做法**：先 dry-run 枚举合法超胞候选集，再按 RSS / 模型排序
- **参考**：`src/sqs_pd/core/supercell_optimizer.py`

### 规则 5：批处理必须使用稳定幂等键

- **错误表现**：重复计算、漏算
- **正确做法**：固定键 `(cif_file, size, l, w, h)`
- **参考**：`src/sqs_pd/batching/batch_service.py`、`src/sqs_pd/batching/batch_storage.py`

### 规则 6：输出策略用 profile 管理

- **错误表现**：日志和报告分散、覆盖冲突
- **正确做法**：优先使用 `generate_sqs_with_report(..., output_profile=...)`
- **参考**：`src/sqs_pd/interface/api.py`

---

## 第二部分：故障排查

### 1) `ParseError`（配置解析失败）

**常见触发：**
- `composition[*].sites` 与 `structure.species` 不匹配
- `composition` 计数与超胞位点总数不一致
- 非法字段或字段类型错误

**排查顺序：**
1. 检查 `build_sqs_config` 生成的配置
2. 确认 `sites` 语义（字符串标签优先）
3. 确认 vacancy 只用 `"0"`

代码对应：`src/sqs_pd/core/config_builder.py`、`src/sqs_pd/runtime/sqs_orchestrator.py`

---

### 2) 优化失败或返回空结果

**常见触发：**
- 结构/配置虽然可解析，但优化空间不合法
- 迭代次数过低导致无稳定结果

**建议：**
- 先 dry-run 验证候选合法性
- 提升 `iterations`
- 打开 `console_log` 与 `log_file`

---

### 3) 结构后处理失败

**常见触发：**
- 全部位点被视为空位，无法重建 `Structure`
- species 序列与坐标展开长度不一致

代码对应：`src/sqs_pd/runtime/sqs_orchestrator.py` — `post_process_result`

---

### 4) 批处理重复计算或错误跳过

**常见触发：**
- 去重键不稳定
- CSV 人工修改导致键字段异常

建议键：`(cif_file, size, l, w, h)`

代码对应：`src/sqs_pd/batching/batch_service.py`、`src/sqs_pd/batching/batch_storage.py`

---

### 5) 模型推理找不到模型

**常见触发：**
- 默认模型目录缺文件
- `ranker_inference_config.json` 缺失

**建议：**
- 先训练并固化默认模型（`python examples/train_sqs_ranker.py`）
- 或显式传入 `model_path` 与 `model_config_path`

代码对应：`src/sqs_pd/ranking/model_ranker_inference.py`

---

## 最小验证

```bash
pytest tests/test_02_sqsgenerator_api.py -v
pytest tests/test_03_critical_semantics.py -v
```

## 深度阅读

- [附录：sqsgenerator 诊断手册](APPENDIX_SQSGENERATOR_DIAGNOSTICS.md)
