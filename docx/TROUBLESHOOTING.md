# 故障排查

## 1) `ParseError`（配置解析失败）

### 常见触发
- `composition[*].sites` 与 `structure.species` 不匹配
- `composition` 计数与超胞位点总数不一致
- 非法字段或字段类型错误

### 排查顺序
1. 检查 `build_sqs_config` 生成的配置
2. 确认 `sites` 语义（字符串标签优先）
3. 确认 vacancy 只用 `"0"`

对应代码：
- `src/sqs_pd/core/config_builder.py`
- `src/sqs_pd/runtime/sqs_orchestrator.py::run_optimization`

---

## 2) 优化失败或返回空结果

### 常见触发
- 结构/配置虽然可解析，但优化空间不合法
- 迭代次数过低导致无稳定结果

### 建议
- 先 dry-run 验证候选合法性
- 提升 `iterations`
- 打开 `console_log` 与 `log_file`

---

## 3) 结构后处理失败

### 常见触发
- 全部位点被视为空位，无法重建 `Structure`
- species 序列与坐标展开长度不一致

对应代码：
- `src/sqs_pd/runtime/sqs_orchestrator.py::post_process_result`

---

## 4) 批处理重复计算或错误跳过

### 常见触发
- 去重键不稳定
- CSV 人工修改导致键字段异常

建议键：`(cif_file, size, l, w, h)`

对应代码：
- `src/sqs_pd/batching/batch_service.py`
- `src/sqs_pd/batching/batch_storage.py`

---

## 5) 模型推理找不到模型

### 常见触发
- 默认模型目录缺文件
- `ranker_inference_config.json` 缺失

建议
- 先训练并固化默认模型
- 或显式传入 `model_path` 与 `model_config_path`

对应代码：
- `src/sqs_pd/ranking/model_ranker_inference.py`
- `src/sqs_pd/ranking/ml_ranker_trainer.py`

---

## 最小验证

```bash
pytest tests/test_02_sqsgenerator_api.py -v
pytest tests/test_03_critical_semantics.py -v
```

## 深度阅读

- [附录 A：ParseError 诊断手册](APPENDIX_PARSEERROR_DIAGNOSTICS.md)
- [附录 C：sqsgenerator 深度调试手册](APPENDIX_SQSGENERATOR_DEBUG_PLAYBOOK.md)
