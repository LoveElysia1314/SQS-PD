# 试错经验与规则

> 本文档保留“高价值试错经验”，但以可复用规则表达，不保留历史过程。

## 规则 1：`sites` 优先使用字符串标签

- 错误表现：`ParseError`、位点分配数量不匹配
- 正确做法：`composition[*].sites` 使用占位符标签字符串
- 最小验证：`pytest tests/test_03_critical_semantics.py -v`

## 规则 2：占位符是“子晶格标签”，不是元素物理含义

- 错误表现：自动元素规范化后，子晶格独立性被破坏
- 正确做法：在配置构建中保持占位符策略一致
- 参考：`src/sqs_pd/core/config_builder.py`

## 规则 3：vacancy 统一用 `"0"`

- 错误表现：配置解析异常、后处理异常
- 正确做法：空位只用 `"0"`，不混用其他符号

## 规则 4：候选超胞先“合法”再“择优”

- 错误表现：拍脑袋指定超胞导致误差不可控
- 正确做法：先 dry-run 枚举合法超胞候选集，再按 RSS/模型排序
- 参考：`src/sqs_pd/core/supercell_optimizer.py`

## 规则 5：批处理必须使用稳定幂等键

- 错误表现：重复计算、漏算
- 正确做法：固定键 `(cif_file, size, l, w, h)`
- 参考：`src/sqs_pd/batching/batch_service.py`、`src/sqs_pd/batching/batch_storage.py`

## 规则 6：输出策略用 profile 管理

- 错误表现：日志和报告分散、覆盖冲突
- 正确做法：优先使用 `generate_sqs_with_report(..., output_profile=...)`
- 参考：`src/sqs_pd/interface/api.py`

---

## 深度阅读

- [附录 A：ParseError 诊断手册](APPENDIX_PARSEERROR_DIAGNOSTICS.md)
- [附录 C：sqsgenerator 深度调试手册](APPENDIX_SQSGENERATOR_DEBUG_PLAYBOOK.md)
