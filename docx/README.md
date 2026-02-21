# SQS-PD 文档中心（docx）

本目录是项目的**统一文档入口**，目标是：

- 高信息量保留
- 与当前代码实现一致
- 精简集约（减少重复与分散）
- 保留关键试错经验（沉淀为“稳定规则”，不保留历史过程）

## 推荐阅读顺序

1. [快速开始](QUICK_START.md)
2. [工作流指南](WORKFLOWS.md)
3. [核心概念](CORE_CONCEPTS.md)
4. [API 参考](API_REFERENCE.md)
5. [故障排查](TROUBLESHOOTING.md)
6. [试错经验与规则](PITFALLS_AND_RULES.md)
7. [ML 指南](ML_GUIDE.md)
8. [开发者模块地图](DEVELOPER_MAP.md)

## 深度附录

1. [附录 A：ParseError 诊断手册](APPENDIX_PARSEERROR_DIAGNOSTICS.md)
2. [附录 B：ML 排序指标解读](APPENDIX_ML_METRICS_INTERPRETATION.md)
3. [附录 C：sqsgenerator 深度调试手册](APPENDIX_SQSGENERATOR_DEBUG_PLAYBOOK.md)

## 文档治理规则（简版）

- 单一事实来源（Single Source of Truth）：同一主题只在一处主讲
- 代码优先：文档描述必须能在 `src/sqs_pd` 对应模块中找到实现
- 去历史化：不记录“某月某轮重构过程”，只保留当前稳定结论
- 经验产品化：保留试错经验，但以“规则/反例/检查清单”形式呈现

## 术语统一约定（全仓）

- 合法超胞候选集：dry-run 在误差约束下得到的全部可行候选
- 最小 RSS 候选子集（硬约束）：从合法超胞候选集中按最小 RSS 过滤后的候选子集
- 模型排序：仅在“最小 RSS 候选子集（硬约束）”内执行
- top-k 实跑：对排序后的前 k 个候选逐个执行 SQS
- 输出配置档（`output_profile`）：`silent/cif/logs/full`（含别名）
- 产物保留策略（`artifact_policy`）：`none/best/all`

## 当前状态

- 文档主体系已迁移到 `docx/`
- 后续以本目录 `README.md + DEVELOPER_MAP.md` 作为持续治理基线
