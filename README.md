# SQS-PD

SQS-PD 是一个面向部分无序晶体结构的 SQS 生成工具，提供从 CIF 无序识别、超胞候选推荐到 SQS 优化与批量数据流水线的一体化能力。

## 文档入口

当前文档已集中到 `docx/`：

- [文档中心](docx/README.md)
- [快速开始](docx/QUICK_START.md)
- [核心概念](docx/CORE_CONCEPTS.md)
- [API 参考](docx/API_REFERENCE.md)
- [工作流指南](docx/WORKFLOWS.md)
- [故障排查](docx/TROUBLESHOOTING.md)
- [试错经验与规则](docx/PITFALLS_AND_RULES.md)
- [ML 指南](docx/ML_GUIDE.md)
- [开发者模块地图](docx/DEVELOPER_MAP.md)

## 安装

```bash
pip install pymatgen sqsgenerator
pip install -e .
```

## 最小示例

```python
from sqs_pd import generate_sqs

result = generate_sqs("data/input/demo_sd.cif")
print(result.success, result.objective, result.supercell_used)
```
