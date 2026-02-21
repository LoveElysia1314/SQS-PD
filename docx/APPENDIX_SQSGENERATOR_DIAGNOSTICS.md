# 附录：sqsgenerator 诊断手册

> 适用于开发调试阶段，需要定位 ParseError、配置合法性或优化行为问题时使用。  
> 普通用户优先查阅[故障排查与实践规则](TROUBLESHOOTING.md)。

---

## 第一部分：ParseError 诊断

### 1.1 关键事实

- `sqsgenerator.parse_config(...)` 可能**返回 ParseError 对象**，而不是抛出异常。
- 排查重点字段：`code`、`key`、`parameter`、`msg`。
- 本项目中最常见问题来源：
  1. `composition[*].sites` 语义错误
  2. split 子晶格计数与位点覆盖不一致
  3. 配置中夹带不被识别字段

代码对应：`src/sqs_pd/core/config_builder.py`、`src/sqs_pd/runtime/sqs_orchestrator.py`

### 1.2 可复用诊断函数

```python
import json
import sqsgenerator


def diagnose_parse_result(config: dict) -> tuple[bool, dict]:
    """返回 (is_ok, detail)"""
    parsed = sqsgenerator.parse_config(config)

    if "ParseError" in str(type(parsed)):
        detail = {
            "ok": False,
            "type": str(type(parsed)),
            "code": getattr(parsed, "code", None),
            "key": getattr(parsed, "key", None),
            "parameter": getattr(parsed, "parameter", None),
            "msg": getattr(parsed, "msg", None),
        }
        return False, detail

    return True, {"ok": True, "type": str(type(parsed))}


def print_diagnostic(config: dict) -> bool:
    ok, detail = diagnose_parse_result(config)
    print("=" * 72)
    print("sqsgenerator parse diagnostic")
    print("=" * 72)
    print(json.dumps(detail, ensure_ascii=False, indent=2))

    if ok:
        print("✓ parse passed")
        return True

    key = str(detail.get("key") or "")
    msg = str(detail.get("msg") or "")

    if "sites" in key or "sites" in msg:
        print("建议: 检查 composition[*].sites 是否为字符串标签并能在 structure.species 中匹配。")
    if "Unknown parameter" in msg:
        print("建议: 清理配置中的非 sqsgenerator 支持字段。")
    if "distribute" in msg and "sites" in msg:
        print("建议: 检查子晶格可分配位点数与元素计数总和是否一致。")

    return False
```

### 1.3 最小诊断流程

1. 调用 `print_diagnostic(config)`，拿到四元信息（code / key / parameter / msg）。
2. 若 `key/msg` 指向 `sites`：优先检查字符串标签匹配关系。
3. 若指向分配数量：核对每个子晶格元素总数与可分配位点数量。
4. 若是未知参数：清理配置后复测。
5. parse 通过后再进入 `optimize`。

### 1.4 常见错误模式与修复

**模式 A：`sites` 用错（字符串 vs 索引）**
- 表现：`distribute ... atoms on ... sites`
- 修复：优先使用字符串标签；索引列表仅在你明确掌握展开后位点编号时使用。

**模式 B：占位符漂移**
- 表现：`sites` 标签在 `structure.species` 中找不到。
- 修复：保持 `config_builder` 的占位符分配策略，不做外部"元素规范化"。

**模式 C：字段污染**
- 表现：`Unknown parameter ...`
- 修复：只保留 `structure / composition / sublattice_mode / iterations` 等有效字段。

---

## 第二部分：sqsgenerator 深度调试

### 2.1 调试总流程

1. 构建配置并落盘（保留现场）。
2. `parse_config` 检查结构合法性。
3. parse 通过后执行 `optimize`。
4. 检查 `best()` 结果对象关键字段。
5. 对比 `config` 与结果中 `species/coords` 的一致性。

### 2.2 最小调试脚本

```python
import json
from pathlib import Path
import sqsgenerator


def debug_sqsgenerator(config: dict, out_dir: str = "debug_artifacts") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved config: {cfg_path}")

    parsed = sqsgenerator.parse_config(config)
    if "ParseError" in str(type(parsed)):
        print("parse failed")
        print("code:", getattr(parsed, "code", None))
        print("key:", getattr(parsed, "key", None))
        print("parameter:", getattr(parsed, "parameter", None))
        print("msg:", getattr(parsed, "msg", None))
        return

    print("parse passed, running optimize...")
    result_pack = sqsgenerator.optimize(config)

    if not hasattr(result_pack, "best"):
        print("optimize returned object without best()")
        return

    best = result_pack.best()
    print("objective:", getattr(best, "objective", None))
    print("has species:", hasattr(best, "species"))
```

### 2.3 典型调试检查点

**配置层：**
- `structure.lattice` 是否为 3×3
- `structure.coords` 与 `structure.species` 长度是否一致
- split 模式下 `composition[*].sites` 是否可解析

**结果层：**
- `best().objective` 是否可读
- `best().species` 长度是否符合超胞展开预期
- vacancy（`0`）是否在后处理时被正确过滤

### 2.4 失败现场保全建议

始终保存：
- `config.json`（落盘配置快照）
- parse 诊断四元组（`code / key / parameter / msg`）
- 失败时的输入 CIF 路径与超胞参数

若批处理失败：记录 `(cif_file, size, l, w, h)` 作为故障键，便于单独复现。

### 2.5 与项目日志联动

使用 `generate_sqs(..., options=SQSOptions(log_file="out.json"))` 时，JSON 日志包含：
- 输入结构信息
- 关键流程日志
- sqsgenerator 配置快照

结合 `ranker_diagnostics.png`（训练产物）与 `per_cif_test_metrics.csv` 可快速定位问题样本。

---

## 最小验证

```bash
pytest tests/test_02_sqsgenerator_api.py -v
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
```
