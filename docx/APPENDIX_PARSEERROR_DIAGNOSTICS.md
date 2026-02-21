# 附录 A：ParseError 诊断手册（深度版）

> 目的：保留可直接复用的 ParseError 深度诊断能力，避免主文档过长。

## 1. 关键事实

- `sqsgenerator.parse_config(...)` 可能返回 ParseError 对象，而不是抛出异常。
- 排查重点字段：`code`、`key`、`parameter`、`msg`。
- 在本项目中，最常见问题来自：
  1. `composition[*].sites` 语义错误
  2. split 子晶格计数与位点覆盖不一致
  3. 配置中夹带不被识别字段

代码对应：
- `src/sqs_pd/core/config_builder.py`
- `src/sqs_pd/runtime/sqs_orchestrator.py`

---

## 2. 可复用诊断函数

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

---

## 3. 最小诊断流程（建议顺序）

1. 先调用 `print_diagnostic(config)`，拿到四元信息。  
2. 若 `key/msg` 指向 `sites`：优先检查字符串标签匹配关系。  
3. 若指向分配数量：核对每个子晶格元素总数与可分配位点数量。  
4. 若是未知参数：清理配置后复测。  
5. parse 通过后再进入 `optimize`。

---

## 4. 常见错误模式与修复

### 模式 A：`sites` 用错（字符串 vs 索引）

- 表现：`distribute ... atoms on ... sites`
- 修复：优先使用字符串标签；索引列表仅在你明确掌握展开后位点编号时使用。

### 模式 B：占位符漂移

- 表现：`sites` 标签在 `structure.species` 中找不到。
- 修复：保持 `config_builder` 的占位符分配策略，不做外部“元素规范化”。

### 模式 C：字段污染

- 表现：`Unknown parameter ...`
- 修复：只保留 `structure/composition/sublattice_mode/iterations` 等有效字段。

---

## 5. 与测试联动

建议定期执行：

```bash
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
```

这些测试可快速覆盖大多数 ParseError 根因场景。
