# 附录 C：sqsgenerator 深度调试手册（开发者）

> 目的：保留与第三方库交互时的“深调试套路”，供开发阶段定位棘手问题。

## 1. 适用边界

- 普通用户优先使用 `sqs_pd` 对外 API。
- 本附录仅用于开发调试：当你需要定位配置解析、优化行为、结果对象细节时使用。

代码对应：
- `src/sqs_pd/runtime/sqs_orchestrator.py`
- `src/sqs_pd/core/config_builder.py`

---

## 2. 调试总流程

1. 构建配置并落盘（保留现场）。  
2. `parse_config` 检查结构合法性。  
3. parse 通过后执行 `optimize`。  
4. 检查 `best()` 结果对象关键字段。  
5. 对比 `config` 与结果中 `species/coords` 的一致性。

---

## 3. 最小调试脚本

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

    print("parse passed, optimize...")
    result_pack = sqsgenerator.optimize(config)

    if not hasattr(result_pack, "best"):
        print("optimize returned object without best()")
        return

    best = result_pack.best()
    print("objective:", getattr(best, "objective", None))
    print("has species:", hasattr(best, "species"))
```

---

## 4. 典型调试检查点

### 配置层

- `structure.lattice` 是否为 3x3
- `structure.coords` 与 `structure.species` 长度是否一致
- split 模式下 `composition[*].sites` 是否可解析

### 结果层

- `best().objective` 是否可读
- `best().species` 长度是否符合超胞展开预期
- vacancy（`0`）是否在后处理时被正确过滤

---

## 5. 失败现场保全建议

- 始终保存：
  - `config.json`
  - parse 诊断四元组（`code/key/parameter/msg`）
  - 失败时的输入 CIF 路径与超胞参数
- 若批处理失败：记录 `(cif_file, size, l, w, h)` 作为故障键，便于复现。

---

## 6. 与项目日志联动

当使用 `generate_sqs(..., options=SQSOptions(log_file=...))` 时，可结合 JSON 日志快速回溯：

- 输入结构信息
- 关键流程日志
- sqsgenerator 配置快照

---

## 7. 最小验证

```bash
pytest tests/test_02_sqsgenerator_api.py -v
pytest tests/test_03_critical_semantics.py -v
```
