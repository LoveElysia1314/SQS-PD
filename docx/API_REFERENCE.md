# API 参考（公开接口）

> 本页只描述对外稳定接口；内部实现细节见[开发者模块地图](DEVELOPER_MAP.md)。

## 核心数据对象

### `SQSOptions`

定义位置：`src/sqs_pd/core/options.py`

主要字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `supercell` | `tuple[int,int,int] \| None` | `None` | 超胞尺寸；`None` 或 `"auto"` 时自动选择 |
| `iterations` | `int` | `DEFAULT_ITERATIONS` | sqsgenerator 迭代次数 |
| `output_file` | `str \| None` | `None` | 输出 CIF 路径 |
| `log_file` | `str \| None` | `None` | 输出 JSON 日志路径 |
| `return_all` | `bool` | `False` | 是否在结果中附带原始分析对象 |
| `include_analysis_details` | `bool` | `False` | 是否附带超胞/配置详细信息 |
| `console_log` | `bool` | `False` | 是否在控制台输出日志 |
| `debug` | `bool` | `False` | 是否开启调试级日志 |

补充说明：
- `supercell=None`（或 `"auto"`）时，当前实现先尝试模型 `top-1` 推荐；
- 若模型推荐异常或不可用，则回退到合法超胞候选集中 RSS 最小且 size 最小的规格。

### `DisorderAnalysisOptions`

定义位置：`src/sqs_pd/core/options.py`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `group_by` | `str` | `"label"` | 位点分组方式 |
| `tol` | `float` | `OCCUPANCY_TOLERANCE` | 占据数归一化容差 |

### `SQSResult`

定义位置：`src/sqs_pd/interface/api.py`

常用字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | `bool` | 是否成功 |
| `structure` | `Structure \| None` | 优化后的结构 |
| `objective` | `float \| None` | 目标函数值（越小越好） |
| `time` | `float` | 总运行时间（秒） |
| `supercell_used` | `tuple \| None` | 实际使用的超胞尺寸 |
| `disorder_types` | `list[str] \| None` | 检测到的无序类型列表 |
| `error` | `str \| None` | 失败时的错误信息 |

方法：
- `save_report(filename, format="txt"|"json")` — 保存文本或 JSON 报告
- `disorder_type` — 属性，返回无序类型字符串（`disorder_types` 的 `"+"` 连接形式）

### `DisorderAnalysisResult`

定义位置：`src/sqs_pd/interface/api.py`

| 字段 | 说明 |
|------|------|
| `success` | 是否成功 |
| `disorder_type` | 无序类型字符串 |
| `disorder_types` | 无序类型列表 |
| `site_results` | 每个位点的分类详情 |
| `warnings` | 解析警告列表 |
| `error` | 错误信息 |

属性：`num_sites`、`num_disordered_sites`

---

## 函数接口

### `generate_sqs(input_file, options=None) → SQSResult`

统一主入口：执行完整 SQS 流程。

**参数：**
- `input_file`：CIF 文件路径（`str | Path`）或 pymatgen `Structure` 对象
- `options`：`SQSOptions` 实例

**自动超胞策略**（`options.supercell` 缺省/auto）：
1. 优先模型推荐（top-1）
2. 异常回退到合法超胞候选集中最小 RSS、并列取最小 size

---

### `generate_sqs_with_report(...) → SQSResult`

便捷入口：自动输出 CIF / JSON 日志 / TXT 报告。

**主要参数：**

| 参数 | 说明 |
|------|------|
| `input_file` | CIF 文件路径 |
| `output_dir` | 输出根目录（默认 `"output"`） |
| `output_folder_name` | 输出子目录名模板，支持 `{stem}`、`{supercell}` |
| `output_profile` | 输出配置档（见下表） |
| `supercell` | 超胞尺寸，`None` 则自动计算 |
| `iterations` | 迭代次数 |
| `console_log` | 是否控制台输出（默认 `True`） |

`output_profile` 取值：

| 值 | 别名 | 含义 |
|----|------|------|
| `silent` | `none` | 不输出任何文件 |
| `cif` | `cif-only`, `cif_only` | 只输出 CIF |
| `logs` | `log` | 输出 CIF + JSON 日志 |
| `full` | `all` | 输出 CIF + JSON 日志 + TXT 报告 |

---

### `analyze_cif_disorder(input_file, options=None) → DisorderAnalysisResult`

只做 CIF 无序分析，不执行优化。

---

### `dry_run_recommend_supercell(input_file, max_error=0.0005, verbose=False) → dict`

只做合法超胞候选集分析与推荐，不执行 SQS 优化。

返回字典关键字段：`recommended_supercell`、`all_candidates`、`num_candidates`、`optimization_success`、`rss`、`disorder_types`

---

### `batch_recommend_supercells(input_files, max_error=0.0005, verbose=False) → list[dict]`

批量 dry-run 推荐入口，复用与单文件 dry-run 一致的分析与错误策略。

---

### `recommend_supercells_with_model(input_file, top_k=None, max_error=0.0005, model_path=None, model_config_path=None) → dict`

先 dry-run 枚举合法超胞候选集，再施加"最小 RSS 候选子集（硬约束）"，最后仅在该子集内做模型排序。

返回字典关键字段：

| 字段 | 说明 |
|------|------|
| `ranked_candidates` | 模型排序结果（仅包含最小 RSS 候选子集） |
| `recommended_supercell` | `ranked_candidates` 第一名 |
| `num_candidates` | 最小 RSS 候选子集数量（排序输入规模） |
| `num_candidates_all` | dry-run 合法超胞候选集总数（过滤前） |
| `constraint` | 当前固定为 `"min-rss-only"` |

---

### `run_topk_sqs_with_model(input_file, top_k=5, ...) → dict`

单文件 top-k 推荐并实跑接口。

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | `5` | 推荐并实跑的候选数量 |
| `max_error` | `0.0005` | 超胞误差约束 |
| `iterations` | `DEFAULT_ITERATIONS` | 每次 SQS 的迭代次数 |
| `artifact_policy` | `"best"` | 产物保留策略（见下表） |
| `output_dir` | `"data/output"` | 输出根目录 |
| `output_folder_name` | `None` | 支持 `{stem}`、`{top_k}` 模板变量 |
| `save_comparison_csv` | `False` | 是否导出 model-vs-actual 对比 CSV |
| `objective_quantum` | `0.1` | 并列判定精度（四舍五入量子） |

`artifact_policy` 取值：

| 值 | 含义 |
|----|------|
| `none` | 不落地任何候选文件，仅返回内存结果 |
| `best` | 仅落地真实最优候选（`best_by_actual_objective.*`） |
| `all` | 落地 top-k 全部候选文件 |

> `artifact_policy` 仅接受 `none / best / all`（大小写不敏感），其他值抛出 `ValueError`。

CSV 导出规则：`save_comparison_csv=True` 且 `top_k > 1` 时导出对比 CSV；`top_k=1` 时不导出（单候选无需对比）。

---

## CLI

定义位置：`src/sqs_pd/interface/cli.py`

```bash
# 分析单个 CIF
python -m sqs_pd.interface.cli analyze data/input/demo_sd.cif

# 批量分析
python -m sqs_pd.interface.cli analyze data/input/demo_sd.cif data/input/demo_pd.cif
```

---

## 最小验证

```bash
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
pytest tests/test_05_cif_disorder_analysis.py -v
```
