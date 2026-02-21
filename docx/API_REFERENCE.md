# API 参考（公开接口）

> 本页只描述对外稳定接口；内部实现细节见开发者地图。

## 核心数据对象

### `SQSOptions`

定义位置：`src/sqs_pd/core/options.py`

主要字段：
- `supercell: tuple[int, int, int] | None`
- `iterations: int`
- `output_file: str | None`
- `return_all: bool`
- `include_analysis_details: bool`
- `log_file: str | None`
- `console_log: bool`
- `debug: bool`

补充说明：
- `supercell=None`（或 `"auto"`）时，当前实现会先尝试模型 `top-1` 推荐；
- 若模型推荐异常或不可用，则回退到合法超胞候选集中 `RSS` 最小且 `size` 最小的规格。

### `DisorderAnalysisOptions`

定义位置：`src/sqs_pd/core/options.py`

- `group_by: str = "label"`
- `tol: float`

### `SQSResult`

定义位置：`src/sqs_pd/interface/api.py`

常用字段：
- `success`
- `structure`
- `objective`
- `time`
- `supercell_used`
- `disorder_types`
- `error`

方法：
- `save_report(filename, format="txt"|"json")`

---

## 函数接口

### `generate_sqs(input_file, options=None) -> SQSResult`

统一主入口：执行完整 SQS 流程。

自动超胞策略（`options.supercell` 缺省/auto）：
- 优先模型推荐（top-1）；
- 异常回退到合法超胞候选集中最小 RSS、并列取最小 size。

### `generate_sqs_with_report(...) -> SQSResult`

便捷入口：可自动输出 CIF/JSON/TXT 报告。

当前不再包含 `min_supercell_multiplier` 参数。

`output_folder_name` 模板变量：
- `{stem}`：输入 CIF 文件名（不含后缀）
- `{supercell}`：超胞标签（如 `4x3x2`，自动模式为 `auto`）

`output_profile` 取值：
- 主值：`silent` / `cif` / `logs` / `full`
- 别名：`none`(=silent), `cif-only`/`cif_only`(=cif), `log`(=logs), `all`(=full)

### `analyze_cif_disorder(input_file, options=None) -> DisorderAnalysisResult`

只做 CIF 无序分析，不执行优化。

### `dry_run_recommend_supercell(input_file, max_error=0.0005, verbose=False) -> dict`

只做合法超胞候选集分析与推荐，不执行优化。

### `batch_recommend_supercells(input_files, max_error=0.0005, verbose=False) -> list[dict]`

批量 dry-run 推荐入口：复用与单文件 dry-run 一致的分析与错误策略。

### `recommend_supercells_with_model(input_file, top_k=None, max_error=0.0005, model_path=None, model_config_path=None) -> dict`

先 dry-run 枚举合法超胞候选集，再施加“最小 RSS 候选子集（硬约束）”，最后仅在该子集内做模型排序。

返回字典关键字段：
- `ranked_candidates`：模型排序结果（仅包含最小 RSS 候选子集）
- `recommended_supercell`：`ranked_candidates` 第一名
- `num_candidates`：最小 RSS 候选子集数量（排序输入规模）
- `num_candidates_all`：dry-run 合法超胞候选集总数（过滤前）
- `constraint`：当前固定为 `"min-rss-only"`

### `run_topk_sqs_with_model(input_file, top_k=5, ..., artifact_policy="best", save_comparison_csv=False, ...) -> dict`

单文件 top-k 推荐并实跑接口（正式 API）：
- 先做模型推荐（仅最小 RSS 候选子集（硬约束）内）
- 再逐个执行 top-k SQS
- 可配置产物保留策略：
	- `artifact_policy="none"`：不落地候选文件
	- `artifact_policy="best"`：仅落地真实最优候选（CIF + JSON + _sqsgenerator.json）
	- `artifact_policy="all"`：落地 top-k 全部候选文件

`artifact_policy` 当前仅接受：`none|best|all`（大小写不敏感，其他值会抛出 `ValueError`）。

CSV 导出规则：
- `save_comparison_csv=True` 且 `top_k>1` 时导出对比 CSV；
- 默认文件名 `topn_model_vs_actual.csv`（可通过 `comparison_csv_name` 覆盖）。

`output_folder_name` 模板变量：
- `{stem}`：输入 CIF 文件名（不含后缀）
- `{top_k}`：请求评估的 top-k 数值

---

## CLI

定义位置：`src/sqs_pd/interface/cli.py`

```bash
python -m sqs_pd.interface.cli analyze data/input/demo_sd.cif
python -m sqs_pd.interface.cli analyze data/input/demo_sd.cif data/input/demo_pd.cif
```

---

## 最小验证

```bash
pytest tests/test_03_critical_semantics.py -v
pytest tests/test_04_project_api.py -v
pytest tests/test_05_cif_disorder_analysis.py -v
```
