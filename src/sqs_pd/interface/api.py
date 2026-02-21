"""
SQS-PD API - 统一接口
=====================================
简洁的公开 API，用于SQS生成

用法示例：
    >>> from sqs_pd import generate_sqs, SQSOptions
    >>>
    >>> # 最简单的用法
    >>> result = generate_sqs("input.cif")
    >>> if result.success:
    ...     result.structure.to(filename="output.cif")
    ...     print(f"Objective: {result.objective}")
    >>>
    >>> # 自定义超胞和迭代数
    >>> options = SQSOptions(supercell=(4, 3, 2), iterations=1000000)
    >>> result = generate_sqs("input.cif", options=options)
    >>>
    >>> # dry-run 推荐（不执行 SQS）
    >>> from sqs_pd import dry_run_recommend_supercell
    >>> info = dry_run_recommend_supercell("input.cif")
    >>> print(info["recommended_supercell"])
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from pymatgen.core import Structure
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import csv

from ..runtime import sqs_orchestrator
from ..analysis import cif_disorder_analyzer
from ..foundation.constants import (
    DEFAULT_ITERATIONS,
    OCCUPANCY_TOLERANCE,
)
from ..runtime.logging_utils import ensure_console_handler, get_logger
from ..core.options import DisorderAnalysisOptions, SQSOptions
from ..runtime.dry_run import analyze_cif_and_recommend_supercell, batch_analyze_cifs
from ..ranking.model_ranker_inference import recommend_supercells_by_model
from ..analysis.disorder_analyzer import analyze_structure
from ..core.supercell_optimizer import find_optimal_supercell
from ..runtime.io_utils import (
    ensure_parent_dir,
    write_json_file,
    write_text_file,
    build_output_prefix,
    resolve_output_profile,
    normalize_artifact_policy,
    derive_sqsgenerator_config_path,
    build_topk_output_root,
    build_topk_candidate_prefix,
    build_topk_best_artifact_paths,
)

_logger = get_logger()

TopkRow = Dict[str, Any]
TopkArtifacts = Dict[str, str]


def _as_supercell_tuple(value: Any) -> Optional[Tuple[int, int, int]]:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(x) for x in value)
    if isinstance(value, list) and len(value) == 3:
        return tuple(int(x) for x in value)
    return None


def _is_auto_supercell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"auto", "none", "default"}
    return False


def _pick_min_rss_smallest_size(
    candidates: List[Dict[str, Any]],
) -> Optional[Tuple[int, int, int]]:
    normalized: List[Dict[str, Any]] = []
    for cand in candidates:
        supercell = _as_supercell_tuple(cand.get("supercell"))
        size_raw = cand.get("size")
        rss_raw = cand.get("rss")
        if supercell is None:
            continue
        try:
            size = int(size_raw)
            rss = float(rss_raw)
        except Exception:
            continue
        normalized.append({"supercell": supercell, "size": size, "rss": rss})

    if not normalized:
        return None

    min_rss = min(row["rss"] for row in normalized)
    min_rss_rows = [
        row for row in normalized if abs(float(row["rss"]) - float(min_rss)) <= 1e-12
    ]
    min_rss_rows.sort(key=lambda row: row["size"])
    return min_rss_rows[0]["supercell"]


def _resolve_auto_supercell_for_cif(cif_file: Union[str, Path]) -> Tuple[int, int, int]:
    model_error: Optional[str] = None
    try:
        ranked = recommend_supercells_by_model(cif_file=cif_file, top_k=1)
        if ranked.get("success"):
            recommended = _as_supercell_tuple(ranked.get("recommended_supercell"))
            if recommended is not None:
                return recommended
        model_error = str(
            ranked.get("message") or "model ranking did not return a valid candidate"
        )
    except Exception as e:
        model_error = str(e)

    dry = analyze_cif_and_recommend_supercell(cif_file=cif_file, max_error=0.0005)
    if dry.get("optimization_success", False):
        all_candidates = list(dry.get("all_candidates") or [])
        fallback_sc = _pick_min_rss_smallest_size(all_candidates)
        if fallback_sc is not None:
            return fallback_sc

        recommended = _as_supercell_tuple(dry.get("recommended_supercell"))
        if recommended is not None:
            return recommended

    fallback_message = str(
        dry.get("message") or "RSS fallback did not produce a valid candidate"
    )
    if model_error:
        raise ValueError(
            "Auto supercell failed: model top-1 unavailable/failed "
            f"({model_error}); RSS fallback failed ({fallback_message})"
        )
    raise ValueError(f"Auto supercell failed: {fallback_message}")


def _resolve_auto_supercell_for_structure(structure: Structure) -> Tuple[int, int, int]:
    analysis = analyze_structure(structure, verbose=False)
    occupancies = list(analysis.get("occupancies") or [])
    result = find_optimal_supercell(
        occupancies,
        max_error=0.0005,
        prefer_smaller=True,
    )
    if not result.get("success"):
        raise ValueError(
            str(result.get("message") or "No valid supercell from RSS optimization")
        )

    supercell = _as_supercell_tuple(result.get("supercell"))
    if supercell is None:
        raise ValueError("RSS optimization returned invalid supercell format")
    return supercell


@dataclass
class SQSResult:
    """SQS生成结果的包装类"""

    success: bool
    structure: Optional[Structure] = None
    objective: Optional[float] = None
    time: float = 0.0
    supercell_used: Optional[Tuple[int, int, int]] = None
    disorder_types: Optional[List[str]] = None
    error: Optional[str] = None
    analysis_details: Optional[Dict[str, Any]] = None  # 详细分析信息（超胞、配置等）
    log_messages: Optional[List[str]] = None  # 流程日志记录
    analysis: Optional[Dict[str, Any]] = None  # return_all=True 时的无序分析结果
    config: Optional[Dict[str, Any]] = None  # return_all=True 时的 sqsgenerator 配置
    raw_result: Optional[Any] = None  # return_all=True 时的 sqsgenerator 原始结果对象

    @property
    def disorder_type(self) -> str:
        """生成 disorder_type 字符串表示（向后兼容）"""
        if not self.disorder_types:
            return "ordered"
        return "+".join(self.disorder_types)

    def save_report(self, filename: str, format: str = "txt") -> bool:
        """
        保存详细报告

        Args:
            filename: 输出文件名
            format: 报告格式 (txt, json)

        Returns:
            是否保存成功
        """
        if not self.success:
            _logger.error("Cannot save report for failed result")
            return False

        try:
            if format == "json":
                report_data = {
                    "success": self.success,
                    "objective": self.objective,
                    "time": self.time,
                    "supercell_used": self.supercell_used,
                    "disorder_types": self.disorder_types,
                    "analysis_details": self.analysis_details,
                    "log_messages": self.log_messages,
                }
                write_json_file(filename, report_data)
            else:  # txt format
                lines = []
                lines.append("=" * 70)
                lines.append("SQS-PD 生成报告")
                lines.append("=" * 70)
                lines.append("")
                lines.append(f"成功: {self.success}")
                lines.append(
                    f"目标函数值: {self.objective:.6f}"
                    if self.objective
                    else "目标函数值: N/A"
                )
                lines.append(f"运行时间: {self.time:.2f}s")
                lines.append(f"使用超胞: {self.supercell_used}")
                lines.append(f"无序类型: {self.disorder_type}")

                if self.analysis_details and "supercell_info" in self.analysis_details:
                    info = self.analysis_details["supercell_info"]
                    lines.append("")
                    lines.append("超胞计算信息:")
                    lines.append(f"  推荐形状: {info.get('supercell')}")
                    lines.append(f"  规模: {info.get('size')} 个原胞")
                    lines.append(f"  RSS: {info.get('rss')}")
                    lines.append(f"  最大误差: {info.get('max_error')}")
                    lines.append(f"  候选数: {info.get('num_candidates')}")

                write_text_file(filename, "\n".join(lines))

            return True
        except Exception as e:
            _logger.exception("Save report failed: %s", e)
            return False

    def __str__(self) -> str:
        """字符串表示"""
        if self.success:
            return (
                f"SQSResult(success=True, "
                f"objective={self.objective:.6f}, "
                f"time={self.time:.2f}s, "
                f"disorder={self.disorder_type})"
            )
        else:
            return f"SQSResult(success=False, error={self.error})"


@dataclass
class DisorderAnalysisResult:
    """CIF 无序分析结果（不进行优化）。"""

    success: bool
    disorder_type: str = "ordered"
    disorder_types: List[str] | None = None
    site_results: List[Dict[str, Any]] | None = None
    warnings: List[str] | None = None
    error: Optional[str] = None

    @property
    def num_sites(self) -> int:
        """位点总数"""
        return len(self.site_results or [])

    @property
    def num_disordered_sites(self) -> int:
        """无序位点数"""
        return sum(
            1 for r in (self.site_results or []) if r.get("site_type") != "Ordered"
        )

    def __str__(self) -> str:
        if self.success:
            types = self.disorder_type
            n_sites = len(self.site_results or [])
            return f"DisorderAnalysisResult(success=True, disorder_type={types}, sites={n_sites})"
        return f"DisorderAnalysisResult(success=False, error={self.error})"


def analyze_cif_disorder(
    input_file: Union[str, Path],
    options: Optional[DisorderAnalysisOptions] = None,
) -> DisorderAnalysisResult:
    """分析 CIF 的无序类型（SD/PD/SPD），但不进行 SQS 优化。

    Args:
        input_file: CIF 文件路径
        options: 配置选项对象

    Returns:
        DisorderAnalysisResult
    """
    opts = options or DisorderAnalysisOptions()
    try:
        raw = cif_disorder_analyzer.analyze_cif_disorder(
            input_file, group_by=opts.group_by, tol=opts.tol
        )
        return DisorderAnalysisResult(
            success=bool(raw.get("success")),
            disorder_type=str(raw.get("disorder_type", "ordered")),
            disorder_types=list(raw.get("disorder_types") or []),
            site_results=list(raw.get("site_results") or []),
            warnings=list(raw.get("warnings") or []),
            error=raw.get("error"),
        )
    except Exception as e:
        return DisorderAnalysisResult(success=False, error=str(e))


def generate_sqs(
    input_file: Union[str, Path, Structure],
    options: Optional[SQSOptions] = None,
) -> SQSResult:
    """
    生成SQS结构

    这是主要的公开API。自动分析无序类型，选择合适的方法，
    生成优化的SQS结构。

    Args:
        input_file: 输入CIF文件路径或Structure对象
        options: 配置选项对象

    Returns:
        SQSResult 对象，包含以下属性：
        - success (bool): 是否成功
        - error (Optional[str]): 错误信息（失败时）
        - structure (Optional[Structure]): 优化后的结构（成功时）
        - objective (Optional[float]): 目标函数值（成功时）
        - time (float): 总运行时间
        - supercell_used (Optional[Tuple]): 使用的超胞尺寸
        - disorder_type (Optional[str]): 无序类型字符串

    Example:
        >>> # 自动计算超胞
        >>> result = generate_sqs("demo_sd.cif")
        >>> if result.success:
        ...     print(result.objective)
        ...     result.structure.to(filename="output.cif")

        >>> # 指定超胞和输出
        >>> result = generate_sqs("demo_sd.cif",
        ...                       options=SQSOptions(supercell=(4,3,2), output_file="output.cif"))
    """
    try:
        # 转换路径
        if isinstance(input_file, (str, Path)):
            input_path: str | Structure = str(input_file)
        else:
            input_path = input_file

        # 使用 options
        opts = options or SQSOptions()

        # 解析 supercell：auto/缺省 => 模型 top-1，异常时回退 RSS 最小且 size 最小
        resolved_supercell = _as_supercell_tuple(opts.supercell)
        if _is_auto_supercell(opts.supercell):
            try:
                if isinstance(input_path, str):
                    resolved_supercell = _resolve_auto_supercell_for_cif(input_path)
                else:
                    resolved_supercell = _resolve_auto_supercell_for_structure(
                        input_path
                    )
            except Exception as e:
                return SQSResult(
                    success=False, error=f"Auto supercell resolution failed: {e}"
                )
        elif resolved_supercell is None:
            return SQSResult(
                success=False,
                error=(
                    "Invalid supercell option. Expected (l,w,h), [l,w,h], "
                    "or auto/None."
                ),
            )

        # 仅在用户要求输出时挂载控制台 handler；默认静默。
        ensure_console_handler(
            _logger,
            enabled=bool(opts.console_log or opts.debug),
            level=10 if opts.debug else 20,
        )

        # 调用核心生成函数（返回 dict）
        raw_result = sqs_orchestrator.generate_sqs(
            input_structure=input_path,
            supercell=resolved_supercell,
            iterations=opts.iterations,
            verbose=opts.console_log,
            debug=opts.debug,
            return_all=opts.return_all,
            log_file=opts.log_file,
            include_details=opts.include_analysis_details,
        )

        # 自动保存
        if raw_result.get("success") and opts.output_file:
            try:
                ensure_parent_dir(opts.output_file)

                structure = raw_result.get("structure")
                if structure is not None:
                    structure.to(filename=opts.output_file)
                    if opts.console_log:
                        _logger.info("Saved to: %s", opts.output_file)
            except Exception as e:
                # 不改变成功与否，但补充错误信息便于定位
                raw_result["save_error"] = str(e)

        # 返回 SQSResult 对象
        return SQSResult(
            success=raw_result.get("success", False),
            structure=raw_result.get("structure"),
            objective=raw_result.get("objective"),
            time=raw_result.get("time", 0.0),
            supercell_used=raw_result.get("supercell_used"),
            disorder_types=raw_result.get("disorder_types"),
            error=raw_result.get("error"),
            analysis_details=raw_result.get("analysis_details"),
            log_messages=raw_result.get("log_messages"),
            analysis=raw_result.get("analysis"),
            config=raw_result.get("config"),
            raw_result=raw_result.get("raw_result"),
        )

    except Exception as e:
        _logger.exception("Unexpected error in generate_sqs: %s", e)
        return SQSResult(
            success=False,
            error=f"Unexpected error: {str(e)}",
            structure=None,
            objective=None,
            time=0.0,
            supercell_used=None,
            disorder_types=None,
        )


def _resolve_output_profile(
    output_profile: Optional[str],
    save_json: bool,
    save_txt_report: bool,
) -> Tuple[bool, bool, bool]:
    return resolve_output_profile(
        output_profile=output_profile,
        save_json=save_json,
        save_txt_report=save_txt_report,
    )


def generate_sqs_with_report(
    input_file: Union[str, Path],
    output_dir: Union[str, Path] = "output",
    output_folder_name: Optional[str] = None,
    output_prefix: Optional[str] = None,
    output_profile: Optional[str] = None,
    supercell: Optional[Tuple[int, int, int]] = None,
    iterations: int = DEFAULT_ITERATIONS,
    console_log: bool = True,
    save_json: bool = True,
    save_txt_report: bool = True,
) -> SQSResult:
    """
    便捷函数：自动生成 CIF + JSON日志 + TXT报告

    默认会按输入文件分类输出目录，自动创建：
    - {output_dir}/{stem}__{supercell}/{stem}__{supercell}.cif
    - {output_dir}/{stem}__{supercell}/{stem}__{supercell}.json
    - {output_dir}/{stem}__{supercell}/{stem}__{supercell}_report.txt

    Args:
        input_file: 输入CIF文件路径
        output_dir: 输出根目录
        output_folder_name: 输出子目录名称，可用 {stem}、{supercell} 作为模板
        output_prefix: 兼容旧版输出前缀（不含扩展名），传入则覆盖 output_dir/output_folder_name
        output_profile: 输出配置档（silent/cif/logs/full），不传则使用 save_json/save_txt_report
        supercell: 超胞尺寸，None则自动计算
        iterations: 优化迭代次数
        console_log: 是否在控制台显示日志（默认True）
        save_json: 是否保存JSON日志文件（默认True）
        save_txt_report: 是否保存文本报告（默认True）

    Returns:
        SQSResult 对象

    Example:
        >>> # 全输出模式（默认）
        >>> result = generate_sqs_with_report("input.cif", output_dir="output")
        >>>
        >>> # 静默模式，只要CIF
        >>> result = generate_sqs_with_report(
        ...     "input.cif",
        ...     output_dir="output",
        ...     output_folder_name="silent_{stem}__{supercell}",
        ...     output_profile="cif",
        ...     console_log=False,
        ... )
        >>>
        >>> # 只要JSON日志，不要文本报告
        >>> result = generate_sqs_with_report(
        ...     "input.cif",
        ...     output_dir="output",
        ...     output_profile="logs"
        ... )
    """
    write_cif, resolved_save_json, resolved_save_txt = _resolve_output_profile(
        output_profile, save_json, save_txt_report
    )
    resolved_prefix = None
    if output_prefix:
        resolved_prefix = Path(output_prefix)
    elif write_cif or resolved_save_json or resolved_save_txt:
        resolved_prefix = build_output_prefix(
            input_file,
            output_dir=output_dir,
            output_folder_name=output_folder_name,
            supercell=supercell,
        )
    result = generate_sqs(
        input_file,
        options=SQSOptions(
            supercell=supercell,
            iterations=iterations,
            output_file=(
                f"{resolved_prefix}.cif" if (write_cif and resolved_prefix) else None
            ),
            log_file=(
                f"{resolved_prefix}.json"
                if (resolved_save_json and resolved_prefix)
                else None
            ),
            include_analysis_details=resolved_save_json
            or resolved_save_txt,  # 需要详细信息时才包含
            console_log=console_log,
        ),
    )

    if result.success and resolved_save_txt and resolved_prefix:
        # 保存文本报告
        result.save_report(f"{resolved_prefix}_report.txt", format="txt")

    return result


def dry_run_recommend_supercell(
    input_file: Union[str, Path],
    max_error: float = 0.0005,
    verbose: bool = False,
) -> Dict[str, Any]:
    """公开 dry-run 接口：仅分析并返回合法超胞候选，不执行 SQS。"""
    return analyze_cif_and_recommend_supercell(
        cif_file=input_file,
        max_error=max_error,
        verbose=verbose,
    )


def batch_recommend_supercells(
    input_files: List[Union[str, Path]],
    max_error: float = 0.0005,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """公开批量 dry-run 接口：批量分析并返回合法超胞候选。"""
    return batch_analyze_cifs(
        cif_files=input_files,
        max_error=max_error,
        verbose=verbose,
    )


def recommend_supercells_with_model(
    input_file: Union[str, Path],
    top_k: Optional[int] = None,
    max_error: float = 0.0005,
    model_path: Optional[Union[str, Path]] = None,
    model_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """公开模型推理接口：解析 CIF 后对合法超胞候选排序并返回 top-k。"""
    return recommend_supercells_by_model(
        cif_file=input_file,
        top_k=top_k,
        max_error=max_error,
        model_path=model_path,
        model_config_path=model_config_path,
    )


def _run_topk_candidate_sqs(
    resolved_input: str,
    evaluated: List[TopkRow],
    policy: str,
    output_root: Path,
    iterations: int,
    console_log: bool,
) -> Tuple[List[TopkRow], List[TopkArtifacts]]:
    rows: List[TopkRow] = []
    exported_all: List[TopkArtifacts] = []

    for row in evaluated:
        rank = int(row["rank"])
        l, w, h = tuple(int(x) for x in row["supercell"])

        output_file = None
        log_file = None
        include_details = False
        if policy == "all":
            prefix = build_topk_candidate_prefix(
                output_root,
                rank=rank,
                supercell=(l, w, h),
            )
            output_file = str(prefix.with_suffix(".cif"))
            log_file = str(prefix.with_suffix(".json"))
            include_details = True

        sqs_result = generate_sqs(
            resolved_input,
            options=SQSOptions(
                supercell=(l, w, h),
                iterations=iterations,
                output_file=output_file,
                log_file=log_file,
                include_analysis_details=include_details,
                console_log=console_log,
            ),
        )

        row_item: TopkRow = {
            "model_rank": rank,
            "supercell": (l, w, h),
            "size": int(row.get("size", l * w * h)),
            "rss": row.get("rss"),
            "max_error": row.get("max_error"),
            "model_score": float(row["model_score"]),
            "success": bool(sqs_result.success),
            "objective": sqs_result.objective,
            "actual_rank": None,
        }

        if output_file and log_file:
            config_file = str(derive_sqsgenerator_config_path(log_file))
            row_item["artifacts"] = {
                "cif": output_file,
                "log_json": log_file,
                "sqsgenerator_config_json": config_file,
            }
            exported_all.append(row_item["artifacts"])

        rows.append(row_item)

    return rows, exported_all


def _assign_actual_ranks(
    rows: List[TopkRow],
    objective_quantum: float,
) -> Tuple[List[TopkRow], Callable[[float], float]]:
    success_rows = [r for r in rows if r["success"] and r["objective"] is not None]

    quantum = Decimal(str(float(objective_quantum)))

    def quantize(value: float) -> float:
        return float(
            Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)
        )

    for item in success_rows:
        item["objective_q"] = quantize(float(item["objective"]))

    success_rows.sort(key=lambda item: (item["objective_q"], float(item["objective"])))

    processed_count = 0
    current_rank = 0
    last_q: Optional[float] = None
    for item in success_rows:
        processed_count += 1
        q = float(item["objective_q"])
        if last_q is None or q != last_q:
            current_rank = processed_count
            last_q = q
        item["actual_rank"] = current_rank

    return success_rows, quantize


def _write_topk_comparison_csv(
    rows: List[TopkRow],
    output_root: Path,
    comparison_csv_name: str,
    quantize: Callable[[float], float],
) -> Path:
    csv_path = output_root / comparison_csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "actual_rank",
                "model_rank",
                "success",
                "supercell_l",
                "supercell_w",
                "supercell_h",
                "size",
                "rss",
                "model_score",
                "objective_q",
                "objective_raw",
            ]
        )
        for item in sorted(rows, key=lambda x: int(x["model_rank"])):
            l, w, h = item["supercell"]
            objective_raw = item.get("objective")
            objective_q = (
                quantize(float(objective_raw)) if objective_raw is not None else None
            )
            writer.writerow(
                [
                    item.get("actual_rank") or "",
                    item["model_rank"],
                    item["success"],
                    l,
                    w,
                    h,
                    item.get("size"),
                    item.get("rss"),
                    item.get("model_score"),
                    objective_q,
                    objective_raw,
                ]
            )
    return csv_path


def _export_topk_best_artifacts(
    resolved_input: str,
    best: TopkRow,
    output_root: Path,
    iterations: int,
    console_log: bool,
) -> Dict[str, Any]:
    best_l, best_w, best_h = best["supercell"]
    best_artifacts = build_topk_best_artifact_paths(output_root)
    best_cif = best_artifacts["cif"]
    best_log = best_artifacts["log_json"]
    best_cfg = best_artifacts["sqsgenerator_config_json"]

    export_result = generate_sqs(
        resolved_input,
        options=SQSOptions(
            supercell=(best_l, best_w, best_h),
            iterations=iterations,
            output_file=str(best_cif),
            log_file=str(best_log),
            include_analysis_details=True,
            console_log=console_log,
        ),
    )

    return {
        "cif": str(best_cif),
        "log_json": str(best_log),
        "sqsgenerator_config_json": str(best_cfg),
        "export_success": bool(export_result.success),
        "export_objective": export_result.objective,
    }


def _build_topk_failed_result(
    resolved_input: str,
    ranked: Dict[str, Any],
    requested_top_k: int,
    policy: str,
) -> Dict[str, Any]:
    return {
        "success": False,
        "message": ranked.get("message", "No ranked candidates"),
        "input_file": resolved_input,
        "model_result": ranked,
        "requested_top_k": requested_top_k,
        "evaluated_top_k": 0,
        "rows": [],
        "best": None,
        "artifact_policy": policy,
        "exported": {},
    }


def _init_topk_exported(
    output_root: Path,
    policy: str,
    exported_all: List[TopkArtifacts],
) -> Dict[str, Any]:
    return {
        "output_dir": str(output_root),
        "comparison_csv": None,
        "best": None,
        "all": exported_all if policy == "all" else [],
    }


def _build_topk_success_result(
    resolved_input: str,
    ranked: Dict[str, Any],
    requested_top_k: int,
    evaluated_count: int,
    policy: str,
    objective_quantum: float,
    rows: List[TopkRow],
    best: Optional[TopkRow],
    exported: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "success": True,
        "message": "Top-k model recommendation and SQS execution completed",
        "input_file": resolved_input,
        "model_result": ranked,
        "requested_top_k": requested_top_k,
        "evaluated_top_k": evaluated_count,
        "artifact_policy": policy,
        "objective_quantum": float(objective_quantum),
        "rows": rows,
        "best": best,
        "exported": exported,
    }


def run_topk_sqs_with_model(
    input_file: Union[str, Path],
    top_k: int = 5,
    max_error: float = 0.0005,
    iterations: int = DEFAULT_ITERATIONS,
    model_path: Optional[Union[str, Path]] = None,
    model_config_path: Optional[Union[str, Path]] = None,
    artifact_policy: str = "best",
    output_dir: Union[str, Path] = "data/output",
    output_folder_name: Optional[str] = None,
    save_comparison_csv: bool = False,
    comparison_csv_name: str = "topn_model_vs_actual.csv",
    objective_quantum: float = 0.1,
    console_log: bool = False,
) -> Dict[str, Any]:
    """单文件 top-k 推荐并实跑。

    行为：
    1) 先在最小 RSS 候选集合内做模型 top-k 推荐；
    2) 对 top-k 逐个执行 SQS；
    3) 产物保留策略由 artifact_policy 控制：
       - none: 不落地候选文件
       - best: 仅落地真实最优候选（CIF + JSON日志 + _sqsgenerator.json）
       - all: 逐个落地 top-k 全部候选文件
    4) save_comparison_csv=True 且 top_k>1 时导出对比 CSV。
    """
    resolved_input = str(input_file)
    policy = normalize_artifact_policy(artifact_policy, default="best")

    requested_top_k = max(1, int(top_k))
    ranked = recommend_supercells_by_model(
        cif_file=resolved_input,
        top_k=requested_top_k,
        max_error=max_error,
        model_path=model_path,
        model_config_path=model_config_path,
    )

    ranked_candidates = list(ranked.get("ranked_candidates") or [])
    if not ranked.get("success") or not ranked_candidates:
        return _build_topk_failed_result(
            resolved_input=resolved_input,
            ranked=ranked,
            requested_top_k=requested_top_k,
            policy=policy,
        )

    evaluated = ranked_candidates[:requested_top_k]
    output_root = build_topk_output_root(
        resolved_input,
        output_dir=output_dir,
        output_folder_name=output_folder_name,
        top_k=requested_top_k,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows, exported_all = _run_topk_candidate_sqs(
        resolved_input=resolved_input,
        evaluated=evaluated,
        policy=policy,
        output_root=output_root,
        iterations=iterations,
        console_log=console_log,
    )

    success_rows, quantize = _assign_actual_ranks(
        rows=rows,
        objective_quantum=objective_quantum,
    )

    best = success_rows[0] if success_rows else None
    exported = _init_topk_exported(
        output_root=output_root,
        policy=policy,
        exported_all=exported_all,
    )

    if save_comparison_csv and requested_top_k > 1:
        csv_path = _write_topk_comparison_csv(
            rows=rows,
            output_root=output_root,
            comparison_csv_name=comparison_csv_name,
            quantize=quantize,
        )
        exported["comparison_csv"] = str(csv_path)

    if policy == "best" and best is not None:
        exported["best"] = _export_topk_best_artifacts(
            resolved_input=resolved_input,
            best=best,
            output_root=output_root,
            iterations=iterations,
            console_log=console_log,
        )

    if policy == "all":
        exported["best"] = best["artifacts"] if (best and "artifacts" in best) else None

    return _build_topk_success_result(
        resolved_input=resolved_input,
        ranked=ranked,
        requested_top_k=requested_top_k,
        evaluated_count=len(evaluated),
        policy=policy,
        objective_quantum=objective_quantum,
        rows=rows,
        best=best,
        exported=exported,
    )
