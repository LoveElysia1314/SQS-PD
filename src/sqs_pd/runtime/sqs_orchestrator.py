"""
SQS 协调器
=====================================
功能：
1. 协调完整的 SQS 生成流程
2. 结构分析 → 配置构建 → 优化 → 结果处理
3. 错误处理和日志记录
4. 结果验证和统计

用法示例：
    >>> from sqs_orchestrator import generate_sqs
    >>> from pymatgen.core import Structure
    >>>
    >>> structure = Structure.from_file("demo_sd.cif")
    >>> result = generate_sqs(structure, supercell=(4,3,2))
    >>> print(result['success'])
    True
    >>> optimized = result['structure']
"""

from typing import Dict, List, Tuple, Any, Optional
from pymatgen.core import Structure, Lattice
import sqsgenerator
import time
import numpy as np

from ..foundation.constants import DEFAULT_ITERATIONS
from .logging_utils import get_logger
from .io_utils import (
    ensure_parent_dir,
    write_json_file,
    derive_sqsgenerator_config_path,
)

_logger = get_logger()


def prepare_structure(
    cif_file: str, verbose: bool = False
) -> Tuple[Structure, Dict[str, Any]]:
    """
    准备结构并分析无序类型

    Args:
        cif_file: CIF 文件路径
        verbose: 是否打印详细信息

    Returns:
        (Structure 对象, 分析结果)

    Example:
        >>> structure, analysis = prepare_structure("demo_sd.cif")
        >>> print(analysis['disorder_type'])
        'chemical'
    """
    from ..analysis.disorder_analyzer import analyze_structure

    if verbose:
        _logger.info("Loading structure from: %s", cif_file)

    # 尝试显式以 UTF-8 编码读取并使用 CifParser 解析，避免 pymatgen 内部未指定 encoding 的警告
    structure = None
    import warnings

    with warnings.catch_warnings():
        # 忽略 pymatgen 内部关于 encoding 的提示以及隐式 mode 的 FutureWarning
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module="pymatgen.io.cif"
        )
        warnings.filterwarnings("ignore", message=".*explicit `encoding`.*")

        try:
            from pymatgen.io.cif import CifParser

            with open(cif_file, "r", encoding="utf-8", errors="replace") as f:
                cif_text = f.read()

            parser = CifParser.from_string(cif_text)
            structures = parser.get_structures(primitive=False)
            if structures:
                structure = structures[0]
            else:
                raise ValueError("CifParser returned no structures")
        except Exception:
            # 回退到 pymatgen 的默认加载方式（保持兼容性）
            if verbose:
                _logger.warning(
                    "Explicit CifParser parsing failed, falling back to Structure.from_file"
                )
            structure = Structure.from_file(cif_file)

    if verbose:
        _logger.info("Analyzing disorder type...")

    analysis = analyze_structure(structure, verbose=verbose)

    return structure, analysis


def build_config(
    structure: Structure,
    analysis: Dict[str, Any],
    supercell: Optional[Tuple[int, int, int]] = None,
    iterations: int = DEFAULT_ITERATIONS,
    verbose: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    构建 sqsgenerator 配置

    Args:
        structure: pymatgen Structure 对象
        analysis: 无序分析结果
        supercell: 超胞尺寸，None 则自动计算
        iterations: 迭代次数
        verbose: 是否打印详细信息

    Returns:
        sqsgenerator 配置字典

    Example:
        >>> config = build_config(structure, analysis)
        >>> print(config['iterations'])
        1000000
    """
    from ..core.config_builder import build_sqs_config, validate_config, get_config_info

    if verbose:
        _logger.info("Building configuration...")
        if supercell:
            _logger.info("  Using supercell: %s", supercell)
        else:
            _logger.info("  Auto-calculating supercell size...")

    config = build_sqs_config(
        structure,
        analysis,
        supercell=supercell,
        iterations=iterations,
        debug=debug,
    )

    # 验证配置
    is_valid, errors = validate_config(config)

    if not is_valid:
        raise ValueError(f"Invalid configuration: {errors}")

    if verbose:
        info = get_config_info(config)
        _logger.info("  Mode: %s", info["mode"])
        _logger.info("  Supercell: %s", info["supercell"])
        _logger.info("  Sublattices: %s", info["num_sublattices"])

    return config


def run_optimization(config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    运行 sqsgenerator 优化

    Args:
        config: sqsgenerator 配置字典
        verbose: 是否打印详细信息

    Returns:
        优化结果字典，包含：
        - success: 是否成功
        - result: sqsgenerator 结果对象（如果成功）
        - error: 错误信息（如果失败）
        - time: 运行时间（秒）

    Example:
        >>> result = run_optimization(config)
        >>> if result['success']:
        ...     print(result['result'].objective)
    """
    if verbose:
        _logger.info("Running SQS optimization...")
        _logger.info("  Iterations: %s", config.get("iterations", "N/A"))

    start_time = time.time()

    try:
        # sqsgenerator 对未知顶层字段会报错，确保配置干净
        clean_config = dict(config)

        # 解析配置
        parsed_config = sqsgenerator.parse_config(clean_config)

        # 检查解析是否成功（ParseError 是特殊类型，且其 __str__ 可能不可读）
        if "ParseError" in str(type(parsed_config)):
            msg = getattr(parsed_config, "msg", None)
            key = getattr(parsed_config, "key", None)
            code = getattr(parsed_config, "code", None)
            parameter = getattr(parsed_config, "parameter", None)
            detail = f"code={code}, key={key}, parameter={parameter}, msg={msg}"
            return {
                "success": False,
                "error": f"Config parse error: {detail}",
                "time": time.time() - start_time,
            }

        # 运行优化（不使用 log_level 参数）
        result = sqsgenerator.optimize(clean_config)

        elapsed_time = time.time() - start_time

        if result is None:
            return {
                "success": False,
                "error": "Optimization returned None",
                "time": elapsed_time,
            }

        # sqsgenerator 结果对象使用 best() 方法获取最佳结果
        if not hasattr(result, "best"):
            return {
                "success": False,
                "error": "Result object has no 'best' method",
                "time": elapsed_time,
            }

        best_result = result.best()
        objective_value = best_result.objective

        if verbose:
            _logger.info("  Optimization complete!")
            _logger.info("  Objective: %.6f", objective_value)
            _logger.info("  Time: %.2fs", elapsed_time)

        return {
            "success": True,
            "result": best_result,
            "objective": objective_value,
            "time": elapsed_time,
        }

    except Exception as e:
        elapsed_time = time.time() - start_time

        # 兼容：部分 sqsgenerator 错误可能以 ParseError 对象/类似对象形式出现
        extra = ""
        if hasattr(e, "code") or hasattr(e, "msg") or hasattr(e, "key"):
            code = getattr(e, "code", None)
            key = getattr(e, "key", None)
            parameter = getattr(e, "parameter", None)
            msg = getattr(e, "msg", None)
            extra = f" (code={code}, key={key}, parameter={parameter}, msg={msg})"

        if verbose:
            _logger.error("  Optimization failed: %s%s", str(e), extra)

        return {"success": False, "error": f"{str(e)}{extra}", "time": elapsed_time}


def post_process_result(
    result_dict: Dict[str, Any],
    config: Dict[str, Any],
    verbose: bool = False,
) -> Optional[Structure]:
    """
    后处理优化结果，重建 Structure

    Args:
        result_dict: run_optimization() 返回的结果
        config: sqsgenerator 配置字典
        verbose: 是否打印详细信息

    Returns:
        优化后的 Structure 对象，失败返回 None

    Example:
        >>> optimized = post_process_result(result, config)
        >>> if optimized:
        ...     optimized.to(filename="output.cif")
    """
    if not result_dict["success"]:
        if verbose:
            _logger.error("Cannot post-process failed result")
        return None

    sqs_result = result_dict["result"]

    try:
        # 提取优化后的物种列表（原子序数）
        species_list = sqs_result.species

        # 重建超胞结构
        nx, ny, nz = config["structure"]["supercell"]
        lattice_matrix = np.array(config["structure"]["lattice"])

        # 构建超胞晶格（对角线扩展）
        supercell_lattice_matrix = lattice_matrix.copy()
        supercell_lattice_matrix[0] *= nx
        supercell_lattice_matrix[1] *= ny
        supercell_lattice_matrix[2] *= nz
        supercell_lattice = Lattice(supercell_lattice_matrix)

        # 生成超胞坐标
        coords = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    for orig_coord in config["structure"]["coords"]:
                        frac = [
                            (orig_coord[0] + ix) / nx,
                            (orig_coord[1] + iy) / ny,
                            (orig_coord[2] + iz) / nz,
                        ]
                        coords.append(frac)

        # 过滤掉空位（原子序数为0）
        # sqsgenerator用0表示空位，但pymatgen无法处理
        filtered_species = []
        filtered_coords = []
        for i, (species, coord) in enumerate(zip(species_list, coords)):
            if species != 0:  # 跳过空位
                filtered_species.append(species)
                filtered_coords.append(coord)

        if len(filtered_species) == 0:
            raise ValueError("All positions are vacancies - cannot create structure")

        # 构建优化后的 Structure（仅包含非空位原子）
        optimized_structure = Structure(
            supercell_lattice, filtered_species, filtered_coords
        )

        if verbose:
            _logger.info("Post-processing complete")
            _logger.info("  Total positions: %s", len(species_list))
            _logger.info("  Vacancies: %s", len(species_list) - len(filtered_species))
            _logger.info("  Atoms: %s", len(filtered_species))
            _logger.info("  Optimized formula: %s", optimized_structure.composition)
            _logger.info("  Lattice abc: %s", optimized_structure.lattice.abc)

        return optimized_structure

    except Exception as e:
        if verbose:
            _logger.exception("Post-processing failed: %s", str(e))
        return None


def generate_sqs(
    input_structure: str | Structure,
    supercell: Optional[Tuple[int, int, int]] = None,
    iterations: int = DEFAULT_ITERATIONS,
    verbose: bool = False,
    debug: bool = False,
    return_all: bool = False,
    log_file: Optional[str] = None,
    include_details: bool = False,
) -> Dict[str, Any]:
    """
    完整的 SQS 生成流程（主入口函数）

    这是整个模块的核心函数，协调所有步骤。

    Args:
        input_structure: CIF 文件路径或 Structure 对象
        supercell: 超胞尺寸，None 则自动计算
        iterations: 迭代次数
        verbose: 是否打印详细信息
        debug: 是否启用调试模式
        return_all: 是否返回所有中间结果
        log_file: JSON 日志文件路径（包含详细配置和中间信息）
        include_details: 是否在结果中包含详细分析信息

    Returns:
        结果字典，包含：
        - success: 是否成功
        - structure: 优化后的 Structure（如果成功）
        - objective: 目标函数值
        - time: 总运行时间
        - analysis: 无序分析结果（如果 return_all=True）
        - config: sqsgenerator 配置（如果 return_all=True）
        - error: 错误信息（如果失败）

    Example:
        >>> result = generate_sqs("demo_sd.cif", supercell=(4,3,2))
        >>> if result['success']:
        ...     result['structure'].to(filename="output.cif")
        ...     print(f"Objective: {result['objective']}")

    Example with auto supercell:
        >>> result = generate_sqs("demo_sd.cif")  # Auto calculate
        >>> print(result['supercell_used'])
        (4, 3, 2)
    """
    total_start = time.time()

    # 初始化日志和详细信息存储
    log_messages = []
    detailed_info = {}

    if verbose:
        _logger.info("%s", "=" * 60)
        _logger.info("SQS Generation Workflow")
        _logger.info("%s", "=" * 60)

    log_messages.append(
        {
            "step": "start",
            "time": time.time() - total_start,
            "message": "SQS generation workflow started",
        }
    )

    try:
        # Step 1: 准备结构
        log_messages.append(
            {
                "step": "prepare_structure",
                "time": time.time() - total_start,
                "message": "Loading and analyzing structure",
            }
        )

        if isinstance(input_structure, str):
            structure, analysis = prepare_structure(input_structure, verbose)
            detailed_info["input_file"] = input_structure
        else:
            from ..analysis.disorder_analyzer import analyze_structure

            structure = input_structure
            analysis = analyze_structure(structure, verbose=verbose)
            detailed_info["input_file"] = "<Structure object>"

        detailed_info["structure_info"] = {
            "formula": str(structure.composition),
            "num_sites": len(structure),
            "lattice_params": {
                "a": float(structure.lattice.a),
                "b": float(structure.lattice.b),
                "c": float(structure.lattice.c),
            },
        }
        detailed_info["disorder_analysis"] = {
            "disorder_types": analysis.get("disorder_types", []),
            "num_site_groups": len(analysis.get("site_groups", [])),
            "occupancies": analysis.get("occupancies", []),
        }

        log_messages.append(
            {
                "step": "prepare_structure",
                "time": time.time() - total_start,
                "message": f"Structure analyzed, disorder types: {analysis.get('disorder_types', [])}",
            }
        )

        # Step 2: 构建配置
        log_messages.append(
            {
                "step": "build_config",
                "time": time.time() - total_start,
                "message": "Building sqsgenerator configuration",
            }
        )

        config = build_config(
            structure,
            analysis,
            supercell,
            iterations,
            verbose,
            debug,
        )

        supercell_used = tuple(config["structure"]["supercell"])

        # 收集超胞计算信息
        if include_details or log_file:
            try:
                from ..core.supercell_optimizer import get_supercell_info_optimized

                supercell_info = get_supercell_info_optimized(
                    analysis.get("occupancies", []),
                )
                detailed_info["supercell_info"] = supercell_info
            except Exception as e:
                detailed_info["supercell_info"] = {"error": str(e)}

        # 保存完整的 sqsgenerator 配置（用于日志记录和调试）
        # 创建配置的深拷贝以避免包含不可序列化的对象
        config_for_log = {
            "iterations": config.get("iterations"),
            "structure": {
                "lattice": config["structure"]["lattice"],
                "coords": config["structure"]["coords"],
                "species": config["structure"]["species"],
                "supercell": config["structure"]["supercell"],
            },
            "composition": config.get("composition", []),
            "sublattice_mode": config.get("sublattice_mode"),
        }

        # 添加可选字段
        if "target_objective" in config:
            config_for_log["target_objective"] = config["target_objective"]
        if "shell_weights" in config:
            config_for_log["shell_weights"] = config["shell_weights"]
        if "pair_weights" in config:
            # pair_weights 可能是 numpy 数组，需要转换
            try:
                import numpy as np

                pw = config["pair_weights"]
                if isinstance(pw, np.ndarray):
                    config_for_log["pair_weights"] = pw.tolist()
                else:
                    config_for_log["pair_weights"] = pw
            except:
                pass

        detailed_info["sqsgenerator_config"] = config_for_log

        log_messages.append(
            {
                "step": "build_config",
                "time": time.time() - total_start,
                "message": f"Configuration built, supercell: {supercell_used}",
            }
        )

        # Step 3: 运行优化
        log_messages.append(
            {
                "step": "run_optimization",
                "time": time.time() - total_start,
                "message": "Starting SQS optimization",
            }
        )

        opt_result = run_optimization(config, verbose)

        log_messages.append(
            {
                "step": "run_optimization",
                "time": time.time() - total_start,
                "message": f"Optimization {'succeeded' if opt_result['success'] else 'failed'}, objective: {opt_result.get('objective', 'N/A')}",
            }
        )

        if not opt_result["success"]:
            log_messages.append(
                {
                    "step": "error",
                    "time": time.time() - total_start,
                    "message": f"Optimization error: {opt_result['error']}",
                }
            )

            result = {
                "success": False,
                "error": opt_result["error"],
                "time": time.time() - total_start,
                "analysis": analysis if return_all else None,
                "config": config if return_all else None,
                "log_messages": log_messages,
                "analysis_details": (
                    detailed_info if (include_details or log_file) else None
                ),
            }

            if log_file:
                _write_json_log(log_file, result, detailed_info, log_messages)

            return result

        # Step 4: 后处理
        log_messages.append(
            {
                "step": "post_process",
                "time": time.time() - total_start,
                "message": "Post-processing optimization result",
            }
        )

        optimized_structure = post_process_result(opt_result, config, verbose)

        if optimized_structure is None:
            log_messages.append(
                {
                    "step": "error",
                    "time": time.time() - total_start,
                    "message": "Post-processing failed",
                }
            )

            result = {
                "success": False,
                "error": "Post-processing failed",
                "time": time.time() - total_start,
                "analysis": analysis if return_all else None,
                "config": config if return_all else None,
                "log_messages": log_messages,
                "analysis_details": (
                    detailed_info if (include_details or log_file) else None
                ),
            }

            if log_file:
                _write_json_log(log_file, result, detailed_info, log_messages)

            return result

        total_time = time.time() - total_start

        log_messages.append(
            {
                "step": "complete",
                "time": total_time,
                "message": f"SQS generation completed successfully, objective: {opt_result['objective']:.6f}",
            }
        )

        if verbose:
            _logger.info("%s", "=" * 60)
            _logger.info("SQS generation complete!")
            _logger.info("  Total time: %.2fs", total_time)
            _logger.info("  Objective: %.6f", opt_result["objective"])
            _logger.info("%s", "=" * 60)

        result = {
            "success": True,
            "structure": optimized_structure,
            "objective": opt_result["objective"],
            "time": total_time,
            "supercell_used": supercell_used,
            "disorder_types": analysis.get("disorder_types", []),
            "log_messages": log_messages,
            "analysis_details": (
                detailed_info if (include_details or log_file) else None
            ),
        }

        if return_all:
            result["analysis"] = analysis
            result["config"] = config
            result["raw_result"] = opt_result["result"]

        # 写入 JSON 日志文件
        if log_file:
            _write_json_log(log_file, result, detailed_info, log_messages)

        return result

    except Exception as e:
        log_messages.append(
            {
                "step": "error",
                "time": time.time() - total_start,
                "message": f"Exception: {str(e)}",
            }
        )

        if verbose:
            _logger.exception("SQS generation failed: %s", str(e))

        result = {
            "success": False,
            "error": str(e),
            "time": time.time() - total_start,
            "log_messages": log_messages,
            "analysis_details": (
                detailed_info if (include_details or log_file) else None
            ),
        }

        if log_file:
            _write_json_log(log_file, result, detailed_info, log_messages)

        return result


def _write_json_log(
    log_file: str,
    result: Dict[str, Any],
    detailed_info: Dict[str, Any],
    log_messages: List[Dict[str, Any]],
) -> None:
    """写入 JSON 格式的详细日志文件"""
    try:
        log_path = ensure_parent_dir(log_file)

        sqs_config = detailed_info.get("sqsgenerator_config")
        sqs_config_file = None
        if sqs_config:
            sqs_config_path = derive_sqsgenerator_config_path(log_path)
            write_json_file(sqs_config_path, sqs_config)
            sqs_config_file = str(sqs_config_path)

        log_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": {
                "success": result.get("success"),
                "objective": result.get("objective"),
                "time": result.get("time"),
                "supercell_used": result.get("supercell_used"),
                "disorder_types": result.get("disorder_types"),
                "error": result.get("error"),
            },
            "sqsgenerator_config_file": sqs_config_file,  # 完整配置单独写入文件
            "input_info": {
                "input_file": detailed_info.get("input_file"),
                "structure_info": detailed_info.get("structure_info"),
                "disorder_analysis": detailed_info.get("disorder_analysis"),
            },
            "supercell_calculation": detailed_info.get("supercell_info"),
            "log_messages": log_messages,
        }

        write_json_file(log_path, log_data)
    except Exception as e:
        _logger.warning("Failed to write JSON log file: %s", e)
