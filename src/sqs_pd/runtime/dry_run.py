"""
Dry-run API：快速分析 CIF 并推荐最优超胞规格
=====================================
功能：
1. 读取 CIF 文件
2. 自动判断无序类型（SD/PD/SPD）
3. 提取占据数
4. 推荐最优超胞规格
5. 输出详细分析报告

用法示例：
    >>> from sqs_pd.runtime.dry_run import analyze_cif_and_recommend_supercell
    >>> result = analyze_cif_and_recommend_supercell("demo_sd.cif")
    >>> print(result['recommended_supercell'])
    (4, 3, 2)
    >>> print(result['disorder_types'])
    ['SD']
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Union

from ..analysis.cif_disorder_analyzer import analyze_cif_disorder
from ..core.supercell_optimizer import get_supercell_info_optimized
from ..batching.batch_common import run_batch_analysis


def extract_disordered_occupancies(
    site_results: List[Dict[str, Any]], min_occ: float = 0.0005, max_occ: float = 0.9995
) -> List[List[float]]:
    """从位点分析结果中提取无序占据数（保留位点结构）

    筛选条件（有序占据被排除）：
    - 跳过空位（元素符号为 "0"）
    - 跳过接近 0 的占据数（occ ≤ min_occ = 0.0005）
    - 跳过接近 1 的占据数（occ ≥ max_occ = 0.9995）
    - 只保留真正无序的占据数（min_occ < occ < max_occ）

    Args:
        site_results: analyze_cif_disorder 返回的 site_results
        min_occ: 最小占据数阈值（≤此值的视为有序，默认 0.0005）
        max_occ: 最大占据数阈值（≥此值的视为有序，默认 0.9995）

    Returns:
        嵌套列表：每个元素是一个位点的占据数列表
        例如: [[0.5, 0.5], [0.3, 0.7]] 表示 2 个位点，第一个位点有 2 种占据

    Example:
        >>> sites = [{'species': {'Fe': 0.5, 'Co': 0.5}}, {'species': {'Ni': 1.0}}]
        >>> extract_disordered_occupancies(sites)
        [[0.5, 0.5]]  # Ni=1.0 被认为是有序，不计入；返回嵌套列表
    """
    occupancies_per_site = []

    for site in site_results:
        species = site.get("species", {})
        site_occs = []

        for element, occ in species.items():
            # 跳过空位
            if element == "0":
                continue

            # 判断是否为有序位点：接近 0 或接近 1 时跳过
            # 有序的定义：occ ≤ min_occ 或 occ ≥ max_occ
            if occ <= min_occ or occ >= max_occ:
                # 这些是有序占据，不计入无序位点的考量
                continue

            # 只保留真正无序的占据数（min_occ < occ < max_occ）
            site_occs.append(float(occ))

        # 只有当位点有无序占据时才加入
        if site_occs:
            occupancies_per_site.append(sorted(site_occs))

    return occupancies_per_site


def analyze_cif_and_recommend_supercell(
    cif_file: Union[str, Path], max_error: float = 0.0005, verbose: bool = False
) -> Dict[str, Any]:
    """分析 CIF 文件并推荐最优超胞规格（主入口函数）

    完整流程：
    1. 读取并解析 CIF
    2. 识别无序类型（SD/PD/SPD）
    3. 提取占据数
    4. 推荐最优超胞规格（最小化 RSS）

    Args:
        cif_file: CIF 文件路径
        max_error: 最大允许误差（默认 0.0005）
        verbose: 是否打印详细信息

    Returns:
        分析结果字典，包含：
        - cif_file: 输入文件路径
        - disorder_types: 无序类型列表 ['SD', 'PD', 'SPD']
        - num_sites: 总位点数
        - num_disordered_sites: 无序位点数
        - occupancies: 提取的占据数列表
        - recommended_supercell: 推荐的超胞形状 (l, w, h)
        - supercell_size: 超胞规模
        - rss: 残差平方和
        - max_error_actual: 实际最大误差
        - optimization_success: 是否找到满足条件的规格
        - site_results: 位点详细信息
        - warnings: 警告信息列表
        - message: 人类可读的描述信息

    Raises:
        FileNotFoundError: 如果 CIF 文件不存在
        ValueError: 如果 CIF 解析失败

    Example:
        >>> result = analyze_cif_and_recommend_supercell("demo_sd.cif")
        >>> print(f"推荐超胞: {result['recommended_supercell']}")
        推荐超胞: (4, 3, 2)
        >>> print(f"无序类型: {result['disorder_types']}")
        无序类型: ['SD']
    """
    cif_path = Path(cif_file)

    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    if verbose:
        print(f"Analyzing CIF: {cif_path}")
        print("-" * 60)

    # 步骤1：解析 CIF 并识别无序类型
    cif_analysis = analyze_cif_disorder(cif_path)

    if not cif_analysis["success"]:
        error_msg = cif_analysis.get("error", "Unknown error")
        raise ValueError(f"CIF analysis failed: {error_msg}")

    disorder_types = cif_analysis.get("disorder_types", [])
    site_results = cif_analysis.get("site_results", [])
    num_sites = cif_analysis.get("num_sites", 0)
    num_disordered_sites = cif_analysis.get("num_disordered_sites", 0)
    warnings = cif_analysis.get("warnings", [])

    if verbose:
        print(f"📊 无序分析结果:")
        print(f"  - 总位点数: {num_sites}")
        print(f"  - 无序位点数: {num_disordered_sites}")
        if disorder_types:
            print(f"  - 无序类型: {', '.join(disorder_types)}")
        else:
            print(f"  - 无序类型: ordered（完全有序）")
        print()

    # 步骤2：提取占据数
    occupancies = extract_disordered_occupancies(site_results)

    if verbose:
        print(f"🔢 提取的占据数:")
        if occupancies:
            print(f"  {occupancies}")
        else:
            print(f"  （无无序占据）")
        print()

    # 步骤3：推荐超胞规格
    if not occupancies:
        # 完全有序结构，不需要特殊超胞
        result = {
            "cif_file": str(cif_path),
            "disorder_types": disorder_types,
            "num_sites": num_sites,
            "num_disordered_sites": num_disordered_sites,
            "occupancies": occupancies,
            "recommended_supercell": None,
            "supercell_size": None,
            "rss": 0.0,
            "max_error_actual": 0.0,
            "optimization_success": True,
            "all_candidates": [],
            "num_candidates": 0,
            "site_results": site_results,
            "warnings": warnings,
            "message": "Ordered structure - no special supercell required",
        }

        if verbose:
            print("✅ 完全有序结构，无需特殊超胞规格")

        return result

    # 优化超胞规格：获取所有满足条件的候选规格
    optimization_result = get_supercell_info_optimized(occupancies, max_error=max_error)

    success = optimization_result["success"]
    recommended_supercell = optimization_result.get("supercell")  # 最优规格
    supercell_size = optimization_result.get("size")  # 最优规格的规模
    rss = optimization_result.get("rss")  # 最优规格的 RSS
    max_error_actual = optimization_result.get("max_error")  # 最优规格的最大误差
    num_candidates = optimization_result["num_candidates"]
    all_candidates = optimization_result.get("all_candidates", [])
    opt_message = optimization_result["message"]

    if verbose:
        print(f"🎯 超胞优化结果:")
        if success:
            print(f"  ✅ 成功")
            print(
                f"  - 最优超胞: {recommended_supercell[0]} × {recommended_supercell[1]} × {recommended_supercell[2]}"
            )
            print(f"  - 超胞规模: {supercell_size} 个原胞")
            print(f"  - 残差平方和 (RSS): {rss:.6e}")
            print(f"  - 最大误差: {max_error_actual:.6f}")
            print(f"  - 满足条件的规格数量: {num_candidates}")

            if num_candidates > 1 and num_candidates <= 10:
                print(f"\n  所有满足条件的规格 (按 RSS 升序):")
                for i, cand in enumerate(all_candidates[:10], 1):
                    supercell_tuple = cand["supercell"]
                    print(
                        f"    [{i}] {supercell_tuple[0]}×{supercell_tuple[1]}×{supercell_tuple[2]} "
                        f"(size={cand['size']}, RSS={cand['rss']:.6e}, max_err={cand['max_error']:.6f})"
                    )
                if num_candidates > 10:
                    print(f"    ... 还有 {num_candidates - 10} 个规格 ...")
        else:
            print(f"  ❌ 失败")
            print(f"  - 原因: {opt_message}")
        print()

    # 构建结果
    result = {
        "cif_file": str(cif_path),
        "disorder_types": disorder_types,
        "num_sites": num_sites,
        "num_disordered_sites": num_disordered_sites,
        "occupancies": occupancies,
        "recommended_supercell": recommended_supercell,
        "supercell_size": supercell_size,
        "rss": rss,
        "max_error_actual": max_error_actual,
        "optimization_success": success,
        "num_candidates": num_candidates,
        "all_candidates": all_candidates,
        "site_results": site_results,
        "warnings": warnings,
        "message": opt_message,
        "errors": optimization_result.get("errors", []),
        "nearest_fractions": optimization_result.get("nearest_fractions", []),
    }

    return result


def batch_analyze_cifs(
    cif_files: List[Union[str, Path]], max_error: float = 0.0005, verbose: bool = False
) -> List[Dict[str, Any]]:
    """批量分析多个 CIF 文件

    Args:
        cif_files: CIF 文件路径列表
        max_error: 最大允许误差
        verbose: 是否打印详细信息

    Returns:
        结果列表，每个元素对应一个 CIF 的分析结果

    Example:
        >>> results = batch_analyze_cifs(["demo_sd.cif", "demo_pd.cif"])
        >>> for r in results:
        ...     print(f"{r['cif_file']}: {r['recommended_supercell']}")
    """
    return run_batch_analysis(
        cif_files=cif_files,
        analyze_single=analyze_cif_and_recommend_supercell,
        max_error=max_error,
        verbose=verbose,
    )


def print_analysis_summary(
    result: Dict[str, Any], show_all_candidates: bool = True
) -> None:
    """打印分析结果摘要（人类可读格式）

    Args:
        result: analyze_cif_and_recommend_supercell 返回的结果
        show_all_candidates: 是否显示所有满足条件的候选规格

    Example:
        >>> result = analyze_cif_and_recommend_supercell("demo.cif")
        >>> print_analysis_summary(result)
    """
    print(format_analysis_summary(result, show_all_candidates=show_all_candidates))


def format_analysis_summary(
    result: Dict[str, Any], show_all_candidates: bool = True
) -> str:
    """格式化分析结果摘要（人类可读格式）。"""
    lines: List[str] = []
    lines.append("\n" + "=" * 60)
    lines.append("CIF 分析与超胞推荐摘要")
    lines.append("=" * 60)

    lines.append(f"📁 文件: {result['cif_file']}")
    lines.append(
        f"📊 位点: {result['num_disordered_sites']}/{result['num_sites']} 无序"
    )

    disorder_types = result.get("disorder_types", [])
    if disorder_types:
        lines.append(f"🔀 无序类型: {', '.join(disorder_types)}")
    else:
        lines.append("🔀 无序类型: ordered")

    occupancies = result.get("occupancies", [])
    if occupancies:
        lines.append(f"🔢 占据数: {occupancies}")

    if result["optimization_success"]:
        supercell = result["recommended_supercell"]
        if supercell:
            lines.append(
                f"✅ 最优超胞: {supercell[0]} × {supercell[1]} × {supercell[2]}"
            )
            lines.append(f"   规模: {result['supercell_size']} 个原胞")
            lines.append(f"   RSS: {result['rss']:.6e}")
            lines.append(f"   最大误差: {result['max_error_actual']:.6f}")

            all_candidates = result.get("all_candidates", [])
            num_candidates = result.get("num_candidates", 0)

            if show_all_candidates and num_candidates > 1:
                lines.append(f"\n📋 所有满足条件的规格 ({num_candidates} 个):")
                for i, cand in enumerate(all_candidates, 1):
                    if i > 20:
                        lines.append(f"   ... 还有 {num_candidates - 20} 个规格 ...")
                        break
                    supercell_tuple = cand["supercell"]
                    size = cand["size"]
                    rss = cand["rss"]
                    max_err = cand["max_error"]
                    is_optimal = "⭐" if i == 1 else "  "
                    lines.append(
                        f"   {is_optimal} [{i:2d}] {supercell_tuple[0]:2d}×{supercell_tuple[1]:2d}×{supercell_tuple[2]:2d} "
                        f"size={size:4d} RSS={rss:.6e} max_err={max_err:.6f}"
                    )
        else:
            lines.append("✅ 完全有序结构")
    else:
        lines.append("❌ 未找到满足条件的超胞规格")
        lines.append(f"   原因: {result['message']}")

    warnings = result.get("warnings", [])
    if warnings:
        lines.append("\n⚠️  警告:")
        for w in warnings:
            lines.append(f"   - {w}")

    lines.append("=" * 60 + "\n")
    return "\n".join(lines)


def format_batch_analysis_summary(results: List[Dict[str, Any]]) -> str:
    """格式化批量分析摘要（简版）。"""
    lines: List[str] = []
    lines.append("\n" + "=" * 70)
    lines.append(f"批量分析完成 ({len(results)} 个文件)")
    lines.append("=" * 70)

    for r in results:
        filename = Path(r["cif_file"]).name
        if r.get("optimization_success"):
            supercell = r.get("recommended_supercell")
            if supercell:
                lines.append(
                    f"✅ {filename}: {supercell[0]}×{supercell[1]}×{supercell[2]} (size={r['supercell_size']})"
                )
            else:
                lines.append(f"✅ {filename}: ordered")
        else:
            lines.append(f"❌ {filename}: {r.get('message', 'failed')}")

    lines.append("=" * 70 + "\n")
    return "\n".join(lines)
