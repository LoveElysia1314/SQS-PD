"""
配置构建器 - Split模式版本
=====================================
功能：
1. 根据无序类型构建 sqsgenerator 配置
2. **全面使用官方支持的 split 模式**
3. 有序位点自动保留，无序位点自动优化
4. 支持自定义超胞尺寸
5. 配置验证

核心改进（V4.0）：
- ✅ 弃用简单模式（会污染有序位点）
- ✅ 弃用手动过滤方法（需要后处理）
- ✅ 全面采用 split 模式（官方推荐）
- ✅ 自动生成正确的 sites 字段

用法示例：
    >>> from config_builder import build_sqs_config
    >>> from disorder_analyzer import analyze_structure
    >>> from pymatgen.core import Structure
    >>>
    >>> structure = Structure.from_file("demo_sd.cif")
    >>> analysis = analyze_structure(structure)
    >>> config = build_sqs_config(structure, analysis, supercell=(4,3,2))
    >>> print(config['sublattice_mode'])
    'split'
"""

from typing import Dict, List, Tuple, Any, Optional
from pymatgen.core import Structure, Element
import math

from ..foundation.constants import DEFAULT_ITERATIONS


def _validate_supercell_size(
    supercell: Tuple[int, int, int], occupancies: List[float]
) -> None:
    """
    验证超胞尺寸（现已移除）

    注意：超胞合法性不再基于 LCM，而是基于 RSS 最小性。
    合法的候选超胞已在分析阶段筛选，只有最小 RSS 的超胞被选中，
    因此无需在此再次验证。

    Args:
        supercell: 超胞尺寸 (nx, ny, nz)
        occupancies: 占据数列表
    """
    # 不再执行 LCM 检查，合法性已由上游保证
    pass


def build_split_config(
    structure: Structure,
    analysis: Dict[str, Any],
    supercell: Tuple[int, int, int],
    iterations: int = DEFAULT_ITERATIONS,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    使用 split 模式构建配置（官方推荐方法）

    split 模式的核心特性：
    1. 只在 composition 中指定需要优化的无序位点
    2. 未指定的位点会自动保留原始元素（有序位点）
    3. 使用 sites 字段指定每个sublattice对应的位点索引

    Args:
        structure: pymatgen Structure 对象
        analysis: disorder_analyzer.analyze_structure() 的结果
        supercell: 超胞尺寸 (nx, ny, nz)
        iterations: 迭代次数
        debug: 是否输出调试信息

    Returns:
        sqsgenerator 配置字典（split模式）

    Example:
        >>> # 结构：W(有序) + FeCo(无序)
        >>> config = build_split_config(structure, analysis, (2,2,2))
        >>> print(config['sublattice_mode'])
        'split'
        >>> print(config['composition'])
        [{'sites': [1], 'Fe': 4, 'Co': 4}]  # 只指定无序位点1
        # W位点0会自动保留
    """
    site_groups = analysis["site_groups"]
    nx, ny, nz = supercell
    num_sites_in_cell = len(site_groups)
    supercell_size = nx * ny * nz

    # 分离有序和无序位点
    ordered_sites = [sg for sg in site_groups if not sg["is_disordered"]]
    disordered_sites = [sg for sg in site_groups if sg["is_disordered"]]

    num_ordered = len(ordered_sites)
    num_disordered = len(disordered_sites)

    if debug:
        print("\n" + "=" * 70)
        print("DEBUG: build_split_config (官方split模式)")
        print("=" * 70)
        print(f"超胞规格: {supercell} = {nx}×{ny}×{nz}")
        print(f"超胞大小: {supercell_size}个原胞")
        print(f"原胞位点数: {num_sites_in_cell}")
        print(f"\n位点分类:")
        print(f"  有序位点数: {num_ordered}")
        print(f"  无序位点数: {num_disordered}")

        if num_ordered > 0:
            print(f"\n有序位点（自动保留）:")
            for sg in ordered_sites:
                clean_species = {
                    elem: round(occ, 10) for elem, occ in sg["species"].items()
                }
                print(f"  位点 {sg['site_idx']}: {clean_species}")

        print(f"\n无序位点（需要优化）:")
        for sg in disordered_sites:
            clean_species = {
                elem: round(occ, 10) for elem, occ in sg["species"].items()
            }
            print(f"  位点 {sg['site_idx']}: {clean_species}")

    # 为 split 模式准备“位点标签（sites）”：
    # - sqsgenerator 在 split 模式下会把 composition[*].sites 解释为一个子晶格包含的位点集合。
    # - 如果 sites 是字符串，会按 structure.species 中的该元素符号匹配，并在 supercell 展开后
    #   自动把所有匹配到的位点纳入该子晶格。
    # - 如果 sites 是索引列表，则是显式列出具体位点（需要超胞展开后的索引），配置会非常大。
    # 因此这里对每个“无序位点”分配一个唯一的合法元素符号作为占位符标签。

    used_real_elements: set[str] = set()
    for sg in site_groups:
        for elem in sg["species"].keys():
            if elem != "0":
                used_real_elements.add(elem)

    placeholder_pool: List[str] = []
    for z in range(1, 119):
        sym = Element.from_Z(z).symbol
        if sym not in used_real_elements:
            placeholder_pool.append(sym)

    if len(placeholder_pool) < num_disordered:
        raise ValueError(
            "Not enough placeholder elements available to label disordered sites in split mode. "
            f"Need {num_disordered}, have {len(placeholder_pool)}."
        )

    placeholder_by_site_idx: Dict[int, str] = {}
    for i, sg in enumerate(disordered_sites):
        placeholder_by_site_idx[sg["site_idx"]] = placeholder_pool[i]

    # 构建 structure 字段（包含所有位点！）
    lattice_matrix = structure.lattice.matrix.tolist()

    # 提取所有位点的坐标和物种
    coords = []
    species = []
    for sg in site_groups:  # 包含所有位点
        coords.append(sg["coords"])

        site_idx = sg["site_idx"]
        if sg["is_disordered"]:
            species.append(placeholder_by_site_idx[site_idx])
        else:
            # 有序位点：使用真实元素符号作为占位符，且不进入 composition（从而保留不变）
            first_element = next(iter(sg["species"].keys()))
            species.append(first_element)

    # 构建 composition 列表（只包含无序位点的sublattice定义）
    composition_list = []

    for sg in disordered_sites:
        site_idx = sg["site_idx"]
        placeholder = placeholder_by_site_idx[site_idx]

        # 计算这个位点在超胞中的元素数量
        # 使用“最大余数法”保证总和严格等于 supercell_size，避免 round 导致的偏差。
        raw_items: List[Tuple[str, float]] = []
        for element, occ in sg["species"].items():
            raw_items.append((element, float(occ) * supercell_size))

        floored: Dict[str, int] = {e: int(math.floor(v + 1e-12)) for e, v in raw_items}
        total_floor = sum(floored.values())
        remainder = supercell_size - total_floor

        if remainder < 0:
            # 极端情况下（浮点误差）可能发生，回退到 round
            floored = {e: int(round(v)) for e, v in raw_items}
            total_floor = sum(floored.values())
            remainder = supercell_size - total_floor

        remainders = sorted(
            ((e, (v - math.floor(v + 1e-12))) for e, v in raw_items),
            key=lambda x: x[1],
            reverse=True,
        )

        counts: Dict[str, int] = dict(floored)
        for i in range(max(0, remainder)):
            e, _ = remainders[i % len(remainders)]
            counts[e] = counts.get(e, 0) + 1

        # 过滤 0 计数
        sublattice_comp = {e: c for e, c in counts.items() if c > 0}

        # split 模式：用字符串 placeholder 选择该位点在超胞展开后的全部等价位点
        sublattice_def = {"sites": placeholder, **sublattice_comp}

        composition_list.append(sublattice_def)

        if debug:
            print(f"\nSublattice for 位点 {site_idx}:")
            print(f'  sites: "{placeholder}" (placeholder)')
            for elem, count in sublattice_comp.items():
                print(f"  {elem}: {count}")

    # 构建最终配置（干净，不包含内部元数据）
    config = {
        "sublattice_mode": "split",
        "structure": {
            "lattice": lattice_matrix,
            "coords": coords,
            "species": species,
            "supercell": list(supercell),
        },
        "composition": composition_list,
        "iterations": iterations,
    }

    if debug:
        print(f"\n最终配置（split模式）:")
        print(f"  sublattice_mode: split")
        print(f"  structure.coords: {len(coords)} 个坐标（所有位点）")
        print(f"  structure.species: {species}")
        print(f"  composition: {len(composition_list)} 个sublattice（只包含无序位点）")
        print(f"\n关键特性:")
        print(f"  ✓ 有序位点（{num_ordered}个）包含在structure中，但不在composition中")
        print(f"  ✓ 有序位点会自动保留原始元素")
        print(f"  ✓ 只有无序位点（{num_disordered}个）会被优化")
        print("=" * 70 + "\n")

    return config


def build_sqs_config(
    structure: Structure,
    analysis: Dict[str, Any],
    supercell: Optional[Tuple[int, int, int]] = None,
    iterations: int = DEFAULT_ITERATIONS,
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    根据无序类型自动构建配置（统一使用split模式）

    这是主入口函数，自动处理所有类型的无序。

    Args:
        structure: pymatgen Structure 对象
        analysis: disorder_analyzer.analyze_structure() 的结果
        supercell: 超胞尺寸，如果为 None 则自动计算
        iterations: 迭代次数
        debug: 是否输出调试信息
        **kwargs: 额外参数

    Returns:
        sqsgenerator 配置字典（split模式）

    Raises:
        ValueError: 如果无序类型无法处理

    Example:
        >>> config = build_sqs_config(structure, analysis)
        # 自动使用split模式
    """
    # 获取无序类型（统一使用多标签格式）
    disorder_types = analysis.get("disorder_types", [])

    # 检查是否完全有序
    if not disorder_types:
        raise ValueError("Structure is fully ordered - no SQS needed")

    # 如果没有指定超胞，使用 RSS 约束下的最优候选
    if supercell is None:
        from .supercell_optimizer import find_optimal_supercell

        opt = find_optimal_supercell(
            analysis.get("occupancies", []),
            max_error=0.0005,
            prefer_smaller=True,
        )
        if not opt.get("success"):
            raise ValueError(
                "Auto supercell selection failed: "
                f"{opt.get('message', 'No valid candidate from RSS optimization')}"
            )
        supercell = tuple(opt["supercell"])

    # 验证超胞规格
    _validate_supercell_size(supercell, analysis["occupancies"])

    # 所有类型的无序统一使用split模式
    config = build_split_config(
        structure, analysis, supercell, iterations, debug=debug, **kwargs
    )

    return config


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证 sqsgenerator 配置的有效性

    Args:
        config: sqsgenerator 配置字典

    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []

    # 检查必需字段
    if "structure" not in config:
        errors.append("Missing 'structure' field")
        return False, errors

    structure = config["structure"]

    # 检查 lattice
    if "lattice" not in structure:
        errors.append("Missing 'structure.lattice'")
    elif not isinstance(structure["lattice"], list):
        errors.append("'structure.lattice' must be a list")
    elif len(structure["lattice"]) != 3:
        errors.append("'structure.lattice' must be 3x3 matrix")

    # 检查 coords
    if "coords" not in structure:
        errors.append("Missing 'structure.coords'")
    elif not isinstance(structure["coords"], list):
        errors.append("'structure.coords' must be a list")

    # 检查 supercell
    if "supercell" not in structure:
        errors.append("Missing 'structure.supercell'")
    elif len(structure["supercell"]) != 3:
        errors.append("'structure.supercell' must have 3 dimensions")
    elif any(s < 1 for s in structure["supercell"]):
        errors.append("'structure.supercell' dimensions must be >= 1")

    # split模式检查
    if "sublattice_mode" not in config:
        errors.append("Missing 'sublattice_mode' (should be 'split')")
    elif config["sublattice_mode"] != "split":
        errors.append(
            f"Invalid sublattice_mode: {config['sublattice_mode']} (should be 'split')"
        )

    if "composition" not in config:
        errors.append("Missing 'composition'")
    elif not isinstance(config["composition"], list):
        errors.append("'composition' must be a list in split mode")
    else:
        coords_len = (
            len(structure.get("coords", [])) if isinstance(structure, dict) else 0
        )
        for i, sub in enumerate(config["composition"]):
            if not isinstance(sub, dict):
                errors.append(f"composition[{i}] must be a dict")
                continue
            if "sites" not in sub:
                errors.append(f"composition[{i}] missing 'sites'")
                continue
            sites = sub.get("sites")

            # 当前 sqsgenerator 版本：sites 可以是字符串（按 species 标签匹配并随 supercell 展开）
            # 或者是索引列表（显式指定具体位点）。
            if isinstance(sites, str):
                if "species" in structure and sites not in structure.get("species", []):
                    errors.append(
                        f'composition[{i}].sites="{sites}" not found in structure.species'
                    )
            elif isinstance(sites, list) and all(isinstance(x, int) for x in sites):
                if coords_len and any((x < 0 or x >= coords_len) for x in sites):
                    errors.append(
                        f"composition[{i}].sites contains out-of-range indices (coords len={coords_len})"
                    )
            else:
                errors.append(
                    f"composition[{i}].sites must be str or list[int] (got {type(sites).__name__})"
                )

    if "species" not in structure:
        errors.append("Missing 'structure.species'")

    # 检查 iterations
    if "iterations" not in config:
        errors.append("Missing 'iterations'")
    elif config["iterations"] < 1:
        errors.append("'iterations' must be >= 1")

    is_valid = len(errors) == 0
    return is_valid, errors


def get_config_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    提取配置的摘要信息（用于调试）

    Args:
        config: sqsgenerator 配置字典

    Returns:
        配置摘要
    """
    info = {
        "mode": "split",
        "supercell": config.get("structure", {}).get("supercell", None),
        "iterations": config.get("iterations", None),
        "num_sites": len(config.get("structure", {}).get("coords", [])),
        "composition_list": config.get("composition", []),
        "num_sublattices": len(config.get("composition", [])),
    }

    return info
