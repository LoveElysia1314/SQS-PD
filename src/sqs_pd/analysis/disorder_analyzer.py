"""
无序分析器
=====================================
功能：
1. 分析晶体结构的无序类型
2. 分类：SD（置换无序）、PD（位置无序/缺位）、SPD（同位点混合无序）、完全有序
3. 提取占据数分母和组成信息
4. 为配置生成器提供标准化输出

用法示例：
    >>> from disorder_analyzer import analyze_structure
    >>> from pymatgen.core import Structure
    >>>
    >>> structure = Structure.from_file("demo_sd.cif")
    >>> result = analyze_structure(structure)
    >>> print(result['disorder_type'])
    'SD'
    >>> print(result['occupancies'])
    [0.5, 0.5]
"""

from typing import Dict, List, Tuple, Any
from pymatgen.core import Structure
from collections import defaultdict

# 导入依赖模块
from .analysis_utils import (
    classify_site_sd_pd_spd as classify_site_sd_pd_spd_util,
    extract_occupancy_fractions as extract_occupancy_fractions_util,
    classify_disorder_types_from_sites,
)
from ..foundation.constants import (
    OCCUPANCY_TOLERANCE,
    FLOAT_PRECISION,
    DISORDER_TYPES_ORDER,
    DISORDER_TOLERANCE,
)


def _extract_symbol(species_obj) -> str:
    """内部辅助函数：提取物种符号

    支持多种对象类型 (Species, Element, str)，避免重复的 hasattr 检查
    """
    if hasattr(species_obj, "element"):
        return str(species_obj.element.symbol)
    if hasattr(species_obj, "symbol"):
        return str(species_obj.symbol)
    return str(species_obj)


def extract_site_info(structure: Structure) -> List[Dict[str, Any]]:
    """从 pymatgen Structure 提取位点信息（优化版）

    Args:
        structure: pymatgen Structure 对象

    Returns:
        位点信息列表，每个位点包含：
        - site_idx: 位点索引
        - coords: 分数坐标
        - species: {element: occupancy} 字典
        - is_disordered: 是否无序（多元素或部分占据）
        - has_vacancy: 是否有空位（总占据度 < 1）

    Example:
        >>> info = extract_site_info(structure)
        >>> info[0]
        {
            'site_idx': 0,
            'coords': [0.5, 0.5, 0.5],
            'species': {'Fe': 0.5, 'Co': 0.5},
            'is_disordered': True,
            'has_vacancy': False
        }
    """
    site_groups = []

    for idx, site in enumerate(structure):
        species_dict = {}
        total_occupancy = 0.0

        # 直接遍历 site.species，避免转换+复制
        for species_obj, occ in site.species.items():
            symbol = _extract_symbol(species_obj)
            species_dict[symbol] = float(occ)
            total_occupancy += float(occ)

        # 标记空位（一次计算）
        has_vacancy = total_occupancy < DISORDER_TOLERANCE
        if has_vacancy:
            vacancy_amount = round(1.0 - total_occupancy, FLOAT_PRECISION)
            species_dict["0"] = vacancy_amount

        # 构建结果字典
        site_groups.append(
            {
                "site_idx": idx,
                "coords": site.frac_coords.tolist(),
                "species": species_dict,
                "is_disordered": len(species_dict) > 1 or has_vacancy,
                "has_vacancy": has_vacancy,
                "total_occupancy": total_occupancy,
            }
        )

    return site_groups


def classify_disorder_type(site_groups: List[Dict]) -> str:
    """
    以结构为单位汇总无序类型（SD/PD/SPD 多标签）

    分类规则（基于位点级互斥分类）：
    - 'ordered': 所有位点为 Ordered
    - 'SD' / 'PD' / 'SPD': 结构中存在对应类型的位点
    - 若同时存在多种无序位点，返回如 'SD+PD'、'PD+SPD' 等组合字符串

    Args:
        site_groups: 位点信息列表

    Returns:
        无序类型字符串

    Example:
        >>> # Fe0.5Co0.5：单个位点、总占据=1、多元素 -> SD
        >>> classify_disorder_type([{
        ...     'species': {'Fe': 0.5, 'Co': 0.5},
        ...     'total_occupancy': 1.0,
        ... }])
        'SD'

        >>> # 单元素、部分占据 -> PD
        >>> classify_disorder_type([
        ...     {'species': {'Cs': 0.97, '0': 0.03}, 'total_occupancy': 0.97},
        ... ])
        'PD'
    """
    if not site_groups:
        return "ordered"

    site_classifications = [_classify_site_disorder_type(s) for s in site_groups]
    disorder_types = classify_disorder_types_from_sites(site_classifications)

    return "ordered" if not disorder_types else "+".join(disorder_types)


def _classify_site_disorder_type(site: Dict[str, Any], tol: float = None) -> str:
    """内部辅助函数：按规范对单个位点分类。

    输入 site 来自 extract_site_info() 的单个 site_info。

    规则：
    - Ordered: m=1 且 W≈1
    - SD:      m>1 且 W≈1
    - PD:      m=1 且 W<1
    - SPD:     m>1 且 W<1

    其中 m 为非空位元素种类数，W 为非空位总占据度。
    """
    if tol is None:
        tol = OCCUPANCY_TOLERANCE
    species: Dict[str, float] = site.get("species", {})
    total_occ: float = float(site.get("total_occupancy", 0.0))
    num_species = len(
        [e for e in species.keys() if e != "0" and float(species[e]) > 0.0]
    )

    return classify_site_sd_pd_spd_util(num_species, total_occ, tol=tol)


def extract_occupancy_fractions(site_groups: List[Dict]) -> List[float]:
    """
    提取所有占据数（用于计算LCM）

    Args:
        site_groups: 位点信息列表

    Returns:
        所有占据数的列表（不包括1.0）

    Example:
        >>> site_groups = [
        ...     {'species': {'Fe': 0.5, 'Co': 0.5}},
        ...     {'species': {'Ni': 1.0}}
        ... ]
        >>> extract_occupancy_fractions(site_groups)
        [0.5, 0.5]
    """
    occupancies = []

    for site in site_groups:
        for element, occ in site["species"].items():
            # 跳过完全占据和空位
            if 0.001 < occ < DISORDER_TOLERANCE and element != "0":
                occupancies.append(float(occ))

    return extract_occupancy_fractions_util(occupancies)


def calculate_average_composition(site_groups: List[Dict]) -> Dict[str, float]:
    """
    计算平均组成（每个原胞的平均元素数）

    Args:
        site_groups: 位点信息列表

    Returns:
        元素到平均数量的映射（不包括空位"0"）

    Example:
        >>> site_groups = [
        ...     {'species': {'Fe': 0.5, 'Co': 0.5}},
        ...     {'species': {'Ni': 1.0}}
        ... ]
        >>> calculate_average_composition(site_groups)
        {'Fe': 0.5, 'Co': 0.5, 'Ni': 1.0}
    """
    composition = defaultdict(float)

    for site in site_groups:
        for element, occ in site["species"].items():
            if element != "0":  # 排除空位
                composition[element] += float(occ)

    return dict(composition)


def count_disordered_sites(site_groups: List[Dict]) -> Tuple[int, int]:
    """
    统计无序位点数量

    Args:
        site_groups: 位点信息列表

    Returns:
        (无序位点数, 总位点数)
    """
    num_disordered = sum(1 for s in site_groups if s["is_disordered"])
    num_total = len(site_groups)

    return num_disordered, num_total


def analyze_structure(structure: Structure, verbose: bool = False) -> Dict[str, Any]:
    """
    分析晶体结构的完整信息

    这是主入口函数，返回所有需要的信息供后续模块使用。

    Args:
        structure: pymatgen Structure 对象
        verbose: 是否打印详细信息

    Returns:
        分析结果字典，包含：
        - disorder_type: 无序类型
        - site_groups: 位点信息列表
        - occupancies: 占据数列表（用于LCM计算）
        - average_composition: 平均组成
        - num_sites: 总位点数
        - num_disordered_sites: 无序位点数
        - has_vacancy: 是否有空位
        - formula: 化学式

    Example:
        >>> structure = Structure.from_file("demo_sd.cif")
        >>> result = analyze_structure(structure)
        >>> result['disorder_type']
        'chemical'
        >>> result['occupancies']
        [0.5, 0.5]
        >>> result['average_composition']
        {'Fe': 0.5, 'Co': 0.5}
    """
    # 提取位点信息
    site_groups = extract_site_info(structure)

    # 分类无序类型
    disorder_type = classify_disorder_type(site_groups)

    # 位点级分类与结构级多标签
    site_disorder_types: List[str] = []
    disorder_types_set: set[str] = set()
    for sg in site_groups:
        cls = _classify_site_disorder_type(sg)
        site_disorder_types.append(cls)
        if cls in DISORDER_TYPES_ORDER:
            disorder_types_set.add(cls)

    disorder_types = [t for t in DISORDER_TYPES_ORDER if t in disorder_types_set]

    # 提取占据数
    occupancies = extract_occupancy_fractions(site_groups)

    # 计算平均组成
    avg_comp = calculate_average_composition(site_groups)

    # 统计无序位点
    num_disordered, num_total = count_disordered_sites(site_groups)
    num_ordered = num_total - num_disordered

    # 检查是否有空位
    has_vacancy = any(s["has_vacancy"] for s in site_groups)

    result = {
        "disorder_type": disorder_type,
        "disorder_types": disorder_types,
        "site_disorder_types": site_disorder_types,
        "site_groups": site_groups,
        "occupancies": occupancies,
        "average_composition": avg_comp,
        "num_sites": num_total,
        "num_disordered_sites": num_disordered,
        "num_ordered_sites": num_ordered,
        "has_vacancy": has_vacancy,
        "formula": str(structure.composition),
        "lattice_abc": [structure.lattice.a, structure.lattice.b, structure.lattice.c],
    }

    if verbose:
        print(f"Structure Analysis:")
        print(f"  Formula: {result['formula']}")
        if result.get("disorder_types"):
            print(f"  Disorder types: {result['disorder_types']}")
        else:
            print(f"  Disorder types: ordered")
        print(
            f"  Sites: {result['num_disordered_sites']}/{result['num_sites']} disordered"
        )
        print(f"  Occupancies: {result['occupancies']}")
        print(f"  Average composition: {result['average_composition']}")
        if result["has_vacancy"]:
            print(f"  Has vacancy: Yes")

    return result
