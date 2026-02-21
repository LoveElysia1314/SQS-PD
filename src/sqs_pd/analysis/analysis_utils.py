"""
共享分析工具函数
======================================
公共的无序分析逻辑，供 disorder_analyzer.py 和 cif_disorder_analyzer.py 使用

提供的功能：
- classify_site_sd_pd_spd：单个位点分类（Ordered/SD/PD/SPD）
- extract_occupancy_fractions：提取占座分数
- classify_disorder_types_from_sites：从位点列表综合分类
"""

from typing import Dict, List, Set

from ..foundation.constants import OCCUPANCY_TOLERANCE, DISORDER_TYPES_ORDER

# 无序类型顺序（从constants导入，这里保持向后兼容）
DISORDER_ORDER = DISORDER_TYPES_ORDER


def classify_site_sd_pd_spd(
    num_species: int, total_occupancy: float, tol: float = None
) -> str:
    """按规范对单个位点分类，返回 Ordered/SD/PD/SPD。

    规则：
    - Ordered: m=1 且 W≈1
    - SD:      m>1 且 W≈1
    - PD:      m=1 且 W<1
    - SPD:     m>1 且 W<1

    其中 m 为非空位元素种类数，W 为非空位总占据度。

    Args:
        num_species: 非空位元素种类数
        total_occupancy: 非空位总占据度
        tol: 判断 W≈1 的容差（若为None则使用 OCCUPANCY_TOLERANCE）

    Returns:
        分类结果字符串：Ordered/SD/PD/SPD/Invalid

    Example:
        >>> classify_site_sd_pd_spd(2, 1.0)  # Fe0.5Co0.5
        'SD'
        >>> classify_site_sd_pd_spd(1, 0.5)  # Ni0.5
        'PD'
        >>> classify_site_sd_pd_spd(1, 1.0)  # Fe
        'Ordered'
    """
    if tol is None:
        tol = OCCUPANCY_TOLERANCE
    if abs(total_occupancy - 1.0) <= tol:
        return "Ordered" if num_species <= 1 else "SD"

    if total_occupancy < 1.0 - tol:
        return "PD" if num_species <= 1 else "SPD"

    # total_occupancy > 1
    return "Invalid"


def extract_occupancy_fractions(occupancy_list: List[float]) -> List[float]:
    """提取所有占据数（用于计算LCM）

    从占据度列表中过滤出需要参与 LCM 计算的分数
    （即排除完全占据1.0和完全空缺0.0的值）

    Args:
        occupancy_list: 占据度列表

    Returns:
        去重并排序的占据数列表，不包括 0 或 1
        若列表为空或所有值都是 0/1，返回 [1.0]

    Example:
        >>> extract_occupancy_fractions([0.5, 0.5, 1.0, 0.5])
        [0.5]
        >>> extract_occupancy_fractions([1.0, 1.0])
        [1.0]
    """
    # 过滤出 (0, 1) 范围内的占座数（排除完全占据和空位）
    filtered = [occ for occ in occupancy_list if 0.001 < occ < 0.999]

    # 去重并排序
    unique_occs = sorted(set(filtered))

    return unique_occs if unique_occs else [1.0]


def classify_disorder_types_from_sites(
    site_classifications: List[str],
) -> List[str]:
    """从位点分类列表综合得出结构级无序类型

    根据所有位点的分类结果（Ordered/SD/PD/SPD），统计哪些无序类型出现过。
    结果中不包含 Ordered 或 Invalid。

    Args:
        site_classifications: 位点分类列表，每个元素为 Ordered/SD/PD/SPD/Invalid

    Returns:
        结构级无序类型列表，按 SD/PD/SPD 顺序排列
        若无任何无序位点，返回空列表

    Example:
        >>> classify_disorder_types_from_sites(['SD', 'Ordered', 'PD'])
        ['SD', 'PD']
        >>> classify_disorder_types_from_sites(['Ordered', 'Ordered'])
        []
    """
    disorder_types_set: Set[str] = set()

    for cls in site_classifications:
        if cls in DISORDER_ORDER:
            disorder_types_set.add(cls)

    # 按规范顺序排列
    return [t for t in DISORDER_ORDER if t in disorder_types_set]
