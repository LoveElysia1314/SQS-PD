"""
超胞规格计算器
=====================================
功能：
1. 从占据数分母计算最小公倍数(LCM)
2. 生成所有有效的超胞规格（LCM的倍数）
3. 为每个规格找到最优形状（最接近正方体）
"""

import math
from functools import reduce, lru_cache
from typing import List, Tuple

from ..foundation.fraction_utils import fraction_from_occupancy


def gcd(a: int, b: int) -> int:
    """
    计算最大公约数（Greatest Common Divisor）

    Args:
        a: 第一个整数
        b: 第二个整数

    Returns:
        最大公约数

    Example:
        >>> gcd(12, 8)
        4
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """
    计算最小公倍数（Least Common Multiple）

    Args:
        a: 第一个整数
        b: 第二个整数

    Returns:
        最小公倍数

    Example:
        >>> lcm(4, 6)
        12
    """
    return abs(a * b) // gcd(a, b)


def lcm_multiple(numbers: List[int]) -> int:
    """
    计算多个数的最小公倍数

    Args:
        numbers: 整数列表

    Returns:
        所有数的最小公倍数

    Example:
        >>> lcm_multiple([2, 3, 4])
        12
    """
    return reduce(lcm, numbers)


def extract_denominators(occupancies: List[float]) -> List[int]:
    """
    从占据数提取分母

    将浮点数占据转换为分数形式，并提取分母。
    使用 limit_denominator(100) 来避免过大的分母。

    Args:
        occupancies: 占据数列表，如 [0.5, 0.25, 0.75]

    Returns:
        分母列表，如 [2, 4, 4]

    Raises:
        ValueError: 如果列表为空或占据数超出范围 (0, 1]
        TypeError: 如果列表中包含非数字类型

    Example:
        >>> extract_denominators([0.5, 0.25, 0.75])
        [2, 4, 4]
        >>> extract_denominators([1/3, 2/3])
        [3, 3]
    """
    # 输入验证
    if not occupancies:
        raise ValueError("占据数列表不能为空")

    denominators = []
    for i, occ in enumerate(occupancies):
        # 类型检查
        if not isinstance(occ, (int, float)):
            raise TypeError(
                f"占据数必须是数字，第 {i} 个元素为 {type(occ).__name__}: {occ}"
            )

        # 范围检查
        if occ <= 0 or occ > 1:
            raise ValueError(f"占据数必须在 (0, 1] 范围内，第 {i} 个元素为 {occ}")

        # 使用较小的limit以获得合理的分数
        # 例如 0.333 -> 1/3 而不是 333/1000
        frac = fraction_from_occupancy(occ, max_den=100)
        denominators.append(frac.denominator)

    return denominators


def calculate_lcm_from_occupancies(occupancies: List[float]) -> int:
    """
    从占据数计算最小公倍数

    这是确保超胞能够精确表示所有占据数的最小尺寸。
    使用缓存优化性能，避免重复计算。

    Args:
        occupancies: 占据数列表

    Returns:
        最小公倍数

    Example:
        >>> calculate_lcm_from_occupancies([0.5, 0.25])
        4
        >>> calculate_lcm_from_occupancies([1/3, 2/3])
        3
    """
    # 转换为元组以支持缓存
    return _calculate_lcm_cached(tuple(occupancies))


@lru_cache(maxsize=32)
def _calculate_lcm_cached(occupancies_tuple: Tuple[float, ...]) -> int:
    """
    带缓存的LCM计算（内部函数）

    使用 @lru_cache 装饰器避免重复计算相同占座数的LCM。
    支持高效的批量处理。

    Args:
        occupancies_tuple: 占据数元组（可哈希）

    Returns:
        最小公倍数

    Raises:
        ValueError: 如果占据数无效或列表为空
        TypeError: 如果占据数包含无效类型
    """
    try:
        denoms = extract_denominators(list(occupancies_tuple))
        result = lcm_multiple(denoms)

        # 验证结果合理性
        if result <= 0:
            raise ValueError(f"LCM 计算结果无效: {result}")

        return result
    except (ValueError, TypeError) as e:
        raise type(e)(f"LCM 计算失败 (占据数: {occupancies_tuple}): {str(e)}") from e
    except Exception as e:
        raise ValueError(f"LCM 计算出现意外错误: {str(e)}") from e
