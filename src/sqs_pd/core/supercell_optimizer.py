"""
超胞规格优化器
=====================================
功能：
1. 加载预定义的合法超胞规格集合
2. 为给定占据数找到最优超胞规格
3. 最小化全局残差平方和（RSS）
4. 支持容差约束

设计目标：
- 使用预计算的合法规格（满足 3 ≤ min_dim ≤ max_dim ≤ 10）
- 确保每个占据数的近似误差 ≤ max_error（默认 0.0005）
- 选择使全局 RSS 最小的规格

用法示例：
    >>> from supercell_optimizer import find_optimal_supercell
    >>> occupancies = [0.5, 0.333, 0.666]
    >>> result = find_optimal_supercell(occupancies)
    >>> print(result['supercell'])
    (6, 6, 3)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import lru_cache

from ..foundation.constants import OCCUPANCY_TOLERANCE

# 模块级缓存：避免重复加载 CSV
_VALID_SPECS_CACHE: Optional[List[Tuple[int, int, int, int]]] = None


def get_data_file_path(filename: str) -> Path:
    """获取数据文件路径

    Args:
        filename: 数据文件名

    Returns:
        数据文件的完整路径
    """
    return Path(__file__).parent / "data" / filename


def _sphericity(l: int, w: int, h: int) -> float:
    """球形度：越接近立方体越大。"""
    geo_mean = (l * w * h) ** (1.0 / 3.0)
    arith_mean = (l + w + h) / 3.0
    return geo_mean / arith_mean if arith_mean > 0 else 0.0


def _generate_valid_specs(
    min_size: int = 27,
    max_size: int = 1000,
    min_dim: int = 3,
    max_dim: int = 10,
) -> List[Tuple[int, int, int, int]]:
    """按规则生成合法超胞规格。

    规则：
    - 枚举 size in [min_size, max_size]
    - 仅保留满足 min_dim <= h <= w <= l <= max_dim 且 l*w*h=size 的形状
    - 对每个 size 选择球形度最大的 (l, w, h)
    """
    specs: List[Tuple[int, int, int, int]] = []

    for size in range(min_size, max_size + 1):
        best_shape: Optional[Tuple[int, int, int]] = None
        best_score = -1.0

        for h in range(min_dim, max_dim + 1):
            if size % h != 0:
                continue
            rest = size // h

            for w in range(h, max_dim + 1):
                if rest % w != 0:
                    continue
                l = rest // w

                if l < w:
                    continue
                if l < min_dim or l > max_dim:
                    continue

                score = _sphericity(l, w, h)
                if score > best_score + 1e-15 or (
                    math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-15)
                    and (best_shape is None or (l, w, h) < best_shape)
                ):
                    best_score = score
                    best_shape = (l, w, h)

        if best_shape is not None:
            l, w, h = best_shape
            specs.append((size, l, w, h))

    return specs


def _write_valid_specs_csv(
    csv_path: Path,
    specs: List[Tuple[int, int, int, int]],
) -> None:
    """将规格写入 CSV。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "l", "w", "h"])
        for row in specs:
            writer.writerow(row)


@lru_cache(maxsize=1)
def load_valid_supercell_specs() -> List[Tuple[int, int, int, int]]:
    """加载预定义的合法超胞规格

    从嵌入的 CSV 文件读取满足以下约束的规格：
    - 最小维度 >= 3
    - 最大维度 <= 10
    - 对于给定超胞规模，选择球形度最大的 (l, w, h) 组合

    Returns:
        规格列表：[(size, l, w, h), ...]
        size: 超胞规模（包含的原胞总数）
        l, w, h: 超胞三个维度（l >= w >= h）

    Raises:
        FileNotFoundError: 如果数据文件不存在

    Example:
        >>> specs = load_valid_supercell_specs()
        >>> specs[0]
        (27, 3, 3, 3)
        >>> len(specs)
        90
    """
    global _VALID_SPECS_CACHE

    if _VALID_SPECS_CACHE is not None:
        return _VALID_SPECS_CACHE

    csv_path = get_data_file_path("valid_supercell_specs.csv")

    if not csv_path.exists():
        # 缺失时按规则自动重建（每个 size 取球形度最优形状）
        generated_specs = _generate_valid_specs(
            min_size=27,
            max_size=1000,
            min_dim=3,
            max_dim=10,
        )
        _write_valid_specs_csv(csv_path, generated_specs)

    specs = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(row["size"])
            l = int(row["l"])
            w = int(row["w"])
            h = int(row["h"])
            specs.append((size, l, w, h))

    _VALID_SPECS_CACHE = specs
    return specs


def compute_approximation_errors(
    occupancies: Union[List[float], List[List[float]]], size: int
) -> Tuple[List[float], List[float], float]:
    """计算给定规模下的占据数近似误差

    对每个占据数，计算最近分数及其误差。

    Args:
        occupancies: 占据数列表（扁平或嵌套）
        size: 超胞规模（作为分母）

    Returns:
        (errors, nearest_fractions, max_abs_error)
        errors: 误差列表 [occ - nearest, ...]
        nearest_fractions: 最近分数列表
        max_abs_error: 最大绝对误差

    Example:
        >>> errors, fracs, max_err = compute_approximation_errors([0.5, 0.333], 27)
        >>> max_err <= 0.0005
        True
    """
    # Flatten if nested
    flat_occs = []
    for item in occupancies:
        if isinstance(item, (list, tuple)):
            flat_occs.extend(item)
        else:
            flat_occs.append(item)

    errors = []
    nearest_fractions = []

    for occ in flat_occs:
        # 计算最近分数：round(occ × size) / size
        numerator = round(occ * size)
        nearest = numerator / size
        error = occ - nearest

        errors.append(error)
        nearest_fractions.append(nearest)

    max_abs_error = max(abs(e) for e in errors) if errors else 0.0

    return errors, nearest_fractions, max_abs_error


def _collect_valid_candidates(
    occupancies: List[float],
    valid_specs: List[Tuple[int, int, int, int]],
    max_error: float,
) -> List[Dict[str, Any]]:
    """收集满足误差约束的候选规格（内部函数）。"""
    candidates: List[Dict[str, Any]] = []

    for size, l, w, h in valid_specs:
        errors, nearest_fractions, max_abs_error = compute_approximation_errors(
            occupancies, size
        )

        if max_abs_error <= max_error:
            rss = sum(e**2 for e in errors)
            candidates.append(
                {
                    "size": size,
                    "supercell": (l, w, h),
                    "rss": rss,
                    "max_error": max_abs_error,
                    "errors": errors,
                    "nearest_fractions": nearest_fractions,
                }
            )

    return candidates


def find_optimal_supercell(
    occupancies: List[float],
    valid_specs: Optional[List[Tuple[int, int, int, int]]] = None,
    max_error: float = 0.0005,
    prefer_smaller: bool = True,
) -> Dict[str, Any]:
    """为给定占据数找到最优超胞规格

    策略：
    1. 遍历所有合法规格
    2. 对每个规格，检查所有占据数的近似误差是否 ≤ max_error
    3. 在满足条件的规格中，选择 RSS 最小的
    4. 如果 RSS 相同，选择规模较小的（prefer_smaller=True）

    Args:
        occupancies: 占据数列表（应已过滤 0.0005 < occ < 0.9995）
        valid_specs: 合法规格列表，None 则自动加载
        max_error: 最大允许误差（默认 0.0005）
        prefer_smaller: RSS 相同时是否优先选择较小规模

    Returns:
        结果字典，包含：
        - success: bool，是否找到满足条件的规格
        - supercell: Tuple[int, int, int]，最优超胞形状 (l, w, h)
        - size: int，超胞规模
        - rss: float，残差平方和
        - max_error: float，该规格下的最大误差
        - errors: List[float]，每个占据数的误差
        - nearest_fractions: List[float]，近似分数
        - num_candidates: int，满足条件的规格数量
        - message: str，描述信息

    Example:
        >>> result = find_optimal_supercell([0.5, 0.333, 0.666])
        >>> result['success']
        True
        >>> result['supercell']
        (6, 6, 3)
    """
    if valid_specs is None:
        valid_specs = load_valid_supercell_specs()

    if not occupancies:
        return {
            "success": False,
            "supercell": None,
            "size": None,
            "rss": None,
            "max_error": None,
            "errors": [],
            "nearest_fractions": [],
            "num_candidates": 0,
            "message": "No occupancies provided",
        }

    candidates = _collect_valid_candidates(occupancies, valid_specs, max_error)

    if not candidates:
        return {
            "success": False,
            "supercell": None,
            "size": None,
            "rss": None,
            "max_error": None,
            "errors": [],
            "nearest_fractions": [],
            "num_candidates": 0,
            "message": f"No valid supercell found with max_error <= {max_error}",
        }

    # 排序：按 RSS 升序，RSS 相同时按 size 升序（如果 prefer_smaller）
    if prefer_smaller:
        candidates.sort(key=lambda x: (x["rss"], x["size"]))
    else:
        candidates.sort(key=lambda x: x["rss"])

    best = candidates[0]
    best_l, best_w, best_h = best["supercell"]
    best_size = best["size"]
    best_rss = best["rss"]
    best_max_err = best["max_error"]
    best_errors = best["errors"]
    best_nearest = best["nearest_fractions"]

    message = (
        f"Perfect match found at size={best_size}"
        if best_rss < 1e-15
        else (
            f"Found optimal supercell: {best_l}×{best_w}×{best_h} "
            f"(size={best_size}, RSS={best_rss:.2e})"
        )
    )

    return {
        "success": True,
        "supercell": (best_l, best_w, best_h),
        "size": best_size,
        "rss": best_rss,
        "max_error": best_max_err,
        "errors": best_errors,
        "nearest_fractions": best_nearest,
        "num_candidates": len(candidates),
        "message": message,
    }


def get_supercell_info_optimized(
    occupancies: List[float], max_error: float = 0.0005
) -> Dict[str, Any]:
    """获取超胞规格的详细信息（优化版本）

    提供比 find_optimal_supercell 更详细的诊断信息。

    Args:
        occupancies: 占据数列表
        max_error: 最大允许误差

    Returns:
        详细信息字典，包含：
        - 基本结果（同 find_optimal_supercell）
        - 额外信息：
          - all_candidates: 所有满足条件的候选规格
          - occupancies_input: 输入占据数
          - rss_range: (min_rss, max_rss) RSS 范围

    Example:
        >>> info = get_supercell_info_optimized([0.5, 0.25])
        >>> print(info['supercell'])
        (4, 4, 4)
        >>> print(len(info['all_candidates']))
        15
    """
    valid_specs = load_valid_supercell_specs()
    all_candidates = _collect_valid_candidates(occupancies, valid_specs, max_error)
    result = find_optimal_supercell(occupancies, valid_specs, max_error)

    # 排序候选
    all_candidates.sort(key=lambda x: (x["rss"], x["size"]))

    # RSS 范围
    if all_candidates:
        rss_values = [c["rss"] for c in all_candidates]
        rss_range = (min(rss_values), max(rss_values))
    else:
        rss_range = (None, None)

    result.update(
        {
            "occupancies_input": occupancies,
            "all_candidates": all_candidates,
            "rss_range": rss_range,
            "num_candidates": len(all_candidates),
        }
    )

    return result
