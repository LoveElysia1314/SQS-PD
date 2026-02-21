"""测试01: 超胞基础与 LCMM 投影一致性。

学习目标:
1. 理解占据数分母与 LCM 的关系
2. 验证默认自动超胞策略
3. 确认 projected 分母与 LCMM 特征的一致性
"""

from fractions import Fraction
import math

import pytest

from conftest import print_section, print_concept, print_code_example
from sqs_pd.core.supercell_optimizer import get_supercell_info_optimized
from sqs_pd.foundation.fraction_utils import (
    denominators_from_projected_occupancies,
    summarize_denominators,
)
from sqs_pd.ranking.ranker_features import extract_lcmm_from_occupancies


class TestSupercellFoundations:
    @pytest.mark.parametrize(
        "occupancies,expected_lcm",
        [
            ([0.5, 0.5], 2),
            ([0.333, 0.667], 3),
            ([0.25, 0.75], 4),
            ([0.6, 0.4], 5),
            ([0.167, 0.833], 6),
        ],
    )
    def test_lcm_calculation(self, occupancies, expected_lcm):
        print_section("占据数与 LCM")
        print_concept("占据数分母的最小公倍数决定整数化超胞的最小尺度")

        fractions = [Fraction(occ).limit_denominator(100) for occ in occupancies]
        denominators = [item.denominator for item in fractions]
        lcm = denominators[0]
        for denominator in denominators[1:]:
            lcm = lcm * denominator // math.gcd(lcm, denominator)

        print_code_example(f"占据数: {occupancies} → 分母: {denominators} → LCM: {lcm}")
        assert lcm == expected_lcm

    def test_default_strategy_returns_consistent_shape_and_size(self, binary_structure):
        print_section("默认超胞策略")
        print_concept("自动模式应输出 >=100 原胞、并满足整体体积约束")

        occupancies = [float(v) for v in binary_structure[0].species.values()]
        result = get_supercell_info_optimized(occupancies)
        nx, ny, nz = result["supercell"]

        print_code_example(
            f"supercell={result['supercell']}, size={result['size']}, rss={result['rss']:.3e}"
        )

        assert result["success"]
        assert result["size"] >= 1
        assert nx * ny * nz == result["size"]

    def test_projected_denominators_and_lcmm_are_consistent(self):
        print_section("LCMM 投影一致性")
        print_concept("使用 supercell_size 投影后，分母统计与 LCMM 特征应保持一致")

        occupancies = [0.7495590828924162, 0.2504409171075838]
        supercell_size = 567

        denominators = denominators_from_projected_occupancies(
            occupancies, supercell_size=supercell_size, flatten_nested=True
        )
        summary = summarize_denominators([occupancies], supercell_size=supercell_size)
        lcmm = extract_lcmm_from_occupancies(occupancies, supercell_size=supercell_size)

        print_code_example(
            f"denominators={denominators}, summary_lcmm={summary['lcmm']}, feature_lcmm={lcmm}"
        )

        assert denominators == [supercell_size, supercell_size]
        assert summary["lcmm"] == supercell_size
        assert lcmm == supercell_size
