"""测试04: CIF 无序分析与 dry-run 优化行为。

学习目标:
1. 保留 SD/PD 识别能力验证
2. 覆盖 dry-run 优化器关键返回语义
3. 吸收 examples 中有效验证逻辑
"""

from pathlib import Path

import pytest

from conftest import print_section, print_concept, print_code_example
from sqs_pd import analyze_cif_and_recommend_supercell, find_optimal_supercell
from sqs_pd.analysis.cif_disorder_analyzer import analyze_cif_disorder
from sqs_pd.runtime.dry_run import extract_disordered_occupancies


class TestDisorderAnalysis:
    def test_analyze_disorder_types(self, demo_sd_cif, demo_pd_cif):
        print_section("CIF 无序类型")
        print_concept("自动识别 SD / PD / SPD")

        sd_result = analyze_cif_disorder(str(demo_sd_cif))
        pd_result = analyze_cif_disorder(str(demo_pd_cif))

        assert (
            "SD" in sd_result["disorder_types"] or "SPD" in sd_result["disorder_types"]
        )
        assert (
            "PD" in pd_result["disorder_types"] or "SPD" in pd_result["disorder_types"]
        )

    def test_extract_disordered_occupancies_boundary(self):
        print_section("占据边界过滤")
        print_concept("<0.0005 与 >=0.9995 视作有序，过滤后仅保留无序占据")

        sites = [
            {"species": {"Fe": 0.0001}},
            {"species": {"Co": 0.0005}},
            {"species": {"Ni": 0.5}},
            {"species": {"Cu": 0.9995}},
            {"species": {"Zn": 0.9999}},
            {"species": {"Al": 0.33}},
        ]
        occupancies = extract_disordered_occupancies(sites)
        print_code_example(f"filtered occupancies = {occupancies}")
        flattened = sorted([value for group in occupancies for value in group])
        assert flattened == [0.33, 0.5]


class TestDryRunOptimization:
    @pytest.mark.parametrize(
        "occupancies",
        [[0.5], [0.5, 0.5], [0.333, 0.666], [0.25, 0.5, 0.75], [0.97]],
    )
    def test_find_optimal_supercell_success_cases(self, occupancies):
        print_section("优化器直接调用")
        print_concept("典型占据数组合应得到有效候选")

        result = find_optimal_supercell(occupancies, max_error=0.0005)
        assert result["success"]
        assert result["size"] > 0
        assert len(result["supercell"]) == 3

    def test_candidates_meet_max_error_constraint(self):
        print_section("候选约束验证")
        print_concept("all_candidates 中每个候选都应满足 max_error 阈值")

        cif_path = Path(__file__).parent.parent / "data" / "input" / "demo_sd.cif"
        analysis = analyze_cif_and_recommend_supercell(cif_path, verbose=False)

        assert analysis["optimization_success"]
        candidates = analysis.get("all_candidates", [])
        assert candidates
        assert all(cand["max_error"] <= 0.0005 for cand in candidates)
