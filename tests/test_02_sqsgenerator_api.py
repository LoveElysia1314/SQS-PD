"""测试02: sqsgenerator 集成与关键语义。

学习目标:
1. 覆盖 simple/split 典型配置路径
2. 保留 ParseError 兼容分支语义
3. 用更少测试保留核心教学信息
"""

import pytest
from pymatgen.core import Structure

from conftest import print_section, print_concept, print_code_example


def _optimize_from_config(config):
    try:
        import sqsgenerator
    except Exception:
        pytest.skip("sqsgenerator package not available")

    parsed = sqsgenerator.parse_config(config)
    if "ParseError" in str(type(parsed)):
        result_pack = sqsgenerator.optimize(config)
    else:
        result_pack = sqsgenerator.optimize(parsed)
    return result_pack.best()


class TestSqsGeneratorIntegration:
    @pytest.mark.parametrize(
        "composition", [{"Ni": 0.5, "Cu": 0.5}, {"Fe": 0.75, "Cr": 0.25}]
    )
    def test_simple_mode_config_and_optimize(self, fcc_lattice, composition):
        print_section("Simple 模式")
        print_concept("单一无序子晶格走标准配置构建与优化链路")

        from sqs_pd.analysis.disorder_analyzer import analyze_structure
        from sqs_pd.core.config_builder import build_sqs_config

        structure = Structure(fcc_lattice, [composition], [[0, 0, 0]])
        analysis = analyze_structure(structure)
        config = build_sqs_config(
            structure, analysis, supercell=(2, 2, 2), iterations=100
        )
        best = _optimize_from_config(config)

        print_code_example("simple: analyze_structure -> build_sqs_config -> optimize")
        assert best is not None
        assert getattr(best, "objective", None) is not None

    def test_split_mode_config_and_optimize(self, fcc_lattice):
        print_section("Split 模式")
        print_concept("有序位点 + 无序位点混合时应能稳定完成优化")

        from sqs_pd.analysis.disorder_analyzer import analyze_structure
        from sqs_pd.core.config_builder import build_sqs_config

        species = [{"O": 1.0}, {"Ni": 0.5, "Cu": 0.5}]
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        structure = Structure(fcc_lattice, species, coords)

        analysis = analyze_structure(structure)
        config = build_sqs_config(
            structure, analysis, supercell=(2, 2, 2), iterations=100
        )
        best = _optimize_from_config(config)

        print_code_example(
            "split: ordered sites stay fixed, disordered sites are optimized"
        )
        assert best is not None
        assert getattr(best, "objective", None) is not None

    def test_sites_semantics_via_project_builder(self, fcc_lattice):
        print_section("sites 参数语义")
        print_concept("统一通过 build_sqs_config，避免手工索引展开错误")

        from sqs_pd.analysis.disorder_analyzer import analyze_structure
        from sqs_pd.core.config_builder import build_sqs_config

        structure = Structure(fcc_lattice, [{"Ni": 0.5, "Cu": 0.5}], [[0, 0, 0]])
        analysis = analyze_structure(structure)
        config = build_sqs_config(
            structure, analysis, supercell=(2, 2, 2), iterations=100
        )
        best = _optimize_from_config(config)

        print_code_example("推荐：始终用 build_sqs_config 管理 sites 映射")
        assert best is not None
