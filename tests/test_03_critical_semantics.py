"""测试03: 项目 API 工作流与自动超胞策略。

学习目标:
1. 验证 generate_sqs 主 API 的常见路径
2. 保留自动超胞策略的 model/fallback 语义
3. 合并原先分散的策略验证脚本
"""

from pathlib import Path

import pytest

from conftest import print_section, print_concept, print_code_example
from sqs_pd import SQSOptions, batch_recommend_supercells, generate_sqs
from sqs_pd.interface import api as api_mod


class TestProjectApiWorkflows:
    def test_generate_sqs_basic(self, demo_sd_cif):
        print_section("generate_sqs 基本链路")
        print_concept("一行 API 完成分析、超胞选择与优化")

        result = generate_sqs(str(demo_sd_cif), options=SQSOptions(iterations=5000))
        print_code_example(
            f"success={result.success}, objective={getattr(result, 'objective', None)}"
        )

        assert result.success
        assert result.structure is not None

    @pytest.mark.parametrize("input_file", ["demo_sd.cif", "demo_pd.cif"])
    def test_complete_workflow_for_sd_and_pd(self, input_file):
        print_section("端到端工作流")
        print_concept("SD/PD 输入都应走通统一入口")

        input_path = Path(__file__).parent.parent / "data" / "input" / input_file
        result = generate_sqs(str(input_path), options=SQSOptions(iterations=5000))
        assert result.success

    def test_batch_recommend_supercells_api(self, demo_sd_cif, demo_pd_cif):
        print_section("批量推荐 API")
        print_concept("批量 dry-run 返回结构化推荐字段")

        rows = batch_recommend_supercells([str(demo_sd_cif), str(demo_pd_cif)])
        assert len(rows) == 2
        assert all("cif_file" in row for row in rows)
        assert all("optimization_success" in row for row in rows)


class TestAutoSupercellPolicy:
    def test_auto_prefers_model_top1(self, monkeypatch):
        captured = {}

        def fake_model_recommend(
            *,
            cif_file,
            top_k,
            max_error=0.0005,
            model_path=None,
            model_config_path=None,
        ):
            return {
                "success": True,
                "recommended_supercell": (6, 4, 3),
                "ranked_candidates": [
                    {"rank": 1, "supercell": (6, 4, 3), "model_score": 0.9}
                ],
            }

        def fake_dry_run(*args, **kwargs):
            raise AssertionError("model 成功时不应进入 dry-run fallback")

        def fake_generate(**kwargs):
            captured["supercell"] = kwargs.get("supercell")
            return {
                "success": True,
                "structure": None,
                "objective": 0.123,
                "time": 0.01,
                "supercell_used": kwargs.get("supercell"),
                "disorder_types": ["SD"],
            }

        monkeypatch.setattr(
            api_mod, "recommend_supercells_by_model", fake_model_recommend
        )
        monkeypatch.setattr(
            api_mod, "analyze_cif_and_recommend_supercell", fake_dry_run
        )
        monkeypatch.setattr(api_mod.sqs_orchestrator, "generate_sqs", fake_generate)

        result = api_mod.generate_sqs("dummy.cif")
        assert result.success
        assert captured["supercell"] == (6, 4, 3)

    def test_auto_fallback_uses_min_rss_then_smallest_size(self, monkeypatch):
        captured = {}

        def fake_model_recommend(
            *,
            cif_file,
            top_k,
            max_error=0.0005,
            model_path=None,
            model_config_path=None,
        ):
            raise RuntimeError("model unavailable")

        def fake_dry_run(*, cif_file, max_error=0.0005, verbose=False):
            return {
                "optimization_success": True,
                "message": "ok",
                "recommended_supercell": (5, 4, 3),
                "all_candidates": [
                    {"supercell": (5, 4, 3), "size": 60, "rss": 1.0e-4},
                    {"supercell": (6, 3, 3), "size": 54, "rss": 1.0e-4},
                    {"supercell": (4, 4, 4), "size": 64, "rss": 2.0e-4},
                ],
            }

        def fake_generate(**kwargs):
            captured["supercell"] = kwargs.get("supercell")
            return {
                "success": True,
                "structure": None,
                "objective": 0.456,
                "time": 0.02,
                "supercell_used": kwargs.get("supercell"),
                "disorder_types": ["PD"],
            }

        monkeypatch.setattr(
            api_mod, "recommend_supercells_by_model", fake_model_recommend
        )
        monkeypatch.setattr(
            api_mod, "analyze_cif_and_recommend_supercell", fake_dry_run
        )
        monkeypatch.setattr(api_mod.sqs_orchestrator, "generate_sqs", fake_generate)

        result = api_mod.generate_sqs("dummy.cif")
        assert result.success
        assert captured["supercell"] == (6, 3, 3)


class TestTopkRunApi:
    def test_topk_csv_export_only_when_n_gt_1(self, monkeypatch, tmp_path):
        def fake_model_recommend(
            *,
            cif_file,
            top_k,
            max_error=0.0005,
            model_path=None,
            model_config_path=None,
        ):
            return {
                "success": True,
                "ranked_candidates": [
                    {
                        "rank": 1,
                        "supercell": (5, 5, 4),
                        "size": 100,
                        "model_score": 2.0,
                        "rss": 1e-4,
                    },
                    {
                        "rank": 2,
                        "supercell": (8, 5, 5),
                        "size": 200,
                        "model_score": 1.0,
                        "rss": 1e-4,
                    },
                ],
            }

        def fake_generate(input_file, options=None):
            sc = tuple(options.supercell)
            objective_map = {(5, 5, 4): 1.2, (8, 5, 5): 1.4}
            return api_mod.SQSResult(
                success=True,
                objective=objective_map[sc],
                supercell_used=sc,
            )

        monkeypatch.setattr(
            api_mod, "recommend_supercells_by_model", fake_model_recommend
        )
        monkeypatch.setattr(api_mod, "generate_sqs", fake_generate)

        one = api_mod.run_topk_sqs_with_model(
            "dummy.cif",
            top_k=1,
            artifact_policy="none",
            save_comparison_csv=True,
            output_dir=str(tmp_path),
        )
        assert one["exported"]["comparison_csv"] is None

        many = api_mod.run_topk_sqs_with_model(
            "dummy.cif",
            top_k=2,
            artifact_policy="none",
            save_comparison_csv=True,
            output_dir=str(tmp_path),
        )
        assert many["exported"]["comparison_csv"] is not None
        assert Path(many["exported"]["comparison_csv"]).exists()

    def test_topk_best_policy_exports_only_best(self, monkeypatch, tmp_path):
        calls = []

        def fake_model_recommend(
            *,
            cif_file,
            top_k,
            max_error=0.0005,
            model_path=None,
            model_config_path=None,
        ):
            return {
                "success": True,
                "ranked_candidates": [
                    {
                        "rank": 1,
                        "supercell": (10, 10, 9),
                        "size": 900,
                        "model_score": 4.0,
                        "rss": 1e-4,
                    },
                    {
                        "rank": 2,
                        "supercell": (5, 5, 4),
                        "size": 100,
                        "model_score": 3.0,
                        "rss": 1e-4,
                    },
                ],
            }

        def fake_generate(input_file, options=None):
            sc = tuple(options.supercell)
            calls.append(
                {
                    "supercell": sc,
                    "output_file": options.output_file,
                    "log_file": options.log_file,
                }
            )
            objective_map = {(10, 10, 9): 2.6, (5, 5, 4): 1.4}
            return api_mod.SQSResult(
                success=True,
                objective=objective_map[sc],
                supercell_used=sc,
            )

        monkeypatch.setattr(
            api_mod, "recommend_supercells_by_model", fake_model_recommend
        )
        monkeypatch.setattr(api_mod, "generate_sqs", fake_generate)

        result = api_mod.run_topk_sqs_with_model(
            "dummy.cif",
            top_k=2,
            artifact_policy="best",
            save_comparison_csv=False,
            output_dir=str(tmp_path),
        )

        assert result["success"]
        assert result["best"]["supercell"] == (5, 5, 4)
        assert len(calls) == 3  # 2次评估 + 1次最佳导出
        assert calls[0]["output_file"] is None and calls[1]["output_file"] is None
        assert calls[2]["output_file"] is not None and calls[2]["log_file"] is not None


class TestOutputPolicyNormalization:
    @pytest.mark.parametrize(
        "profile, expected",
        [
            (None, (True, True, True)),
            ("silent", (False, False, False)),
            ("none", (False, False, False)),
            ("cif-only", (True, False, False)),
            ("logs", (True, True, False)),
            ("full", (True, True, True)),
        ],
    )
    def test_output_profile_aliases(self, profile, expected):
        assert (
            api_mod._resolve_output_profile(
                output_profile=profile,
                save_json=True,
                save_txt_report=True,
            )
            == expected
        )

    def test_output_profile_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown output_profile"):
            api_mod._resolve_output_profile(
                output_profile="bad-profile",
                save_json=True,
                save_txt_report=True,
            )

    def test_artifact_policy_invalid_raises(self):
        with pytest.raises(ValueError, match="artifact_policy must be one of"):
            api_mod.run_topk_sqs_with_model(
                "dummy.cif",
                top_k=1,
                artifact_policy="invalid-policy",
            )


class TestOutputFolderTemplate:
    def test_generate_with_report_output_folder_template(self, monkeypatch, tmp_path):
        captured = {}

        def fake_generate(input_file, options=None):
            captured["output_file"] = options.output_file
            captured["log_file"] = options.log_file
            return api_mod.SQSResult(success=True, objective=0.1)

        monkeypatch.setattr(api_mod, "generate_sqs", fake_generate)

        api_mod.generate_sqs_with_report(
            "demo_sd.cif",
            output_dir=str(tmp_path),
            output_folder_name="case_{stem}_{supercell}",
            supercell=(4, 3, 2),
            output_profile="logs",
            save_txt_report=False,
        )

        assert "case_demo_sd_4x3x2" in str(captured["log_file"])
        assert str(captured["output_file"]).endswith(".cif")

    def test_topk_output_folder_template(self, monkeypatch, tmp_path):
        def fake_model_recommend(
            *,
            cif_file,
            top_k,
            max_error=0.0005,
            model_path=None,
            model_config_path=None,
        ):
            return {
                "success": True,
                "ranked_candidates": [
                    {
                        "rank": 1,
                        "supercell": (5, 5, 4),
                        "size": 100,
                        "model_score": 1.0,
                        "rss": 1e-4,
                    }
                ],
            }

        def fake_generate(input_file, options=None):
            return api_mod.SQSResult(
                success=True,
                objective=1.0,
                supercell_used=tuple(options.supercell),
            )

        monkeypatch.setattr(
            api_mod, "recommend_supercells_by_model", fake_model_recommend
        )
        monkeypatch.setattr(api_mod, "generate_sqs", fake_generate)

        result = api_mod.run_topk_sqs_with_model(
            "demo_sd.cif",
            top_k=3,
            artifact_policy="none",
            output_dir=str(tmp_path),
            output_folder_name="rank_{stem}_k{top_k}",
        )

        assert result["success"]
        assert Path(result["exported"]["output_dir"]).name == "rank_demo_sd_k3"
