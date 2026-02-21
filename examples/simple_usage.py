"""
SQS-PD 极简使用示例
====================
仅保留两个常用场景：批量处理与单文件处理
"""

import os
import warnings
from pathlib import Path
from sqs_pd import (
    generate_sqs,
    SQSOptions,
    dry_run_recommend_supercell,
    recommend_supercells_with_model,
    run_topk_sqs_with_model,
)

# 抑制 pymatgen 的警告（可选）
warnings.filterwarnings("ignore", category=EncodingWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=FutureWarning, module="pymatgen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
output_dir = PROJECT_ROOT / "data" / "output"
input_dir = PROJECT_ROOT / "data" / "input"
os.makedirs(output_dir, exist_ok=True)

print("SQS-PD 极简使用示例")
print("=" * 70)

# ============================================================================
# 场景1: 批量处理 demo_sd 和 demo_pd（full 输出）
# ============================================================================
print("\n场景1: 批量处理")
print("-" * 70)
input_files = [
    input_dir / "demo_sd.cif",
    input_dir / "demo_pd.cif",
]

results = []
for input_file in input_files:
    name = Path(input_file).stem
    output_subdir = output_dir / f"batch_{name}__auto"
    os.makedirs(output_subdir, exist_ok=True)
    result = generate_sqs(
        str(input_file),
        options=SQSOptions(
            output_file=str(output_subdir / f"{name}__auto.cif"),
            log_file=str(output_subdir / f"{name}__auto.json"),
            iterations=100000,
            console_log=False,
            include_analysis_details=True,
        ),
    )
    results.append((name, result.objective, result.success))

print("批量优化结果:")
for name, obj, success in results:
    status = "✓" if success else "✗"
    obj_str = f"{obj:.6f}" if obj is not None else "N/A"
    print(f"  {status} {name}: objective={obj_str}")

# ============================================================================
# 场景2: 单文件处理 100193.cif（full 输出）
# ============================================================================
print("\n场景2: 单文件处理")
print("-" * 70)
output_subdir = output_dir / "single_100193__auto"
os.makedirs(output_subdir, exist_ok=True)

result = generate_sqs(
    str(input_dir / "100193.cif"),
    options=SQSOptions(
        output_file=str(output_subdir / "100193__auto.cif"),
        log_file=str(output_subdir / "100193__auto.json"),
        iterations=100000,
        console_log=True,
        include_analysis_details=True,
    ),
)
obj_str = f"{result.objective:.6f}" if result.objective is not None else "N/A"
print(f"✓ 优化完成: objective={obj_str}")
print(f"  生成目录: {output_subdir}")

# ============================================================================
# 场景3: 15143.cif 自动推荐超胞规格
# ============================================================================
print("\n场景3: 自动推荐超胞规格")
print("-" * 70)

demo_cif = input_dir / "15143.cif"
output_subdir = output_dir / "single_15143__model_top5_validation"
os.makedirs(output_subdir, exist_ok=True)

# 3.1 dry-run：解析合法候选（不执行 SQS），并输出“最小 RSS”对应的全部规格
dry_run_result = dry_run_recommend_supercell(str(demo_cif))
all_candidates = dry_run_result.get("all_candidates", [])
min_rss_candidates = []
if all_candidates:
    min_rss = min(float(row["rss"]) for row in all_candidates)
    min_rss_candidates = [
        row for row in all_candidates if abs(float(row["rss"]) - min_rss) <= 1e-12
    ]

print("dry-run 结果:")
print(f"  optimization_success={dry_run_result.get('optimization_success')}")
print(f"  num_candidates={dry_run_result.get('num_candidates')}")
print(f"  best_by_rss={dry_run_result.get('recommended_supercell')}")

print("最小 RSS 对应规格:")
if min_rss_candidates:
    min_rss_supercells = [tuple(row["supercell"]) for row in min_rss_candidates]
    print(f"  {min_rss_supercells}")
else:
    print("  (无候选)")

# 3.2 模型排序 + 实际运行：先执行 top-5，再合并输出对比信息
model_rank_result = recommend_supercells_with_model(
    str(demo_cif),
    top_k=5,
)
ranked_candidates = model_rank_result.get("ranked_candidates", [])

print("模型排序与实跑验证:")
print(f"  success={model_rank_result.get('success')}")
print(f"  recommended={model_rank_result.get('recommended_supercell')}")
print("  note: score 仅用于同一 CIF 内相对排序；objective 越低越好")

print("\n执行模型 top-5 实际优化...")
run_topk_result = run_topk_sqs_with_model(
    str(demo_cif),
    top_k=5,
    max_error=0.0005,
    iterations=100000,
    artifact_policy="best",
    output_dir=output_dir,
    output_folder_name="single_15143__model_top5_validation",
    save_comparison_csv=True,
    comparison_csv_name="top5_model_vs_actual.csv",
    objective_quantum=0.1,
    console_log=False,
)

successful_rows = [
    item
    for item in (run_topk_result.get("rows") or [])
    if item.get("success") and item.get("objective") is not None
]
successful_rows.sort(
    key=lambda item: (
        float(item.get("objective_q", 9999.0)),
        float(item.get("objective", 9999.0)),
    )
)

print("\n真实 objective 排名 (越低越好):")
if successful_rows:
    for item in successful_rows:
        print(
            f"  actual_rank={item['actual_rank']}, model_rank={item['model_rank']}, "
            f"supercell={item['supercell']}, objective_q={item['objective_q']:.1f}, "
            f"objective_raw={item['objective']:.6f}, "
            f"model_score={item['model_score']:.6f}"
        )

    best = run_topk_result.get("best") or successful_rows[0]
    exported = run_topk_result.get("exported") or {}
    best_exported = exported.get("best") or {}

    print("\n保留最优超胞结果:")
    print(f"  supercell={best['supercell']}, objective={best['objective']:.6f}")
    if best_exported.get("export_objective") is not None:
        print(f"  exported_objective={best_exported['export_objective']:.6f}")
    print(f"  cif={best_exported.get('cif')}")
    print(f"  log={best_exported.get('log_json')}")
    print(f"  sqsgenerator_config={best_exported.get('sqsgenerator_config_json')}")
    print(f"  topn_comparison={exported.get('comparison_csv')}")
else:
    print("  无成功结果，无法生成真实 objective 排名")
    print(
        f"  topn_comparison={(run_topk_result.get('exported') or {}).get('comparison_csv')}"
    )

print("\n" + "=" * 70)
print("✓ 所有场景演示完成！")
