"""
SQS-PD: 特殊准随机结构生成工具

主要功能：
- 自动分析晶体结构的无序类型（化学无序、位置无序、混合无序）
- 基于占据数自动计算合理的超胞尺寸
- 调用 sqsgenerator 生成优化的特殊准随机结构（SQS）

快速开始：
    >>> from sqs_pd import generate_sqs
    >>> result = generate_sqs("input.cif")
    >>> if result['success']:
    ...     result['structure'].to(filename="output.cif")
    ...     print(f"Objective: {result['objective']}")

更多信息：
    - 文档: https://github.com/drzqr/SQS-PD
    - API 参考: sqs_pd.interface.api
    - 异常类型: sqs_pd.foundation.exceptions
"""

__version__ = "2.0.0"

# 导出主要公开 API
from .interface.api import (
    DisorderAnalysisResult,
    SQSResult,
    analyze_cif_disorder,
    batch_recommend_supercells,
    dry_run_recommend_supercell,
    generate_sqs,
    generate_sqs_with_report,
    recommend_supercells_with_model,
    run_topk_sqs_with_model,
)
from .foundation.exceptions import (
    SQSPDError,
    ConfigurationError,
    OccupancyError,
    SupercellError,
    StructureError,
    OptimizationError,
    DisorderAnalysisError,
    APIError,
)
from .foundation.constants import (
    DEFAULT_ITERATIONS,
    OCCUPANCY_TOLERANCE,
    FLOAT_PRECISION,
    DISORDER_TYPE_SD,
    DISORDER_TYPE_PD,
    DISORDER_TYPE_SPD,
    DISORDER_TYPE_ORDERED,
    DISORDER_TYPES_ALL,
    VACANCY_SYMBOL,
)

from .core.options import DisorderAnalysisOptions, SQSOptions

# 公开的子模块（用户可以直接导入）
from .analysis.disorder_analyzer import analyze_structure
from .core.supercell_calculator import (
    calculate_lcm_from_occupancies,
)
from .core.supercell_optimizer import (
    find_optimal_supercell,
    load_valid_supercell_specs,
    get_supercell_info_optimized,
)
from .runtime.dry_run import (
    analyze_cif_and_recommend_supercell,
    batch_analyze_cifs,
    print_analysis_summary,
)

__all__ = [
    # 版本
    "__version__",
    # 主要 API
    "generate_sqs",
    "analyze_cif_disorder",
    "batch_recommend_supercells",
    "dry_run_recommend_supercell",
    "recommend_supercells_with_model",
    "run_topk_sqs_with_model",
    "generate_sqs_with_report",
    "SQSResult",
    "DisorderAnalysisResult",
    "SQSOptions",
    "DisorderAnalysisOptions",
    # 异常类
    "SQSPDError",
    "ConfigurationError",
    "OccupancyError",
    "SupercellError",
    "StructureError",
    "OptimizationError",
    "DisorderAnalysisError",
    "APIError",
    # 常数
    "DEFAULT_ITERATIONS",
    "OCCUPANCY_TOLERANCE",
    "FLOAT_PRECISION",
    "DISORDER_TYPE_SD",
    "DISORDER_TYPE_PD",
    "DISORDER_TYPE_SPD",
    "DISORDER_TYPE_ORDERED",
    "DISORDER_TYPES_ALL",
    "VACANCY_SYMBOL",
    # 模块 API
    "analyze_structure",
    "calculate_lcm_from_occupancies",
    # 超胞优化 API
    "find_optimal_supercell",
    "load_valid_supercell_specs",
    "get_supercell_info_optimized",
    # Dry-run API
    "analyze_cif_and_recommend_supercell",
    "batch_analyze_cifs",
    "print_analysis_summary",
]
