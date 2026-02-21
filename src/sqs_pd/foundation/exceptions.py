"""
统一的异常类定义
=====================================
为 SQS-PD 项目定义自定义异常，便于错误处理和调试

异常层级：
    SQSPDError (基类)
    ├── ConfigurationError - 配置相关错误
    ├── OccupancyError - 占据数相关错误
    ├── SupercellError - 超胞计算错误
    ├── StructureError - 结构分析错误
    └── OptimizationError - SQS优化错误
"""


class SQSPDError(Exception):
    """SQS-PD 项目的基础异常类

    所有其他异常都应该继承此类，便于用户捕获所有项目相关的错误。

    Example:
        >>> try:
        ...     result = generate_sqs(...)
        >>> except SQSPDError as e:
        ...     print(f"SQS-PD error: {e}")
    """

    pass


class ConfigurationError(SQSPDError):
    """配置相关错误

    当配置格式不正确、必需字段缺失或值无效时抛出。
    """

    pass


class OccupancyError(SQSPDError):
    """占据数相关错误

    当占据数计算出错或不满足数学约束时抛出。

    Examples:
        - 占据数分母无法计算
        - LCM 计算异常
        - 超胞尺寸与占据数不匹配
    """

    pass


class SupercellError(SQSPDError):
    """超胞计算错误

    当超胞尺寸计算或验证失败时抛出。

    Examples:
        - 超胞尺寸不是 LCM 的倍数
        - 无法计算合理的超胞尺寸
        - 超胞尺寸超出允许范围
    """

    pass


class StructureError(SQSPDError):
    """结构分析错误

    当分析晶体结构时出现问题。

    Examples:
        - 无法加载 CIF 文件
        - 结构格式无效
        - 无序分类异常
    """

    pass


class OptimizationError(SQSPDError):
    """SQS 优化错误

    当调用 sqsgenerator 进行优化时出现问题。

    Examples:
        - 配置解析失败
        - 优化过程崩溃
        - 返回结果无效
    """

    pass


class DisorderAnalysisError(SQSPDError):
    """无序分析错误

    当分析结构的无序类型时出现问题。
    """

    pass


class APIError(SQSPDError):
    """高层 API 错误

    当使用公开 API（api.py）时出现问题。
    """

    pass
