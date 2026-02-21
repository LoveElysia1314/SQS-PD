"""Interface subpackage.

This package keeps __init__ lightweight to avoid import cycles.
Use explicit imports from `sqs_pd.interface.api` or `sqs_pd.interface.cli`.
"""

__all__ = ["api", "cli"]
