"""POMDP solver benchmarking suite for sequential-decision-making."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark import BenchmarkConfig, run_benchmark_suite, save_results

__all__ = [
    "BenchmarkConfig",
    "run_benchmark_suite",
    "save_results",
]


def __getattr__(name: str):
    if name in __all__:
        from .benchmark import BenchmarkConfig, run_benchmark_suite, save_results

        exports = {
            "BenchmarkConfig": BenchmarkConfig,
            "run_benchmark_suite": run_benchmark_suite,
            "save_results": save_results,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
