"""
Mock components for testing pg_diagnose.

These mocks use realistic PostgreSQL tuning data (golden_data.py)
to simulate AI responses and benchmark results without making
actual API calls or running real benchmarks.
"""

from .golden_data import (
    STRATEGIES,
    TUNING_ROUNDS,
    BENCHMARK_CONFIGS,
    OS_TUNING,
    MOCK_TPS_PROGRESSION,
    MOCK_FIRST_SIGHT,
    SNAPSHOT_TEST_DATA,
    get_strategies_for_workload,
    get_tuning_for_round,
    get_benchmark_for_workload,
    get_expected_tps,
    generate_analysis_response,
    validate_golden_data,
    WorkloadType,
    SystemProfile,
)

from .mock_agent import MockGeminiAgent
from .mock_benchmark import MockBenchmarkRunner
from .mock_snapshot import (
    MockSnapshotCapture,
    MockSnapshotRestore,
    MockSnapshotManager,
    MockSnapshotFactory,
)

__all__ = [
    # Agent and benchmark mocks
    'MockGeminiAgent',
    'MockBenchmarkRunner',
    # Snapshot mocks
    'MockSnapshotCapture',
    'MockSnapshotRestore',
    'MockSnapshotManager',
    'MockSnapshotFactory',
    # Golden data
    'STRATEGIES',
    'TUNING_ROUNDS',
    'BENCHMARK_CONFIGS',
    'SNAPSHOT_TEST_DATA',
    'WorkloadType',
    'SystemProfile',
    # Validation
    'validate_golden_data',
]
