"""
Mock Benchmark Runner - Returns realistic benchmark results without running pgbench.

Simulates TPS progression through tuning rounds with realistic variance.
"""

import random
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .golden_data import (
    MOCK_TPS_PROGRESSION,
    get_expected_tps,
)


@dataclass
class MockBenchmarkMetrics:
    """Mock metrics matching real BenchmarkMetrics structure."""
    tps: float
    latency_avg_ms: float
    latency_p99_ms: float
    latency_stddev_ms: float
    transactions_processed: int
    connections_used: int
    duration_seconds: int


@dataclass
class MockBenchmarkResult:
    """Mock result matching real BenchmarkResult structure."""
    success: bool
    tps: float
    metrics: MockBenchmarkMetrics
    raw_output: str
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MockBenchmarkRunner:
    """
    Mock benchmark runner that returns realistic results.

    Simulates:
    - TPS progression through tuning rounds
    - Realistic variance in results
    - Latency correlation with TPS
    - Error scenarios (0 TPS, timeout, etc.)

    Usage:
        runner = MockBenchmarkRunner(scenario="balanced_tps")
        result = runner.run(duration=60, clients=32)
        print(f"TPS: {result.tps}")
    """

    def __init__(
        self,
        scenario: str = "balanced_tps",
        variance_pct: float = 5.0,
        simulate_errors: bool = False,
    ):
        """
        Initialize mock benchmark runner.

        Args:
            scenario: TPS progression scenario from MOCK_TPS_PROGRESSION
            variance_pct: Random variance in TPS (default 5%)
            simulate_errors: Whether to simulate error conditions
        """
        self.scenario = scenario
        self.variance_pct = variance_pct
        self.simulate_errors = simulate_errors
        self.run_count = 0
        self.round_num = 0
        self._force_tps: Optional[float] = None
        self._force_error: Optional[str] = None

    def set_round(self, round_num: int):
        """Set current tuning round (affects TPS returned)."""
        self.round_num = round_num

    def force_tps(self, tps: float):
        """Force specific TPS for next run (for testing)."""
        self._force_tps = tps

    def force_error(self, error: str):
        """Force error on next run (for testing)."""
        self._force_error = error

    def run(
        self,
        strategy=None,
        telemetry_collector=None,
        duration: int = 60,
        clients: int = 32,
        threads: int = 8,
        scale: int = 100,
        custom_sql: Optional[str] = None,
        **kwargs,
    ):
        """
        Run mock benchmark and return results.

        Compatible with real BenchmarkRunner interface - accepts strategy object.

        Args:
            strategy: Optional StrategyDefinition with execution plan
            telemetry_collector: Ignored in mock (for interface compatibility)
            duration: Benchmark duration (used if no strategy provided)
            clients: Number of clients (used if no strategy provided)
            threads: Number of threads
            scale: pgbench scale factor
            custom_sql: Custom SQL script (recorded but not executed)

        Returns:
            BenchmarkResult compatible with real runner
        """
        from ...protocol.result import BenchmarkResult, BenchmarkMetrics

        self.run_count += 1

        # Extract execution params from strategy if provided
        if strategy and hasattr(strategy, 'execution_plan') and strategy.execution_plan:
            plan = strategy.execution_plan
            duration = getattr(plan, 'duration_seconds', 60)
            clients = getattr(plan, 'clients', 32)
            threads = getattr(plan, 'threads', 8)
            scale = getattr(plan, 'scale', 100)

        strategy_id = getattr(strategy, 'id', 'mock-strategy') if strategy else 'mock-strategy'

        # Check for forced error
        if self._force_error:
            error = self._force_error
            self._force_error = None
            return BenchmarkResult(
                strategy_id=strategy_id,
                success=False,
                error=error,
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=0,
                metrics=BenchmarkMetrics(tps=0),
                telemetry=[],
                criteria_met={},
                raw_output=f"Error: {error}",
            )

        # Check for forced TPS
        if self._force_tps is not None:
            base_tps = self._force_tps
            self._force_tps = None
        else:
            # Get expected TPS for current round
            base_tps = get_expected_tps(self.scenario, self.round_num)

        # Handle error scenario
        if self.scenario == "error_scenario" and self.round_num == 0:
            return BenchmarkResult(
                strategy_id=strategy_id,
                success=True,  # Benchmark ran, but 0 TPS
                error=None,
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=duration,
                metrics=BenchmarkMetrics(tps=0),
                telemetry=[],
                criteria_met={},
                raw_output=self._generate_output(0, duration, clients),
            )

        # Add realistic variance
        variance = base_tps * (self.variance_pct / 100)
        actual_tps = base_tps + random.uniform(-variance, variance)
        actual_tps = max(0, actual_tps)  # Can't be negative

        # Scale TPS based on clients (more clients = higher TPS, up to a point)
        client_factor = min(clients / 32, 1.5)  # Max 50% boost from more clients
        actual_tps *= client_factor

        # Generate correlated metrics
        mock_metrics = self._generate_metrics(actual_tps, duration, clients)

        # Convert to real BenchmarkMetrics
        metrics = BenchmarkMetrics(
            tps=mock_metrics.tps,
            latency_avg_ms=mock_metrics.latency_avg_ms,
            latency_stddev_ms=mock_metrics.latency_stddev_ms,
            transactions=mock_metrics.transactions_processed,
        )

        return BenchmarkResult(
            strategy_id=strategy_id,
            success=True,
            error=None,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration_seconds=duration,
            metrics=metrics,
            telemetry=[],
            criteria_met={},
            raw_output=self._generate_output(actual_tps, duration, clients),
        )

    def run_baseline(self, **kwargs) -> MockBenchmarkResult:
        """Run baseline benchmark (round 0)."""
        self.round_num = 0
        return self.run(**kwargs)

    def run_round(self, round_num: int, **kwargs) -> MockBenchmarkResult:
        """Run benchmark for specific round."""
        self.round_num = round_num
        return self.run(**kwargs)

    def _generate_metrics(
        self,
        tps: float,
        duration: int,
        clients: int,
    ) -> MockBenchmarkMetrics:
        """Generate correlated metrics based on TPS."""
        # Latency inversely correlated with TPS
        # At 5000 TPS with 32 clients: ~6ms avg latency
        # Formula: latency ≈ (clients * 1000) / tps
        if tps > 0:
            latency_avg = (clients * 1000) / tps
            latency_avg = max(0.5, min(100, latency_avg))  # Clamp 0.5-100ms
        else:
            latency_avg = 0

        # P99 is typically 3-5x average
        latency_p99 = latency_avg * random.uniform(3, 5)

        # Stddev is typically 50-100% of average
        latency_stddev = latency_avg * random.uniform(0.5, 1.0)

        return MockBenchmarkMetrics(
            tps=tps,
            latency_avg_ms=round(latency_avg, 2),
            latency_p99_ms=round(latency_p99, 2),
            latency_stddev_ms=round(latency_stddev, 2),
            transactions_processed=int(tps * duration),
            connections_used=clients,
            duration_seconds=duration,
        )

    def _zero_metrics(self, duration: int) -> MockBenchmarkMetrics:
        """Generate zero metrics for error cases."""
        return MockBenchmarkMetrics(
            tps=0,
            latency_avg_ms=0,
            latency_p99_ms=0,
            latency_stddev_ms=0,
            transactions_processed=0,
            connections_used=0,
            duration_seconds=duration,
        )

    def _generate_output(
        self,
        tps: float,
        duration: int,
        clients: int,
    ) -> str:
        """Generate realistic pgbench-like output."""
        if tps == 0:
            return """pgbench (PostgreSQL 15.4)
starting vacuum...end.
connection to server failed: connection refused
"""

        transactions = int(tps * duration)
        latency_avg = (clients * 1000) / tps if tps > 0 else 0

        return f"""pgbench (PostgreSQL 15.4)
starting vacuum...end.
transaction type: <builtin: TPC-B (sort of)>
scaling factor: 100
query mode: simple
number of clients: {clients}
number of threads: 8
duration: {duration} s
number of transactions actually processed: {transactions}
latency average = {latency_avg:.3f} ms
tps = {tps:.6f} (including connections establishing)
tps = {tps * 1.01:.6f} (excluding connections establishing)
"""

    def reset(self):
        """Reset runner state for new test."""
        self.run_count = 0
        self.round_num = 0
        self._force_tps = None
        self._force_error = None


class MockBenchmarkFactory:
    """Factory for creating mock benchmark runners with different scenarios."""

    @staticmethod
    def create(scenario: str = "balanced_tps") -> MockBenchmarkRunner:
        """Create a mock benchmark runner for a specific scenario."""
        return MockBenchmarkRunner(scenario=scenario)

    @staticmethod
    def create_with_progression(
        tps_values: list,
    ) -> MockBenchmarkRunner:
        """
        Create runner with custom TPS progression.

        Args:
            tps_values: List of TPS values for each run
                        [baseline, round1, round2, ...]
        """
        runner = MockBenchmarkRunner()

        # Override get_expected_tps behavior
        original_run = runner.run

        def custom_run(*args, **kwargs):
            idx = min(runner.run_count, len(tps_values) - 1)
            runner._force_tps = tps_values[idx]
            return original_run(*args, **kwargs)

        runner.run = custom_run
        return runner

    @staticmethod
    def create_for_test(test_id: str) -> MockBenchmarkRunner:
        """
        Create benchmark runner based on test ID.

        Maps test IDs to appropriate scenarios:
            E3.1 (Happy Path) -> balanced_tps
            E3.3 (Error Recovery) -> error_scenario
            etc.
        """
        mapping = {
            "E3.1": "balanced_tps",
            "E3.2": "balanced_tps",
            "E3.3": "error_scenario",
            "E3.4": "balanced_tps",
            "E3.5": "balanced_tps",
            "E3.6": "balanced_tps",
            "E3.7": "balanced_tps",
        }

        scenario = mapping.get(test_id, "balanced_tps")
        return MockBenchmarkRunner(scenario=scenario)
