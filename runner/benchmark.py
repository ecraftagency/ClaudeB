"""
BenchmarkRunner - Executes benchmarks per SDL specifications.

Supports:
- Standard pgbench
- Custom SQL scripts
- Telemetry collection during execution
"""

import subprocess
import tempfile
import re
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from ..protocol.sdl import StrategyDefinition, ExecutionPlan
from ..protocol.result import BenchmarkResult, BenchmarkMetrics


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_database: str = "postgres"
    pg_password: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_user: str = "ubuntu"
    output_dir: Path = Path("/tmp/pg_diagnose_benchmark")


class BenchmarkRunner:
    """
    Executes pgbench benchmarks based on SDL specifications.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        database: str = "postgres",
        password: Optional[str] = None,
    ):
        if config:
            self.config = config
        else:
            self.config = BenchmarkConfig(
                pg_host=host,
                pg_port=port,
                pg_user=user,
                pg_database=database,
                pg_password=password,
            )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        strategy: StrategyDefinition,
        telemetry_collector=None,
    ) -> BenchmarkResult:
        """
        Execute benchmark per strategy specification.

        Args:
            strategy: StrategyDefinition with execution plan
            telemetry_collector: Optional TelemetryCollector for metrics

        Returns:
            BenchmarkResult with metrics and telemetry
        """
        if not strategy.execution_plan:
            raise ValueError("Strategy has no execution_plan")

        plan = strategy.execution_plan

        # Prepare custom SQL if needed
        custom_sql_file = None
        if plan.benchmark_type == "custom" and plan.custom_sql:
            custom_sql_file = self._write_custom_sql(plan.custom_sql)

        # Build pgbench command
        cmd = self._build_pgbench_command(plan, custom_sql_file)

        # Start telemetry if provided
        if telemetry_collector:
            telemetry_collector.start()

        # Execute benchmark
        start_time = datetime.utcnow()
        try:
            result = self._execute_pgbench(cmd, plan.duration_seconds + plan.warmup_seconds + 30)
            success = True
            error = None
        except Exception as e:
            result = {"raw_output": "", "error": str(e)}
            success = False
            error = str(e)

        end_time = datetime.utcnow()
        duration_sec = (end_time - start_time).total_seconds()

        # Stop telemetry
        telemetry_points = []
        if telemetry_collector:
            samples = telemetry_collector.stop()
            telemetry_points = telemetry_collector.to_telemetry_points()

        # Parse results
        metrics = self._parse_pgbench_output(result.get("raw_output", ""))

        # Check success criteria
        criteria_met = self._check_criteria(metrics, strategy.success_criteria)

        # Cleanup
        if custom_sql_file:
            try:
                Path(custom_sql_file).unlink()
            except Exception:
                pass

        return BenchmarkResult(
            strategy_id=strategy.id,
            success=success,
            error=error,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration_sec,
            metrics=metrics,
            telemetry=telemetry_points,
            criteria_met=criteria_met,
            raw_output=result.get("raw_output", ""),
        )

    def _build_pgbench_command(
        self,
        plan: ExecutionPlan,
        custom_sql_file: Optional[str] = None,
    ) -> List[str]:
        """Build pgbench command from execution plan."""
        cmd = ["pgbench"]

        # Connection parameters
        cmd.extend(["-h", self.config.pg_host])
        cmd.extend(["-p", str(self.config.pg_port)])
        cmd.extend(["-U", self.config.pg_user])

        # Benchmark parameters
        cmd.extend(["-c", str(plan.clients)])
        # Threads: use specified value, or default to min(clients, 4)
        threads = plan.threads if plan.threads else min(plan.clients, 4)
        cmd.extend(["-j", str(threads)])
        cmd.extend(["-T", str(plan.duration_seconds)])

        # Custom SQL or built-in
        if custom_sql_file:
            cmd.extend(["-f", custom_sql_file])
        else:
            # Use scale for initialization info
            cmd.extend(["-s", str(plan.scale)])

        # Progress reporting
        cmd.extend(["-P", "5"])  # Report every 5 seconds

        # Database
        cmd.append(self.config.pg_database)

        return cmd

    def _write_custom_sql(self, sql: str) -> str:
        """Write custom SQL to temporary file."""
        fd, path = tempfile.mkstemp(suffix=".sql", prefix="pgbench_")
        with os.fdopen(fd, 'w') as f:
            f.write(sql)
        return path

    def _execute_pgbench(self, cmd: List[str], timeout: int) -> Dict[str, Any]:
        """Execute pgbench command."""
        env = os.environ.copy()
        if self.config.pg_password:
            env['PGPASSWORD'] = self.config.pg_password

        if self.config.ssh_host:
            # Run remotely
            ssh_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                f"{self.config.ssh_user}@{self.config.ssh_host}",
                ' '.join(cmd)
            ]
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

        return {
            "raw_output": result.stdout + "\n" + result.stderr,
            "returncode": result.returncode,
        }

    def _parse_pgbench_output(self, output: str) -> BenchmarkMetrics:
        """Parse pgbench output into metrics."""
        metrics = BenchmarkMetrics()

        # Parse TPS
        # "tps = 1234.56789 (without initial connection time)"
        tps_match = re.search(r'tps\s*=\s*([\d.]+)', output)
        if tps_match:
            metrics.tps = float(tps_match.group(1))

        # Parse latency
        # "latency average = 12.345 ms"
        latency_avg_match = re.search(r'latency average\s*=\s*([\d.]+)\s*ms', output)
        if latency_avg_match:
            metrics.latency_avg_ms = float(latency_avg_match.group(1))

        # "latency stddev = 5.678 ms"
        latency_std_match = re.search(r'latency stddev\s*=\s*([\d.]+)\s*ms', output)
        if latency_std_match:
            metrics.latency_stddev_ms = float(latency_std_match.group(1))

        # Parse transactions
        # "number of transactions actually processed: 12345"
        txn_match = re.search(r'number of transactions.*:\s*(\d+)', output)
        if txn_match:
            metrics.transactions = int(txn_match.group(1))

        # Parse connection time
        # "connection time = 1.234 ms"
        conn_time_match = re.search(r'connection time\s*=\s*([\d.]+)\s*ms', output)
        if conn_time_match:
            metrics.connection_time_ms = float(conn_time_match.group(1))

        # Parse progress lines for min/max latency
        # "progress: 5.0 s, 1234.5 tps, lat 8.123 ms stddev 2.345"
        progress_lines = re.findall(
            r'progress:.*lat\s+([\d.]+)\s*ms',
            output
        )
        if progress_lines:
            latencies = [float(l) for l in progress_lines]
            metrics.latency_min_ms = min(latencies)
            metrics.latency_max_ms = max(latencies)

        return metrics

    def _check_criteria(
        self,
        metrics: BenchmarkMetrics,
        criteria,
    ) -> Dict[str, bool]:
        """Check if metrics meet success criteria."""
        if not criteria:
            return {}

        results = {}

        if criteria.target_tps and metrics.tps:
            results['target_tps'] = metrics.tps >= criteria.target_tps

        if criteria.max_latency_p99_ms and metrics.latency_max_ms:
            # Using max as proxy for p99 when exact percentile not available
            results['max_latency_p99_ms'] = metrics.latency_max_ms <= criteria.max_latency_p99_ms

        if criteria.min_cache_hit_ratio:
            # Would need to get this from pgstat telemetry
            pass

        return results

    def initialize_pgbench(self, scale: int = 100) -> bool:
        """Initialize pgbench tables at specified scale."""
        cmd = [
            "pgbench",
            "-i",  # Initialize
            "-s", str(scale),
            "-h", self.config.pg_host,
            "-p", str(self.config.pg_port),
            "-U", self.config.pg_user,
            self.config.pg_database,
        ]

        env = os.environ.copy()
        if self.config.pg_password:
            env['PGPASSWORD'] = self.config.pg_password

        try:
            if self.config.ssh_host:
                ssh_cmd = [
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    f"{self.config.ssh_user}@{self.config.ssh_host}",
                    ' '.join(cmd)
                ]
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, env=env)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            return result.returncode == 0
        except Exception:
            return False
