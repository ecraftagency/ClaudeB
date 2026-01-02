"""
BenchmarkResult - Runner → Agent protocol.

Contains benchmark execution results, telemetry, and criteria evaluation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class BenchmarkMetrics:
    """Simple metrics from pgbench output."""
    tps: float = 0.0
    latency_avg_ms: float = 0.0
    latency_stddev_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    transactions: int = 0
    connection_time_ms: float = 0.0


@dataclass
class TelemetryPoint:
    """A single telemetry data point with timestamp."""
    timestamp: str = ""  # Relative: T+0ms, T+1000ms
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Summary statistics from a benchmark run."""
    tps: float
    latency_avg_ms: float
    latency_stddev_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    transactions_committed: int
    transactions_failed: int
    error_rate_pct: float


@dataclass
class ExecutionStep:
    """Results from a single client-level benchmark step."""
    clients: int
    threads: int
    duration_sec: int
    summary: BenchmarkSummary


@dataclass
class PgStatsDatabase:
    """PostgreSQL pg_stat_database delta."""
    xact_commit: int = 0
    xact_rollback: int = 0
    deadlocks: int = 0
    blks_hit: int = 0
    blks_read: int = 0
    temp_files: int = 0
    temp_bytes: int = 0


@dataclass
class PgStatsBgwriter:
    """PostgreSQL pg_stat_bgwriter delta."""
    checkpoints_timed: int = 0
    checkpoints_req: int = 0
    buffers_checkpoint: int = 0
    buffers_backend: int = 0


@dataclass
class PgStatsWal:
    """PostgreSQL pg_stat_wal delta."""
    wal_records: int = 0
    wal_bytes: int = 0


@dataclass
class PgStatsDelta:
    """Combined PostgreSQL stats delta (start → end)."""
    database: PgStatsDatabase = field(default_factory=PgStatsDatabase)
    bgwriter: PgStatsBgwriter = field(default_factory=PgStatsBgwriter)
    wal: Optional[PgStatsWal] = None


@dataclass
class SystemMetricsSummary:
    """System metrics summary during benchmark."""
    cpu_user_avg_pct: float = 0.0
    cpu_system_avg_pct: float = 0.0
    cpu_iowait_avg_pct: float = 0.0
    disk_read_mb_per_sec: float = 0.0
    disk_write_mb_per_sec: float = 0.0
    disk_iops_read: float = 0.0
    disk_iops_write: float = 0.0


@dataclass
class ComparisonResult:
    """Comparison against baseline or previous iteration."""
    primary_kpi_delta_pct: float
    status: str  # IMPROVED, DEGRADED, STABLE


@dataclass
class ConstraintResult:
    """Result of checking a single constraint."""
    met: bool
    actual: float
    threshold: float


@dataclass
class CriteriaEvaluation:
    """Success criteria evaluation (computed by Runner)."""
    primary_kpi_value: float
    primary_kpi_met: bool
    constraints_met: Dict[str, ConstraintResult]
    overall_status: str  # PASSED, FAILED, DEGRADED

    vs_baseline: Optional[ComparisonResult] = None
    vs_previous: Optional[ComparisonResult] = None


@dataclass
class BenchmarkTiming:
    """Time alignment metadata."""
    benchmark_start: str
    benchmark_end: str
    duration_ms: int
    telemetry_base: str  # Same as benchmark_start


@dataclass
class TimeSeriesData:
    """Time series telemetry data with relative timestamps."""
    timestamps: List[str]  # Relative: T+0ms, T+1000ms, etc.
    values: List[Dict[str, Any]]


@dataclass
class BenchmarkLogs:
    """Raw logs for debugging."""
    pgbench_stdout: str = ""
    pgbench_stderr: str = ""
    setup_log: Optional[str] = None
    teardown_log: Optional[str] = None


@dataclass
class HumanFeedback:
    """
    DBA feedback injected into benchmark results for AI context.

    This allows the DBA to provide observations, corrections, or new directions
    that the AI should consider when analyzing results.
    """
    feedback_text: str = ""
    intent: str = "CLARIFICATION"  # CORRECTION, NEW_DIRECTION, CLARIFICATION
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class BenchmarkResult:
    """
    Benchmark result from Runner (simplified for pgbench).

    Used by BenchmarkRunner to return results.
    """
    strategy_id: str = ""
    success: bool = False
    error: Optional[str] = None
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    metrics: Optional[BenchmarkMetrics] = None
    telemetry: List[TelemetryPoint] = field(default_factory=list)
    criteria_met: Dict[str, bool] = field(default_factory=dict)
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class BenchmarkResultFull:
    """
    Complete benchmark result sent from Runner to Agent (v2 protocol).

    More detailed than BenchmarkResult, used for AI analysis.
    """
    protocol_version: str = "v2"
    strategy_id: str = ""
    session_id: str = ""
    executed_at: str = ""

    # Per-client-level results
    execution_steps: List[ExecutionStep] = field(default_factory=list)

    # Timing
    timing: Optional[BenchmarkTiming] = None

    # Collected telemetry
    telemetry: Dict[str, TimeSeriesData] = field(default_factory=dict)

    # PostgreSQL stats delta
    pg_stats_delta: Optional[PgStatsDelta] = None

    # System metrics summary
    system_metrics_summary: Optional[SystemMetricsSummary] = None

    # Criteria evaluation
    criteria_evaluation: Optional[CriteriaEvaluation] = None

    # Raw logs
    logs: Optional[BenchmarkLogs] = None

    # Human feedback (v2.3) - DBA observations/corrections
    human_feedback: Optional[HumanFeedback] = None

    def __post_init__(self):
        if not self.executed_at:
            self.executed_at = datetime.utcnow().isoformat() + "Z"

    def get_best_tps(self) -> float:
        """Get the highest TPS achieved across all steps."""
        if not self.execution_steps:
            return 0.0
        return max(step.summary.tps for step in self.execution_steps)

    def get_final_tps(self) -> float:
        """Get TPS from the last (highest concurrency) step."""
        if not self.execution_steps:
            return 0.0
        return self.execution_steps[-1].summary.tps
