"""
Protocol definitions for pg_diagnose.

This module contains all the JSON protocol schemas as Python dataclasses:
- ContextPacket: Runner → Agent (system/schema discovery)
- StrategyDefinition (SDL): Agent → Runner (benchmark instructions)
- BenchmarkResult: Runner → Agent (execution results)
- TuningProposal/TuningChunk: Agent → Runner (config changes)
- SessionConclusion: Agent → Runner (v2.3 - hardware saturation "game over")
- ErrorPacket/TuningErrorPacket: Runner → Agent (error recovery)
"""

from .context import (
    ContextPacket,
    SystemContext,
    RuntimeContext,
    SchemaContext,
    TableStats,
    IndexStats,
    HeuristicHints,
)
from .sdl import (
    StrategyDefinition,
    ExecutionPlan,
    BenchmarkConfig,
    TelemetryRequirement,
    SuccessCriteria,
)
from .result import (
    BenchmarkResult,
    ExecutionStep,
    BenchmarkSummary,
    CriteriaEvaluation,
    TimeSeriesData,
    HumanFeedback,  # v2.3
)
from .tuning import (
    TuningProposal,
    TuningChunk,
    Diagnosis,
    TuningResult,
    TuningSnapshot,
)
from .conclusion import (  # v2.3
    SessionConclusion,
    TuningSummary,
    HardwareSaturationAnalysis,
    ScalingRecommendation,
    BottleneckResource,
    ScalingAction,
)
from .errors import (
    ErrorPacket,
    TuningErrorPacket,
    RepairContext,
    FailureContext,
    RollbackStatus,
)

__all__ = [
    # Context
    "ContextPacket",
    "SystemContext",
    "RuntimeContext",
    "SchemaContext",
    "TableStats",
    "IndexStats",
    "HeuristicHints",
    # SDL
    "StrategyDefinition",
    "ExecutionPlan",
    "BenchmarkConfig",
    "TelemetryRequirement",
    "SuccessCriteria",
    # Result
    "BenchmarkResult",
    "ExecutionStep",
    "BenchmarkSummary",
    "CriteriaEvaluation",
    "TimeSeriesData",
    "HumanFeedback",  # v2.3
    # Tuning
    "TuningProposal",
    "TuningChunk",
    "Diagnosis",
    "TuningResult",
    "TuningSnapshot",
    # Conclusion (v2.3)
    "SessionConclusion",
    "TuningSummary",
    "HardwareSaturationAnalysis",
    "ScalingRecommendation",
    "BottleneckResource",
    "ScalingAction",
    # Errors
    "ErrorPacket",
    "TuningErrorPacket",
    "RepairContext",
    "FailureContext",
    "RollbackStatus",
]
