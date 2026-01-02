"""
Runner module - Orchestrates the diagnostic workflow.

The Runner is the "Executor" in the v2.0 Dynamic Contextual Architecture.
It:
- Executes state machine transitions
- Runs benchmarks per SDL specifications
- Collects telemetry
- Applies tuning with resilience
"""

from .engine import DiagnosticEngine, EngineConfig
from .state import StateMachine, State
from .benchmark import BenchmarkRunner

__all__ = [
    "DiagnosticEngine",
    "EngineConfig",
    "StateMachine",
    "State",
    "BenchmarkRunner",
]
