"""
pg_diagnose - AI-Powered PostgreSQL Diagnostic & Tuning Tool

Version 2.2 - Resilience & Self-Healing Architecture

An autonomous database tuning platform that uses LLM to dynamically
architect execution strategies based on database schema, system resources,
and inferred business logic.

Usage:
    # As a module
    python -m pg_diagnose -h localhost -d mydb

    # Programmatically
    from pg_diagnose import DiagnosticEngine, EngineConfig

    config = EngineConfig(pg_host="localhost", pg_database="mydb")
    engine = DiagnosticEngine(config=config, connection=conn)
    summary = engine.run()
"""

__version__ = "2.2.0"
__author__ = "PostgreSQL Benchmark Team"

# Main exports
from .runner.engine import DiagnosticEngine, EngineConfig
from .runner.state import StateMachine, State

# Protocol exports
from .protocol.context import ContextPacket
from .protocol.sdl import StrategyDefinition
from .protocol.result import BenchmarkResult
from .protocol.tuning import TuningProposal, TuningChunk

# Agent exports
from .agent.client import GeminiAgent, MockGeminiAgent

__all__ = [
    # Version
    "__version__",
    # Engine
    "DiagnosticEngine",
    "EngineConfig",
    "StateMachine",
    "State",
    # Protocol
    "ContextPacket",
    "StrategyDefinition",
    "BenchmarkResult",
    "TuningProposal",
    "TuningChunk",
    # Agent
    "GeminiAgent",
    "MockGeminiAgent",
]
