"""
Mock Gemini Agent - Returns realistic PostgreSQL tuning recommendations.

This mock agent uses the golden_data module to return battle-tested
PostgreSQL configurations without making actual API calls to Gemini.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .golden_data import (
    STRATEGIES,
    MOCK_FIRST_SIGHT,
    MOCK_TPS_PROGRESSION,
    get_strategies_for_workload,
    get_tuning_for_round,
    get_benchmark_for_workload,
    get_expected_tps,
    generate_analysis_response,
    WorkloadType,
)


@dataclass
class MockExecutionPlan:
    """Mock execution plan matching real ExecutionPlan structure."""
    benchmark_type: str = "pgbench"
    scale: int = 100
    clients: int = 32
    threads: int = 8
    duration_seconds: int = 60
    custom_sql: Optional[str] = None


@dataclass
class MockStrategy:
    """Mock strategy matching real Strategy structure."""
    name: str
    hypothesis: str
    execution_plan: MockExecutionPlan
    target_kpis: Dict[str, Any]
    rationale: str = ""


@dataclass
class MockTuningChunk:
    """Mock tuning chunk matching real TuningChunk structure."""
    name: str
    category: str
    apply_commands: List[str]
    rationale: str
    requires_restart: bool = False


@dataclass
class MockAnalysisResult:
    """Mock analysis result matching real AnalysisResult structure."""
    tuning_chunks: List[MockTuningChunk]
    observations: List[str]
    confidence: float
    expected_improvement: float


class MockGeminiAgent:
    """
    Mock AI agent that returns realistic PostgreSQL tuning recommendations.

    Uses golden_data.py for all responses - no actual API calls.

    Usage:
        agent = MockGeminiAgent()
        strategies = agent.get_first_sight(context)
        analysis = agent.analyze_results(strategy, result, ...)
    """

    def __init__(self, scenario: str = "balanced_tps"):
        """
        Initialize mock agent.

        Args:
            scenario: Which test scenario to simulate
                - "balanced_tps": Normal progression to target
                - "wal_optimized": Fast WAL-focused optimization
                - "error_scenario": Simulates errors (0 TPS, etc.)
        """
        self.scenario = scenario
        self.call_count = 0
        self.round_num = 0
        self._workload = WorkloadType.MIXED

    def first_sight_analysis(
        self,
        context: Dict[str, Any],
        workload_hint: Optional[str] = None,
    ):
        """
        Get initial system analysis and strategy recommendations.

        Returns mock first sight data with realistic observations.
        Compatible with real GeminiAgent interface.
        """
        from ...protocol.first_sight import FirstSightResponse, StrategyOption
        self.call_count += 1

        # Determine workload type
        if workload_hint:
            self._workload = WorkloadType(workload_hint)

        # Get strategies for this workload
        strategies = get_strategies_for_workload(self._workload)
        benchmark_config = get_benchmark_for_workload(self._workload)

        # Build response as proper FirstSightResponse
        strategy_options = [
            StrategyOption(
                id=f"strategy-{s['name'].lower().replace(' ', '-')}",
                name=s["name"],
                goal=s["hypothesis"],
                hypothesis=s["hypothesis"],
                target_kpis=s["target_kpis"],
                rationale=f"Recommended for {self._workload.value} workloads",
                estimated_duration_minutes=benchmark_config.get('duration_seconds', 60) // 60 + 1,
                risk_level="LOW",
            )
            for s in strategies[:3]  # Return top 3 strategies
        ]

        return FirstSightResponse(
            protocol_version="v2",
            system_overview=MOCK_FIRST_SIGHT["system_info"].get("overview", "8-core system with 32GB RAM"),
            schema_overview=MOCK_FIRST_SIGHT.get("schema_overview", "Database ready for benchmarking"),
            key_observations=MOCK_FIRST_SIGHT["observations"],
            warnings=[],
            strategy_options=strategy_options,
        )

    def get_next_strategies(
        self,
        context: Dict[str, Any],
        previous_session: Optional[Dict] = None,
        excluded_strategy: Optional[Dict] = None,
    ):
        """
        Get new strategies after a completed session.

        Used when user wants to try different approach.
        Returns FirstSightResponse compatible with real agent.
        """
        from ...protocol.first_sight import FirstSightResponse, StrategyOption
        self.call_count += 1

        # Get all strategies except excluded
        all_strategies = list(STRATEGIES.values())
        excluded_name = excluded_strategy.get('name') if isinstance(excluded_strategy, dict) else excluded_strategy
        if excluded_name:
            all_strategies = [s for s in all_strategies if s["name"] != excluded_name]

        strategy_options = [
            StrategyOption(
                id=f"strategy-{s['name'].lower().replace(' ', '-')}",
                name=s["name"],
                goal=s["hypothesis"],
                hypothesis=s["hypothesis"],
                target_kpis=s["target_kpis"],
                rationale=f"Alternative approach: {s['hypothesis']}",
                estimated_duration_minutes=5,
                risk_level="LOW",
            )
            for s in all_strategies[:3]
        ]

        return FirstSightResponse(
            protocol_version="v2",
            system_overview="Updated analysis based on previous session results.",
            schema_overview="Schema analysis unchanged.",
            key_observations=[
                f"Previous session achieved {previous_session.get('best_tps', 0) if previous_session else 0} TPS",
                "System shows capacity for further optimization",
            ],
            warnings=[],
            strategy_options=strategy_options,
        )

    def suggest_target(
        self,
        baseline_tps: float,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return target suggestion based on baseline TPS."""
        self.call_count += 1

        # Get expected improvement from golden data
        expected_tps = get_expected_tps(self.scenario, 3)  # After 3 rounds
        improvement_factor = expected_tps / baseline_tps if baseline_tps > 0 else 1.8

        target_tps = int(baseline_tps * improvement_factor)
        return {
            'target_tps': target_tps,
            'rationale': f"Based on {self.scenario} optimization, expect {(improvement_factor-1)*100:.0f}% improvement",
        }

    def get_round1_config(
        self,
        context: Dict[str, Any],
        first_sight,
        chosen_strategy: Dict[str, Any],
        dba_message: Optional[str] = None,
    ):
        """Return Round 1 configuration with tuning chunks from golden data."""
        from ...protocol.round1 import Round1Config
        from ...protocol.sdl import TuningChunk, OsTuning

        # Get round 1 tuning from golden data
        tuning = get_tuning_for_round(self.scenario, 1)

        tuning_chunks = []
        for i, change in enumerate(tuning.get('changes', [])):
            tuning_chunks.append(TuningChunk(
                id=f"chunk-{i+1:03d}",
                category=change.get('category', 'misc'),
                name=change.get('name', f'Tuning {i+1}'),
                description=change.get('rationale', ''),
                apply_commands=change.get('pg_configs', []),
                rollback_commands=[],
                requires_restart=change.get('requires_restart', False),
                priority="HIGH" if i == 0 else "MEDIUM",
            ))

        return Round1Config(
            protocol_version="v2",
            response_type="round1_config",
            rationale=tuning.get('name', 'Initial optimization'),
            tuning_chunks=tuning_chunks,
            os_tuning=[],
            restart_required=any(c.get('requires_restart', False) for c in tuning.get('changes', [])),
            restart_reason="Configuration changes require restart" if any(c.get('requires_restart', False) for c in tuning.get('changes', [])) else "",
        )

    def generate_strategy(self, context: Dict[str, Any]):
        """Return a strategy definition for testing."""
        from ...protocol.sdl import StrategyDefinition, ExecutionPlan, SuccessCriteria

        benchmark_config = get_benchmark_for_workload(self._workload)

        return StrategyDefinition(
            protocol_version="v2",
            id=f"strategy-{self.scenario}",
            name=f"Test Strategy ({self.scenario})",
            hypothesis="Testing database performance optimization",
            execution_plan=ExecutionPlan(
                benchmark_type="pgbench",
                scale=benchmark_config.get('scale', 100),
                clients=benchmark_config.get('clients', 32),
                threads=benchmark_config.get('threads', 8),
                duration_seconds=benchmark_config.get('duration_seconds', 60),
                warmup_seconds=5,
            ),
            success_criteria=SuccessCriteria(
                target_tps=9000,
                max_latency_p99_ms=100,
                min_cache_hit_ratio=95.0,
            ),
        )

    def analyze_results(
        self,
        strategy: Any,
        result: Any,
        telemetry_summary: str = "",
        current_config: Dict[str, str] = None,
        target_kpis: Dict[str, Any] = None,
        tuning_history: Dict[str, Any] = None,
        human_feedback: Optional[Dict] = None,
    ):
        """
        Analyze benchmark results and return tuning recommendations.

        Returns TuningProposal or SessionConclusion compatible with real agent.
        """
        from ...protocol.tuning import TuningProposal, TuningChunk, ExpectedImprovement
        from ...protocol.conclusion import (
            SessionConclusion,
            TuningSummary,
            HardwareSaturationAnalysis,
            ScalingRecommendation,
        )

        self.call_count += 1
        self.round_num += 1

        # Get current TPS from result
        current_tps = 0
        if hasattr(result, 'metrics') and result.metrics:
            current_tps = result.metrics.tps
        elif hasattr(result, 'tps'):
            current_tps = result.tps
        elif isinstance(result, dict):
            current_tps = result.get('tps', 0)

        # Get target from KPIs
        target_tps = 9000
        if target_kpis:
            target_tps = target_kpis.get('target_tps', 9000)

        # Check if target is achieved
        if current_tps >= target_tps:
            baseline_tps = tuning_history.get('baseline_tps', current_tps / 1.5) if tuning_history else current_tps / 1.5
            return SessionConclusion(
                protocol_version="v2",
                response_type="conclude_session",
                session_id="mock-session",
                conclusion_reason="TARGET_ACHIEVED",
                tuning_summary=TuningSummary(
                    total_iterations=self.round_num,
                    baseline_tps=baseline_tps,
                    final_tps=current_tps,
                    improvement_pct=((current_tps / baseline_tps) - 1) * 100 if baseline_tps > 0 else 0,
                    key_changes_applied=["Applied golden data tuning"],
                ),
                hardware_saturation_analysis=HardwareSaturationAnalysis(
                    is_saturated=False,
                    bottleneck_resource=None,
                    evidence=[],
                    scaling_recommendation=None,
                ),
                final_report_markdown=f"# Session Complete\n\nTarget of {target_tps} TPS achieved with {current_tps:.0f} TPS.",
            )

        # Handle error scenario
        if self.scenario == "error_scenario" and self.round_num == 1:
            current_tps = 0  # Simulate 0 TPS error

        # Check for saturation (after 5 rounds with diminishing returns)
        if tuning_history and tuning_history.get('iterations_completed', 0) >= 5:
            baseline_tps = tuning_history.get('baseline_tps', 5000)
            return SessionConclusion(
                protocol_version="v2",
                response_type="conclude_session",
                session_id="mock-session",
                conclusion_reason="DIMINISHING_RETURNS",
                tuning_summary=TuningSummary(
                    total_iterations=self.round_num,
                    baseline_tps=baseline_tps,
                    final_tps=current_tps,
                    improvement_pct=((current_tps / baseline_tps) - 1) * 100 if baseline_tps > 0 else 0,
                    key_changes_applied=["Applied all available tuning"],
                ),
                hardware_saturation_analysis=HardwareSaturationAnalysis(
                    is_saturated=True,
                    bottleneck_resource="CPU",
                    evidence=["CPU utilization near 100% during benchmarks"],
                    scaling_recommendation=ScalingRecommendation(
                        action="SCALE_UP",
                        details="Consider adding more CPU cores or upgrading to faster processors",
                    ),
                ),
                final_report_markdown=f"# Session Complete\n\nHardware saturation reached at {current_tps:.0f} TPS.",
            )

        # Generate realistic analysis
        analysis = generate_analysis_response(
            round_num=self.round_num,
            current_tps=current_tps,
            target_tps=target_tps,
            strategy=self.scenario,
        )

        # Apply human feedback if provided
        if human_feedback and human_feedback.get('message'):
            analysis['observations'].append(
                f"DBA Feedback: {human_feedback['message']}"
            )

        # Build tuning chunks
        tuning_chunks = []
        for i, chunk in enumerate(analysis.get('tuning_chunks', [])):
            tuning_chunks.append(TuningChunk(
                id=f"chunk-{self.round_num:02d}-{i+1:02d}",
                category=chunk.get('category', 'misc'),
                name=chunk.get('name', f'Tuning {i+1}'),
                description=chunk.get('rationale', ''),
                apply_commands=chunk.get('apply_commands', []),
                rollback_commands=[],
                requires_restart=chunk.get('requires_restart', False),
                priority="HIGH" if i == 0 else "MEDIUM",
            ))

        return TuningProposal(
            protocol_version="v2",
            response_type="tuning_proposal",
            observations=analysis.get('observations', []),
            tuning_chunks=tuning_chunks,
            expected_improvement=ExpectedImprovement(
                tps_improvement_pct=analysis.get('expected_improvement', 10),
                confidence_score=analysis.get('confidence', 0.7),
            ),
            restart_required=any(c.requires_restart for c in tuning_chunks),
            restart_reason="Some changes require PostgreSQL restart" if any(c.requires_restart for c in tuning_chunks) else "",
        )

    def reset(self):
        """Reset agent state for new test."""
        self.call_count = 0
        self.round_num = 0


class MockAgentFactory:
    """Factory for creating mock agents with different scenarios."""

    @staticmethod
    def create(scenario: str = "balanced_tps") -> MockGeminiAgent:
        """
        Create a mock agent for a specific test scenario.

        Args:
            scenario: One of:
                - "balanced_tps": Normal happy path
                - "wal_optimized": Fast WAL optimization
                - "error_scenario": Simulates errors
                - "slow_progress": Many rounds, slow improvement
        """
        return MockGeminiAgent(scenario=scenario)

    @staticmethod
    def create_for_test(test_id: str) -> MockGeminiAgent:
        """
        Create agent based on test ID.

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
        return MockGeminiAgent(scenario=scenario)
