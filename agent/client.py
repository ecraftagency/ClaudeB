"""
GeminiAgent - AI client for strategy generation and analysis.

The Agent is the "Architect" in the v2.0 Dynamic Contextual Architecture.
It receives ContextPackets and produces SDL/TuningProposals.
"""

import os
import time
import warnings
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

# Suppress Google API deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["GRPC_VERBOSITY"] = "ERROR"

from ..protocol.context import ContextPacket
from ..protocol.sdl import StrategyDefinition, Round1Config
from ..protocol.tuning import TuningProposal
from ..protocol.result import BenchmarkResult
from ..protocol.errors import TuningErrorPacket
from ..protocol.conclusion import SessionConclusion
from ..protocol.first_sight import FirstSightResponse
from .prompts import PromptBuilder
from .parser import SDLParser, ParseError


class GeminiAgentError(Exception):
    """Error in Gemini agent operation."""
    pass


class GeminiAgent:
    """
    AI Agent using Google Gemini for PostgreSQL diagnostics.

    Responsibilities:
    - Analyze system context and generate diagnostic strategies
    - Analyze benchmark results and produce tuning recommendations
    - Handle error recovery suggestions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3-flash-preview",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Gemini agent.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Gemini model to use
            max_retries: Maximum retry attempts for API calls
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise GeminiAgentError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize Gemini client
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        except ImportError:
            raise GeminiAgentError(
                "google-generativeai package required. "
                "Install with: pip install google-generativeai"
            )

    def generate_strategy(
        self,
        context: Union[ContextPacket, Dict[str, Any]]
    ) -> StrategyDefinition:
        """
        Generate a diagnostic strategy from context.

        Args:
            context: ContextPacket or dict with system/runtime/schema context

        Returns:
            StrategyDefinition with benchmark plan

        Raises:
            GeminiAgentError: On API or parsing failure
        """
        # Convert dataclass to dict if needed
        if hasattr(context, '__dataclass_fields__'):
            context_dict = asdict(context)
        else:
            context_dict = context

        # Build prompt
        prompt = PromptBuilder.build_strategy_prompt(
            system_context=context_dict.get('system_context', {}),
            runtime_context=context_dict.get('runtime_context', {}),
            schema_context=context_dict.get('schema_context', {}),
            previous_iteration=context_dict.get('previous_iteration'),
        )

        # Call Gemini with retries
        response_text = self._call_gemini(prompt)

        # Parse response
        try:
            strategy = SDLParser.parse_strategy(response_text)

            # Validate
            issues = SDLParser.validate_strategy(strategy)
            if any(issue.startswith('Error:') for issue in issues):
                raise GeminiAgentError(f"Strategy validation failed: {issues}")

            return strategy

        except ParseError as e:
            raise GeminiAgentError(f"Failed to parse strategy response: {e}")

    def first_sight_analysis(
        self,
        context: Union[ContextPacket, Dict[str, Any]]
    ) -> FirstSightResponse:
        """
        Get initial system analysis and strategy options.

        This is the first AI interaction - before any benchmarking.
        Returns a brief overview and multiple strategy options for user selection.

        Args:
            context: ContextPacket or dict with system/runtime/schema context

        Returns:
            FirstSightResponse with overview and strategy options

        Raises:
            GeminiAgentError: On API or parsing failure
        """
        # Convert dataclass to dict if needed
        if hasattr(context, '__dataclass_fields__'):
            context_dict = asdict(context)
        else:
            context_dict = context

        # Build prompt
        prompt = PromptBuilder.build_first_sight_prompt(
            system_context=context_dict.get('system_context', {}),
            runtime_context=context_dict.get('runtime_context', {}),
            schema_context=context_dict.get('schema_context', {}),
        )

        # Call Gemini with retries
        response_text = self._call_gemini(prompt)

        # Parse response
        try:
            return SDLParser.parse_first_sight(response_text)
        except ParseError as e:
            raise GeminiAgentError(f"Failed to parse first sight response: {e}")

    def suggest_target(
        self,
        baseline_tps: float,
        context: Union[ContextPacket, Dict[str, Any]],
        strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Suggest target TPS based on baseline measurement.

        Args:
            baseline_tps: Measured baseline TPS
            context: System context
            strategy: Chosen strategy

        Returns:
            Dict with 'target_tps' and 'rationale'
        """
        # Convert dataclass to dict if needed
        if hasattr(context, '__dataclass_fields__'):
            context_dict = asdict(context)
        else:
            context_dict = context

        system_ctx = context_dict.get('system_context', {})

        prompt = f"""Based on the baseline benchmark and system analysis, suggest a realistic target TPS.

## Baseline Result
- Measured TPS: {baseline_tps:.0f}

## System Context
- CPU: {system_ctx.get('cpu_model', 'Unknown')} ({system_ctx.get('cpu_cores', '?')} cores)
- RAM: {system_ctx.get('total_memory_gb', '?')} GB
- Storage: {system_ctx.get('storage_type', 'Unknown')}

## Strategy
- Name: {strategy.get('name', 'Unknown')}
- Goal: {strategy.get('goal', 'Improve performance')}

## Instructions
1. Analyze the baseline TPS relative to system capacity
2. Suggest a realistic target TPS (typically 15-40% improvement)
3. Consider hardware limits - don't suggest impossible targets
4. Provide brief rationale

Respond in this exact format:
TARGET_TPS: <number>
IMPROVEMENT_PCT: <number>
RATIONALE: <one sentence explanation>
"""

        try:
            response_text = self._call_gemini(prompt)

            # Parse simple response
            target_tps = baseline_tps * 1.2  # Default 20%
            rationale = "Default improvement target"

            for line in response_text.strip().split('\n'):
                if line.startswith('TARGET_TPS:'):
                    try:
                        target_tps = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('RATIONALE:'):
                    rationale = line.split(':', 1)[1].strip()

            return {
                'target_tps': target_tps,
                'rationale': rationale,
            }
        except Exception as e:
            # Fallback to default
            return {
                'target_tps': int(baseline_tps * 1.2),
                'rationale': f"Default 20% improvement (AI error: {e})",
            }

    def get_round1_config(
        self,
        context: Union[ContextPacket, Dict[str, Any]],
        first_sight: FirstSightResponse,
        chosen_strategy: Dict[str, Any],
        dba_message: Optional[str] = None,
    ) -> Round1Config:
        """
        Get Round 1 configuration recommendations before benchmarking.

        Args:
            context: ContextPacket or dict with system/runtime context
            first_sight: FirstSightResponse from initial analysis
            chosen_strategy: The strategy chosen by the DBA
            dba_message: Optional DBA customization message

        Returns:
            Round1Config with tuning recommendations

        Raises:
            GeminiAgentError: On API or parsing failure
        """
        # Convert dataclass to dict if needed
        if hasattr(context, '__dataclass_fields__'):
            context_dict = asdict(context)
        else:
            context_dict = context

        # Convert first_sight to summary dict
        first_sight_summary = {
            'system_overview': first_sight.system_overview,
            'schema_overview': first_sight.schema_overview,
            'key_observations': first_sight.key_observations,
            'warnings': first_sight.warnings,
        }

        # Build prompt
        prompt = PromptBuilder.build_round1_config_prompt(
            system_context=context_dict.get('system_context', {}),
            runtime_context=context_dict.get('runtime_context', {}),
            first_sight_summary=first_sight_summary,
            chosen_strategy=chosen_strategy,
            dba_message=dba_message,
        )

        # Call Gemini with retries
        response_text = self._call_gemini(prompt)

        # Parse response
        try:
            return SDLParser.parse_round1_config(response_text)
        except ParseError as e:
            raise GeminiAgentError(f"Failed to parse round1 config response: {e}")

    def get_next_strategies(
        self,
        context: Union[ContextPacket, Dict[str, Any]],
        previous_session: Dict[str, Any],
        excluded_strategy: Dict[str, Any],
    ) -> FirstSightResponse:
        """
        Get new strategy options after a completed tuning session.

        Asks AI to suggest different strategies based on what was learned,
        explicitly excluding the previously tested strategy.

        Args:
            context: ContextPacket or dict with system/runtime/schema context
            previous_session: Results from the completed session (baseline TPS, best TPS, applied changes)
            excluded_strategy: The strategy that was already tested (to exclude)

        Returns:
            FirstSightResponse with new strategy options

        Raises:
            GeminiAgentError: On API or parsing failure
        """
        # Convert dataclass to dict if needed
        if hasattr(context, '__dataclass_fields__'):
            context_dict = asdict(context)
        else:
            context_dict = context

        # Build prompt
        prompt = PromptBuilder.build_next_strategy_prompt(
            system_context=context_dict.get('system_context', {}),
            runtime_context=context_dict.get('runtime_context', {}),
            schema_context=context_dict.get('schema_context', {}),
            previous_session=previous_session,
            excluded_strategy=excluded_strategy,
        )

        # Call Gemini
        response_text = self._call_gemini(prompt)

        # Parse response (reuses FirstSightResponse since format is similar)
        try:
            return SDLParser.parse_first_sight(response_text)
        except ParseError as e:
            raise GeminiAgentError(f"Failed to parse next strategy response: {e}")

    def analyze_results(
        self,
        strategy: StrategyDefinition,
        result: Union[BenchmarkResult, Dict[str, Any]],
        telemetry_summary: str,
        current_config: Dict[str, Any],
        target_kpis: Optional[Dict[str, Any]] = None,
        tuning_history: Optional[Dict[str, Any]] = None,
        human_feedback: Optional[Dict[str, Any]] = None,
    ) -> Union[TuningProposal, SessionConclusion]:
        """
        Analyze benchmark results and generate tuning proposal or session conclusion.

        v2.3: Now supports hardware saturation detection. Returns SessionConclusion
        when AI determines no further software tuning can help.

        Args:
            strategy: The strategy that was executed
            result: Benchmark results
            telemetry_summary: Aggregated telemetry as text
            current_config: Current PostgreSQL configuration
            target_kpis: Target KPIs from selected strategy option (IMPORTANT for correct evaluation)
            tuning_history: v2.3 - History of applied tuning for AI context
            human_feedback: v2.3 - DBA feedback/observations

        Returns:
            TuningProposal with recommended changes, or
            SessionConclusion if hardware is saturated
        """
        # Convert dataclasses to dicts
        strategy_dict = asdict(strategy) if hasattr(strategy, '__dataclass_fields__') else strategy
        result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result

        # Build prompt (v2.3 with tuning history and feedback)
        prompt = PromptBuilder.build_analysis_prompt(
            strategy=strategy_dict,
            benchmark_result=result_dict,
            telemetry_summary=telemetry_summary,
            current_config=current_config,
            target_kpis=target_kpis,
            tuning_history=tuning_history,
            human_feedback=human_feedback,
        )

        # Call Gemini
        response_text = self._call_gemini(prompt)

        # v2.3: Detect response type and parse accordingly
        try:
            response_type = SDLParser.detect_response_type(response_text)

            if response_type == 'conclusion':
                # AI determined hardware is saturated
                conclusion = SDLParser.parse_session_conclusion(response_text)
                return conclusion
            else:
                # Standard tuning proposal
                proposal = SDLParser.parse_tuning_proposal(response_text)

                # Validate
                issues = SDLParser.validate_tuning_proposal(proposal)
                if any(issue.startswith('Error:') for issue in issues):
                    raise GeminiAgentError(f"Tuning proposal validation failed: {issues}")

                return proposal

        except ParseError as e:
            raise GeminiAgentError(f"Failed to parse analysis response: {e}")

    def get_recovery_recommendation(
        self,
        error_packet: Union[TuningErrorPacket, Dict[str, Any]],
        service_logs: list,
        failed_chunk: Dict[str, Any],
        system_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get recovery recommendation for a failed tuning operation.

        Args:
            error_packet: TuningErrorPacket with failure details
            service_logs: PostgreSQL service logs
            failed_chunk: The tuning chunk that failed
            system_context: Current system state

        Returns:
            Recovery recommendation dict
        """
        error_dict = asdict(error_packet) if hasattr(error_packet, '__dataclass_fields__') else error_packet

        prompt = PromptBuilder.build_error_recovery_prompt(
            error_packet=error_dict,
            service_logs=service_logs,
            failed_chunk=failed_chunk,
            system_context=system_context,
        )

        response_text = self._call_gemini(prompt)

        try:
            return SDLParser.parse_recovery_recommendation(response_text)
        except ParseError as e:
            raise GeminiAgentError(f"Failed to parse recovery recommendation: {e}")

    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini API with retry logic.

        Args:
            prompt: The prompt to send

        Returns:
            Response text from Gemini

        Raises:
            GeminiAgentError: On API failure after retries
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.generate_content(prompt)

                # Extract text from response
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'parts'):
                    return ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                else:
                    raise GeminiAgentError("Unexpected response format from Gemini")

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                continue

        raise GeminiAgentError(f"Gemini API call failed after {self.max_retries} attempts: {last_error}")

    def health_check(self) -> bool:
        """
        Check if Gemini API is accessible.

        Returns:
            True if API is working
        """
        try:
            response = self._client.generate_content("Say 'OK' if you're working.")
            return 'ok' in response.text.lower()
        except Exception:
            return False


class MockGeminiAgent:
    """
    Mock agent for testing without API calls.

    Returns predefined responses based on context.
    """

    def first_sight_analysis(self, context) -> FirstSightResponse:
        """Return a mock first sight analysis for testing."""
        from ..protocol.first_sight import StrategyOption

        return FirstSightResponse(
            protocol_version="v2",
            system_overview="8-core CPU with 32GB RAM running PostgreSQL 16 on NVMe RAID10 storage. Well-provisioned system for moderate OLTP workloads.",
            schema_overview="Database contains 5 main tables totaling 2.5GB. High write activity detected on 'transactions' and 'wallets' tables suggesting financial/payment workload.",
            key_observations=[
                "Write-heavy workload pattern detected (transactions table has 10M+ inserts)",
                "shared_buffers set to 8GB (25% of RAM) - reasonable setting",
                "WAL on separate NVMe device - good for write performance",
                "Autovacuum running but may need tuning for high-write tables",
            ],
            warnings=[
                "checkpoint_completion_target at default 0.9 - consider adjusting for write workload",
            ],
            strategy_options=[
                StrategyOption(
                    id="strategy-oltp-baseline",
                    name="OLTP Baseline Test",
                    goal="Establish baseline transaction throughput",
                    hypothesis="Measure current TPS capacity under mixed read/write workload",
                    target_kpis={"primary": "TPS", "target_tps": 5000, "max_latency_ms": 50},
                    rationale="Start with a standard OLTP benchmark to understand current performance baseline before any tuning",
                    estimated_duration_minutes=5,
                    risk_level="LOW",
                ),
                StrategyOption(
                    id="strategy-write-stress",
                    name="Write-Heavy Stress Test",
                    goal="Test WAL and checkpoint performance",
                    hypothesis="High write load may reveal WAL or checkpoint bottlenecks",
                    target_kpis={"primary": "TPS", "target_tps": 3000, "secondary": "WAL write latency"},
                    rationale="Given the write-heavy pattern detected, stress testing the write path will identify tuning opportunities",
                    estimated_duration_minutes=10,
                    risk_level="MEDIUM",
                ),
                StrategyOption(
                    id="strategy-connection-pool",
                    name="Connection Scaling Test",
                    goal="Test connection handling under load",
                    hypothesis="Identify connection bottlenecks as client count increases",
                    target_kpis={"primary": "TPS", "secondary": "Connection latency"},
                    rationale="Test how well the system handles increasing concurrent connections",
                    estimated_duration_minutes=8,
                    risk_level="LOW",
                ),
            ],
        )

    def get_next_strategies(
        self,
        context,
        previous_session: Dict[str, Any],
        excluded_strategy: Dict[str, Any],
    ) -> FirstSightResponse:
        """Return mock next strategies for testing."""
        from ..protocol.first_sight import StrategyOption

        return FirstSightResponse(
            protocol_version="v2",
            system_overview="Updated view based on previous session results.",
            schema_overview="Schema analysis unchanged.",
            key_observations=[
                f"Previous session achieved {previous_session.get('best_tps', 0)} TPS",
                "System shows capacity for further optimization",
            ],
            warnings=[],
            strategy_options=[
                StrategyOption(
                    id="strategy-read-intensive",
                    name="Read-Heavy Workload Test",
                    goal="Test query performance and cache efficiency",
                    hypothesis="Explore read path performance after write optimization",
                    target_kpis={"primary": "TPS", "target_tps": 8000, "max_latency_ms": 30},
                    rationale="Now that writes are optimized, test read-heavy scenarios",
                    estimated_duration_minutes=3,
                    risk_level="LOW",
                ),
                StrategyOption(
                    id="strategy-mixed-workload",
                    name="Mixed Read/Write Test",
                    goal="Test realistic mixed workload",
                    hypothesis="Balanced workload may reveal contention issues",
                    target_kpis={"primary": "TPS", "target_tps": 6000},
                    rationale="Test a more realistic mixed workload pattern",
                    estimated_duration_minutes=5,
                    risk_level="MEDIUM",
                ),
            ],
        )

    def suggest_target(
        self,
        baseline_tps: float,
        context,
        strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return mock target suggestion."""
        # Suggest 25% improvement for mock
        target_tps = int(baseline_tps * 1.25)
        return {
            'target_tps': target_tps,
            'rationale': f"Mock suggestion: 25% improvement over baseline ({baseline_tps:.0f} → {target_tps})",
        }

    def get_round1_config(
        self,
        context,
        first_sight,
        chosen_strategy: Dict[str, Any],
        dba_message: Optional[str] = None,
    ) -> Round1Config:
        """Return mock Round 1 configuration for testing."""
        from ..protocol.sdl import TuningChunk, OsTuning

        return Round1Config(
            protocol_version="v2",
            response_type="round1_config",
            rationale="Based on the write-heavy workload pattern and current configuration, these optimizations will improve write throughput before benchmarking.",
            tuning_chunks=[
                TuningChunk(
                    id="chunk-001",
                    category="wal",
                    name="Optimize WAL for Write Workload",
                    description="Increase wal_buffers and enable wal_compression for better write performance",
                    apply_commands=[
                        "ALTER SYSTEM SET wal_buffers = '64MB'",
                        "ALTER SYSTEM SET wal_compression = 'on'",
                        "SELECT pg_reload_conf()",
                    ],
                    rollback_commands=[
                        "ALTER SYSTEM SET wal_buffers = '16MB'",
                        "ALTER SYSTEM SET wal_compression = 'off'",
                        "SELECT pg_reload_conf()",
                    ],
                    requires_restart=False,
                    verification_command="SHOW wal_buffers",
                    verification_expected="64MB",
                    priority="HIGH",
                ),
                TuningChunk(
                    id="chunk-002",
                    category="checkpoint",
                    name="Tune Checkpoint Settings",
                    description="Spread checkpoints over longer period to reduce I/O spikes",
                    apply_commands=[
                        "ALTER SYSTEM SET checkpoint_completion_target = '0.9'",
                        "ALTER SYSTEM SET max_wal_size = '4GB'",
                        "SELECT pg_reload_conf()",
                    ],
                    rollback_commands=[
                        "ALTER SYSTEM SET checkpoint_completion_target = '0.5'",
                        "ALTER SYSTEM SET max_wal_size = '1GB'",
                        "SELECT pg_reload_conf()",
                    ],
                    requires_restart=False,
                    verification_command="SHOW checkpoint_completion_target",
                    verification_expected="0.9",
                    priority="MEDIUM",
                ),
            ],
            os_tuning=[
                OsTuning(
                    id="os-001",
                    name="Optimize dirty page ratio",
                    description="Reduce dirty ratio to prevent large write stalls",
                    apply_command="sudo sysctl -w vm.dirty_ratio=10",
                    rollback_command="sudo sysctl -w vm.dirty_ratio=20",
                    persistent_file="/etc/sysctl.d/99-pg-tuning.conf",
                    persistent_line="vm.dirty_ratio = 10",
                ),
            ],
            restart_required=False,
            restart_reason="",
        )

    def generate_strategy(self, context) -> StrategyDefinition:
        """Return a basic strategy for testing."""
        from ..protocol.sdl import ExecutionPlan, SuccessCriteria

        return StrategyDefinition(
            protocol_version="v2",
            id="strategy-mock-001",
            name="Mock Diagnostic Strategy",
            hypothesis="Testing I/O throughput",
            execution_plan=ExecutionPlan(
                benchmark_type="pgbench",
                scale=100,
                clients=10,
                threads=2,
                duration_seconds=30,
                warmup_seconds=5,
            ),
            success_criteria=SuccessCriteria(
                target_tps=1000,
                max_latency_p99_ms=100,
                min_cache_hit_ratio=95.0,
            ),
        )

    def analyze_results(
        self,
        strategy,
        result,
        telemetry_summary,
        current_config,
        target_kpis=None,
        tuning_history=None,
        human_feedback=None,
    ) -> Union[TuningProposal, SessionConclusion]:
        """Return a basic tuning proposal for testing (v2.3 compatible)."""
        from ..protocol.tuning import TuningChunk, ExpectedImprovement
        from ..protocol.conclusion import (
            SessionConclusion,
            TuningSummary,
            HardwareSaturationAnalysis,
            ScalingRecommendation,
        )

        # Simulate saturation detection after 3 iterations
        if tuning_history and tuning_history.get('iterations_completed', 0) >= 3:
            return SessionConclusion(
                protocol_version="v2",
                response_type="conclude_session",
                session_id="mock-session",
                conclusion_reason="DIMINISHING_RETURNS",
                tuning_summary=TuningSummary(
                    total_iterations=3,
                    baseline_tps=1000,
                    final_tps=1500,
                    improvement_pct=50,
                    key_changes_applied=["Increased shared_buffers"],
                ),
                hardware_saturation_analysis=HardwareSaturationAnalysis(
                    is_saturated=True,
                    bottleneck_resource="CPU",
                    evidence=["CPU usage at 95% during benchmark"],
                    scaling_recommendation=ScalingRecommendation(
                        action="SCALE_UP",
                        details="Consider adding more CPU cores",
                    ),
                ),
                final_report_markdown="# Mock Session Complete\n\nHardware is saturated.",
            )

        return TuningProposal(
            protocol_version="v2",
            analysis_summary="Mock analysis: I/O appears to be the bottleneck",
            bottleneck_type="io",
            confidence=0.75,
            tuning_chunks=[
                TuningChunk(
                    id="chunk-mock-001",
                    category="memory",
                    name="Increase shared_buffers",
                    rationale="More buffer cache to reduce disk reads",
                    apply_commands=[
                        "ALTER SYSTEM SET shared_buffers = '2GB'",
                        "SELECT pg_reload_conf()",
                    ],
                    rollback_commands=[
                        "ALTER SYSTEM SET shared_buffers = '128MB'",
                        "SELECT pg_reload_conf()",
                    ],
                    requires_restart=True,
                    recovery_strategy="SQL_REVERT",
                    verification_command="SHOW shared_buffers",
                    verification_expected="2GB",
                    priority="HIGH",
                ),
            ],
            expected_improvement=ExpectedImprovement(
                tps_increase_pct=15,
                latency_reduction_pct=10,
            ),
        )

    def get_recovery_recommendation(self, error_packet, service_logs, failed_chunk, system_context):
        """Return a basic recovery recommendation."""
        return {
            "diagnosis": "Mock diagnosis: Configuration value too high",
            "root_cause": "config",
            "recovery_steps": [
                {
                    "step": 1,
                    "action": "Restore original configuration",
                    "command": "sudo systemctl restart postgresql",
                    "verify": "pg_isready",
                }
            ],
            "prevention": "Validate values against system resources before applying",
            "retry_with_modifications": {
                "should_retry": True,
                "modified_chunk": failed_chunk,
            },
        }

    def health_check(self) -> bool:
        """Mock always returns True."""
        return True
