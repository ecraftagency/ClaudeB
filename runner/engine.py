"""
DiagnosticEngine - Main orchestrator for the diagnostic workflow.

Implements the v2.0 Dynamic Contextual Architecture:
- Runner (this) is the "dumb executor"
- Agent (AI) is the "architect"
- They communicate via protocol dataclasses
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from datetime import datetime

if TYPE_CHECKING:
    import psycopg2

from .state import StateMachine, State
from .benchmark import BenchmarkRunner, BenchmarkConfig
from ..protocol.context import ContextPacket
from ..protocol.sdl import StrategyDefinition
from ..protocol.result import BenchmarkResult, HumanFeedback
from ..protocol.tuning import TuningProposal, TuningResult
from ..protocol.conclusion import SessionConclusion
from ..protocol.errors import TuningErrorPacket
from ..discovery.schema import SchemaScanner
from ..discovery.system import SystemScanner, SystemScannerConfig
from ..discovery.runtime import RuntimeScanner
from ..telemetry.collector import TelemetryCollector, CollectorConfig
from ..tuning.executor import TuningExecutor, ExecutorConfig
from ..agent.client import GeminiAgent, MockGeminiAgent
from ..ui.interaction import InteractionManager, UserAction


@dataclass
class EngineConfig:
    """Configuration for diagnostic engine."""
    # Database connection
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_password: Optional[str] = None
    pg_database: str = "postgres"

    # SSH for remote operations
    ssh_host: Optional[str] = None
    ssh_user: str = "ubuntu"
    ssh_key: Optional[str] = None

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./pg_diagnose_output"))

    # Behavior
    max_iterations: int = 3
    auto_apply_tuning: bool = False
    use_mock_agent: bool = False
    interactive_mode: bool = True  # v2.3: Enable human-in-the-loop

    # Gemini
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3-flash-preview"


class DiagnosticEngine:
    """
    Main orchestrator for PostgreSQL diagnostics.

    State machine:
    INIT → DISCOVER → STRATEGIZE → EXECUTE → ANALYZE → TUNE → VERIFY → COMPLETE
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        connection=None,
    ):
        self.config = config or EngineConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Database connection
        self.conn = connection

        # State machine
        self.state_machine = StateMachine()

        # Components (initialized lazily)
        self._agent = None
        self._schema_scanner = None
        self._system_scanner = None
        self._runtime_scanner = None
        self._telemetry_collector = None
        self._benchmark_runner = None
        self._tuning_executor = None

        # Session data
        self._session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._context_packet: Optional[ContextPacket] = None
        self._current_strategy: Optional[StrategyDefinition] = None
        self._current_result: Optional[BenchmarkResult] = None
        self._current_proposal: Optional[TuningProposal] = None
        self._tuning_results: list = []

        # v2.3: Session conclusion and tuning history
        self._session_conclusion: Optional[SessionConclusion] = None
        self._tuning_history: list = []  # Track all applied tuning across iterations
        self._baseline_tps: float = 0.0  # First benchmark TPS for comparison
        self._current_feedback: Optional[HumanFeedback] = None

        # v2.3: Interaction manager for human-in-the-loop
        self._interaction_manager: Optional[InteractionManager] = None

        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_progress: Optional[Callable] = None

    def set_connection(self, connection):
        """Set database connection."""
        self.conn = connection

    def on_state_change(self, callback: Callable[[State, State, Dict], None]):
        """Register callback for state changes."""
        self._on_state_change = callback

    def on_progress(self, callback: Callable[[str, float], None]):
        """Register callback for progress updates."""
        self._on_progress = callback

    @property
    def agent(self):
        """Get or initialize AI agent."""
        if self._agent is None:
            if self.config.use_mock_agent:
                self._agent = MockGeminiAgent()
            else:
                self._agent = GeminiAgent(
                    api_key=self.config.gemini_api_key,
                    model=self.config.gemini_model,
                )
        return self._agent

    @property
    def schema_scanner(self):
        """Get or initialize schema scanner."""
        if self._schema_scanner is None:
            self._schema_scanner = SchemaScanner(self.conn)
        return self._schema_scanner

    @property
    def system_scanner(self):
        """Get or initialize system scanner."""
        if self._system_scanner is None:
            config = None
            if self.config.ssh_host:
                config = SystemScannerConfig(
                    ssh_host=self.config.ssh_host,
                    ssh_user=self.config.ssh_user,
                    ssh_key=self.config.ssh_key,
                )
            self._system_scanner = SystemScanner(config=config)
        return self._system_scanner

    @property
    def runtime_scanner(self):
        """Get or initialize runtime scanner."""
        if self._runtime_scanner is None:
            self._runtime_scanner = RuntimeScanner(self.conn, self.system_scanner)
        return self._runtime_scanner

    @property
    def telemetry_collector(self):
        """Get or initialize telemetry collector."""
        if self._telemetry_collector is None:
            self._telemetry_collector = TelemetryCollector(
                connection=self.conn,
                config=CollectorConfig(
                    ssh_host=self.config.ssh_host,
                    ssh_user=self.config.ssh_user,
                ),
            )
        return self._telemetry_collector

    @property
    def benchmark_runner(self):
        """Get or initialize benchmark runner."""
        if self._benchmark_runner is None:
            self._benchmark_runner = BenchmarkRunner(
                config=BenchmarkConfig(
                    pg_host=self.config.pg_host,
                    pg_port=self.config.pg_port,
                    pg_user=self.config.pg_user,
                    pg_database=self.config.pg_database,
                    pg_password=self.config.pg_password,
                    ssh_host=self.config.ssh_host,
                    ssh_user=self.config.ssh_user,
                    output_dir=self.config.output_dir / "benchmark",
                )
            )
        return self._benchmark_runner

    @property
    def tuning_executor(self):
        """Get or initialize tuning executor."""
        if self._tuning_executor is None:
            self._tuning_executor = TuningExecutor(
                connection=self.conn,
                config=ExecutorConfig(
                    ssh_host=self.config.ssh_host,
                    ssh_user=self.config.ssh_user,
                    ssh_key=self.config.ssh_key,
                ),
            )
        return self._tuning_executor

    @property
    def interaction_manager(self):
        """Get or initialize interaction manager (v2.3)."""
        if self._interaction_manager is None:
            self._interaction_manager = InteractionManager(quiet=not self.config.interactive_mode)
        return self._interaction_manager

    def run(self) -> Dict[str, Any]:
        """
        Run the full diagnostic workflow (v2.3 with human-in-the-loop).

        Returns:
            Summary of the diagnostic session
        """
        try:
            # Phase 1: Discover
            self._transition(State.DISCOVER)
            self._discover()

            # Phase 2: Strategize
            self._transition(State.STRATEGIZE)
            self._strategize()

            # Iteration loop
            for iteration in range(self.config.max_iterations):
                # Phase 3: Execute benchmark
                self._transition(State.EXECUTE)
                self._execute()

                # Track baseline TPS from first iteration
                if iteration == 0 and self._current_result and self._current_result.metrics:
                    self._baseline_tps = self._current_result.metrics.tps

                # v2.3: Phase 3.5: Await user input (if interactive mode)
                if self.config.interactive_mode:
                    self._transition(State.AWAIT_USER_INPUT)
                    user_action = self._await_user_input()

                    if user_action == UserAction.QUIT:
                        self._transition(State.COMPLETE, {"reason": "user_quit"})
                        break
                    elif user_action == UserAction.SKIP:
                        # Skip AI analysis, go directly to complete or next iteration
                        if self.config.auto_apply_tuning and iteration < self.config.max_iterations - 1:
                            continue  # Next iteration without analysis
                        else:
                            self._transition(State.COMPLETE, {"reason": "user_skip"})
                            break

                # Phase 4: Analyze results
                self._transition(State.ANALYZE)
                continue_session = self._analyze()

                # v2.3: Check for session conclusion (hardware saturation)
                if self._session_conclusion:
                    self._transition(State.COMPLETE, {"reason": self._session_conclusion.conclusion_reason})
                    break

                # Check if tuning needed
                if not self._current_proposal or not self._current_proposal.tuning_chunks:
                    self._transition(State.COMPLETE, {"reason": "no_tuning_needed"})
                    break

                # Phase 5: Apply tuning (if auto-apply enabled)
                if self.config.auto_apply_tuning:
                    self._transition(State.TUNE)
                    success = self._tune()

                    if not success:
                        # Emergency rollback already handled in tuning
                        self._transition(State.EMERGENCY_ROLLBACK)
                        # Try to recover
                        continue

                    # Track applied tuning for history
                    for chunk in self._current_proposal.tuning_chunks:
                        self._tuning_history.append({
                            "iteration": iteration,
                            "chunk_name": chunk.name,
                            "category": chunk.category,
                        })

                    # Phase 6: Verify
                    self._transition(State.VERIFY)
                    verified = self._verify()

                    if not verified:
                        # Continue to next iteration
                        continue
                else:
                    # Manual mode - complete after analysis
                    self._transition(State.COMPLETE, {"reason": "manual_mode"})
                    break

            if not self.state_machine.is_terminal():
                self._transition(State.COMPLETE, {"reason": "max_iterations"})

        except Exception as e:
            self._transition(State.FAILED, {"error": str(e)})
            raise

        return self._generate_summary()

    def _await_user_input(self) -> UserAction:
        """v2.3: Await user decision after benchmark."""
        # Build summary for display
        summary = ""
        if self._current_result and self._current_result.metrics:
            metrics = self._current_result.metrics
            summary = f"TPS: {metrics.tps:.2f} | Avg Latency: {metrics.latency_avg_ms:.2f}ms"

        action, feedback = self.interaction_manager.await_user_input(summary)

        if feedback:
            self._current_feedback = feedback
            self.interaction_manager.display_feedback_summary(feedback)

        if action == UserAction.CONTINUE:
            self.interaction_manager.notify_sending_to_ai(has_feedback=feedback is not None)

        return action

    def _transition(self, to_state: State, metadata: Optional[Dict] = None):
        """Transition state machine and notify callbacks."""
        from_state = self.state_machine.state
        self.state_machine.transition(to_state, metadata)

        if self._on_state_change:
            self._on_state_change(from_state, to_state, metadata or {})

    def _progress(self, message: str, pct: float):
        """Report progress."""
        if self._on_progress:
            self._on_progress(message, pct)

    def _discover(self):
        """Phase 1: Discover system context."""
        self._progress("Scanning system...", 0.1)

        system_context = self.system_scanner.scan()

        self._progress("Scanning schema...", 0.3)
        schema_context = self.schema_scanner.scan()

        self._progress("Scanning runtime config...", 0.5)
        runtime_context = self.runtime_scanner.scan()

        # Build context packet
        self._context_packet = ContextPacket(
            protocol_version="v2",
            timestamp=datetime.utcnow().isoformat(),
            session_id=self._session_id,
            system_context=system_context,
            runtime_context=runtime_context,
            schema_context=schema_context,
        )

        # Save discovery results
        self._save_json("discovery/context.json", asdict(self._context_packet))

        self._progress("Discovery complete", 1.0)

    def _strategize(self):
        """Phase 2: Get strategy from AI agent."""
        self._progress("Generating strategy...", 0.0)

        self._current_strategy = self.agent.generate_strategy(self._context_packet)

        # Save strategy
        self._save_json("strategy/strategy.json", asdict(self._current_strategy))

        self._progress("Strategy generated", 1.0)

    def _execute(self):
        """Phase 3: Execute benchmark."""
        self._progress("Running benchmark...", 0.0)

        # Reset telemetry
        self._telemetry_collector = None

        self._current_result = self.benchmark_runner.run(
            strategy=self._current_strategy,
            telemetry_collector=self.telemetry_collector,
        )

        # Save results
        iteration = self.state_machine.iteration
        self._save_json(
            f"benchmark/result_iter{iteration}.json",
            asdict(self._current_result)
        )

        self._progress("Benchmark complete", 1.0)

    def _analyze(self) -> bool:
        """
        Phase 4: Analyze results with AI (v2.3).

        Returns:
            True if session should continue, False if concluded
        """
        self._progress("Analyzing results...", 0.0)

        # Get telemetry summary
        telemetry_summary = self.telemetry_collector.get_summary()
        telemetry_text = self.telemetry_collector.to_telemetry_points()

        from ..telemetry.aggregator import TelemetryAggregator
        aggregator = TelemetryAggregator()
        summary_text = aggregator.format_for_ai(telemetry_summary)

        # Get current config
        current_config = self.runtime_scanner._get_active_config()

        # v2.3: Build tuning history for AI context
        tuning_history = {
            "iterations_completed": self.state_machine.iteration,
            "baseline_tps": self._baseline_tps,
            "current_tps": self._current_result.metrics.tps if self._current_result and self._current_result.metrics else 0,
            "applied_changes": self._tuning_history,
        }

        # v2.3: Include human feedback if provided
        human_feedback_dict = None
        if self._current_feedback:
            human_feedback_dict = {
                "text": self._current_feedback.feedback_text,
                "intent": self._current_feedback.intent,
                "timestamp": self._current_feedback.timestamp,
            }
            # Clear feedback after use
            self._current_feedback = None

        # Request analysis from AI (may return TuningProposal or SessionConclusion)
        response = self.agent.analyze_results(
            strategy=self._current_strategy,
            result=self._current_result,
            telemetry_summary=summary_text,
            current_config=current_config,
            tuning_history=tuning_history,
            human_feedback=human_feedback_dict,
        )

        # v2.3: Check response type
        if isinstance(response, SessionConclusion):
            self._session_conclusion = response
            self._current_proposal = None

            # Save conclusion
            self._save_json(
                "conclusion/session_conclusion.json",
                response.to_dict()
            )

            self._progress("Session concluded (hardware saturation detected)", 1.0)
            return False
        else:
            self._current_proposal = response
            self._session_conclusion = None

            # Save proposal
            iteration = self.state_machine.iteration
            self._save_json(
                f"tuning/proposal_iter{iteration}.json",
                asdict(self._current_proposal)
            )

            self._progress("Analysis complete", 1.0)
            return True

    def _tune(self) -> bool:
        """Phase 5: Apply tuning changes."""
        if not self._current_proposal:
            return True

        self._tuning_results = []

        for i, chunk in enumerate(self._current_proposal.tuning_chunks):
            self._progress(f"Applying: {chunk.name}", i / len(self._current_proposal.tuning_chunks))

            result = self.tuning_executor.apply_tuning_safe(chunk)
            self._tuning_results.append(result)

            if isinstance(result, TuningErrorPacket):
                # Tuning failed - executor already handled rollback
                self._save_json(
                    f"tuning/error_{chunk.id}.json",
                    asdict(result)
                )
                return False

        self._progress("Tuning complete", 1.0)
        return True

    def _verify(self) -> bool:
        """Phase 6: Verify tuning took effect."""
        self._progress("Verifying changes...", 0.0)

        all_verified = True
        for result in self._tuning_results:
            if isinstance(result, TuningResult) and not result.verified:
                all_verified = False
                break

        self._progress("Verification complete", 1.0)
        return all_verified

    def _save_json(self, relative_path: str, data: Dict):
        """Save data as JSON file."""
        import json

        path = self.config.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate session summary (v2.3 with conclusion support)."""
        summary = {
            "session_id": self._session_id,
            "state_machine": self.state_machine.get_summary(),
            "context": asdict(self._context_packet) if self._context_packet else None,
            "strategy": asdict(self._current_strategy) if self._current_strategy else None,
            "result": asdict(self._current_result) if self._current_result else None,
            "proposal": asdict(self._current_proposal) if self._current_proposal else None,
            "tuning_results": [
                asdict(r) if hasattr(r, '__dataclass_fields__') else r
                for r in self._tuning_results
            ],
            "output_dir": str(self.config.output_dir),
        }

        # v2.3: Include session conclusion if present
        if self._session_conclusion:
            summary["session_conclusion"] = self._session_conclusion.to_dict()
            summary["hardware_saturated"] = self._session_conclusion.is_hardware_limited()

        # v2.3: Include tuning history
        if self._tuning_history:
            summary["tuning_history"] = self._tuning_history
            summary["baseline_tps"] = self._baseline_tps

        return summary

    def get_proposal(self) -> Optional[TuningProposal]:
        """Get current tuning proposal (for manual review)."""
        return self._current_proposal

    def apply_proposal(self, proposal: Optional[TuningProposal] = None) -> list:
        """
        Manually apply a tuning proposal.

        Args:
            proposal: Proposal to apply (uses current if None)

        Returns:
            List of tuning results
        """
        proposal = proposal or self._current_proposal
        if not proposal:
            raise ValueError("No proposal to apply")

        results = self.tuning_executor.apply_multiple(proposal.tuning_chunks)
        self._tuning_results = results

        return results
