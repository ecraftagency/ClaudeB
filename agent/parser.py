"""
SDLParser - Parses AI responses into protocol dataclasses.

Handles:
- JSON extraction from AI responses
- Validation against protocol schemas
- Conversion to dataclass objects
"""

import json
import re
import uuid
from typing import Dict, Any, Optional, Union
from dataclasses import fields

from ..protocol.sdl import (
    StrategyDefinition,
    ExecutionPlan,
    TelemetryRequirement,
    SuccessCriteria,
    Round1Config,
    TuningChunk as SDLTuningChunk,
    OsTuning,
)
from ..protocol.tuning import (
    TuningProposal,
    TuningChunk,
    ExpectedImprovement,
    VerificationBenchmark,
)
from ..protocol.conclusion import (
    SessionConclusion,
    TuningSummary,
    HardwareSaturationAnalysis,
    ScalingRecommendation,
)
from ..protocol.first_sight import (
    FirstSightResponse,
    StrategyOption,
)


class ParseError(Exception):
    """Failed to parse AI response."""
    pass


class ValidationError(Exception):
    """Parsed JSON doesn't match expected schema."""
    pass


class SDLParser:
    """
    Parses AI responses into protocol-compliant dataclasses.
    """

    @staticmethod
    def extract_json(response: str) -> Dict[str, Any]:
        """
        Extract JSON from AI response text.

        Handles responses with markdown code blocks or raw JSON.
        """
        # Try to find JSON in code blocks
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, response)

        if matches:
            # Use the first JSON block found
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Try parsing the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the response
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ParseError(f"Could not extract valid JSON from response: {response[:200]}...")

    @staticmethod
    def parse_strategy(response: str) -> StrategyDefinition:
        """
        Parse AI response into StrategyDefinition.

        Args:
            response: Raw AI response text

        Returns:
            StrategyDefinition dataclass

        Raises:
            ParseError: If JSON extraction fails
            ValidationError: If JSON doesn't match schema
        """
        data = SDLParser.extract_json(response)

        # Ensure required fields
        if 'id' not in data:
            data['id'] = f"strategy-{uuid.uuid4().hex[:8]}"

        # Parse execution plan
        execution_plan = None
        if 'execution_plan' in data:
            ep = data['execution_plan']
            execution_plan = ExecutionPlan(
                benchmark_type=ep.get('benchmark_type', 'pgbench'),
                custom_sql=ep.get('custom_sql'),
                scale=ep.get('scale', 100),
                clients=ep.get('clients', 10),
                threads=ep.get('threads', 2),
                duration_seconds=ep.get('duration_seconds', 60),
                warmup_seconds=ep.get('warmup_seconds', 10),
            )

        # Parse telemetry requirements
        telemetry_reqs = []
        for tr in data.get('telemetry_requirements', []):
            # Handle field name variations from AI
            metric_keys = tr.get('metric_keys') or tr.get('metrics', [])
            sampling_interval = tr.get('sampling_interval_ms') or tr.get('interval_ms', 1000)
            telemetry_reqs.append(TelemetryRequirement(
                source=tr.get('source', ''),
                metric_keys=metric_keys,
                sampling_interval_ms=sampling_interval,
            ))

        # Parse success criteria
        success_criteria = None
        if 'success_criteria' in data:
            sc = data['success_criteria']
            success_criteria = SuccessCriteria(
                target_tps=sc.get('target_tps'),
                max_latency_p99_ms=sc.get('max_latency_p99_ms'),
                min_cache_hit_ratio=sc.get('min_cache_hit_ratio'),
                vs_baseline=sc.get('vs_baseline'),
            )

        return StrategyDefinition(
            protocol_version=data.get('protocol_version', 'v2'),
            id=data['id'],
            name=data.get('name', 'Unnamed Strategy'),
            hypothesis=data.get('hypothesis', ''),
            execution_plan=execution_plan,
            telemetry_requirements=telemetry_reqs,
            success_criteria=success_criteria,
        )

    @staticmethod
    def parse_tuning_proposal(response: str) -> TuningProposal:
        """
        Parse AI response into TuningProposal.

        Args:
            response: Raw AI response text

        Returns:
            TuningProposal dataclass
        """
        data = SDLParser.extract_json(response)

        # Parse tuning chunks
        chunks = []
        for i, chunk_data in enumerate(data.get('tuning_chunks', [])):
            # Handle field name variations from AI
            depends_on = chunk_data.get('depends_on') or chunk_data.get('dependencies', [])
            priority = chunk_data.get('priority', 'MEDIUM')
            if isinstance(priority, int):
                priority = ['HIGH', 'MEDIUM', 'LOW'][min(priority, 2)]

            chunk = TuningChunk(
                id=chunk_data.get('id', f"chunk-{i:03d}"),
                category=chunk_data.get('category', 'config'),
                name=chunk_data.get('name', f"Tuning {i+1}"),
                description=chunk_data.get('description', ''),
                rationale=chunk_data.get('rationale', ''),
                apply_commands=chunk_data.get('apply_commands', []),
                rollback_commands=chunk_data.get('rollback_commands', []),
                requires_restart=chunk_data.get('requires_restart', False),
                recovery_strategy=chunk_data.get('recovery_strategy', 'SQL_REVERT'),
                target_config_file=chunk_data.get('target_config_file'),
                verification_command=chunk_data.get('verification_command', ''),
                verification_expected=chunk_data.get('verification_expected', ''),
                priority=priority,
                depends_on=depends_on,
            )
            chunks.append(chunk)

        # Parse expected improvement
        expected_improvement = None
        if 'expected_improvement' in data:
            ei = data['expected_improvement']
            expected_improvement = ExpectedImprovement(
                tps_increase_pct=ei.get('tps_increase_pct', 0),
                latency_reduction_pct=ei.get('latency_reduction_pct', 0),
            )

        # Parse verification benchmark
        verification_benchmark = None
        if 'verification_benchmark' in data:
            vb = data['verification_benchmark']
            verification_benchmark = VerificationBenchmark(
                duration_seconds=vb.get('duration_seconds', 30),
                clients=vb.get('clients', 10),
            )

        return TuningProposal(
            protocol_version=data.get('protocol_version', 'v2'),
            analysis_summary=data.get('analysis_summary', ''),
            bottleneck_type=data.get('bottleneck_type', 'unknown'),
            confidence=data.get('confidence', 0.5),
            tuning_chunks=chunks,
            expected_improvement=expected_improvement,
            verification_benchmark=verification_benchmark,
        )

    @staticmethod
    def parse_recovery_recommendation(response: str) -> Dict[str, Any]:
        """
        Parse AI response into recovery recommendation.

        Returns raw dict as recovery format is flexible.
        """
        return SDLParser.extract_json(response)

    @staticmethod
    def parse_session_conclusion(response: str) -> SessionConclusion:
        """
        Parse AI response into SessionConclusion (v2.3).

        This is triggered when AI determines tuning is complete due to:
        - Hardware saturation
        - Success criteria met
        - Diminishing returns

        Args:
            response: Raw AI response text

        Returns:
            SessionConclusion dataclass
        """
        data = SDLParser.extract_json(response)

        # Parse tuning summary
        tuning_summary = None
        if 'tuning_summary' in data and data['tuning_summary']:
            ts = data['tuning_summary']
            tuning_summary = TuningSummary(
                total_iterations=ts.get('total_iterations', 0),
                baseline_tps=ts.get('baseline_tps', 0.0),
                final_tps=ts.get('final_tps', 0.0),
                improvement_pct=ts.get('improvement_pct', 0.0),
                key_changes_applied=ts.get('key_changes_applied', []),
            )

        # Parse hardware saturation analysis
        hardware_saturation = None
        if 'hardware_saturation_analysis' in data and data['hardware_saturation_analysis']:
            hsa = data['hardware_saturation_analysis']

            # Parse scaling recommendation if present
            scaling_rec = None
            if 'scaling_recommendation' in hsa and hsa['scaling_recommendation']:
                sr = hsa['scaling_recommendation']
                scaling_rec = ScalingRecommendation(
                    action=sr.get('action', 'NONE_NEEDED'),
                    details=sr.get('details', ''),
                )

            hardware_saturation = HardwareSaturationAnalysis(
                is_saturated=hsa.get('is_saturated', False),
                bottleneck_resource=hsa.get('bottleneck_resource', 'NONE'),
                evidence=hsa.get('evidence', []),
                scaling_recommendation=scaling_rec,
            )

        return SessionConclusion(
            protocol_version=data.get('protocol_version', 'v2'),
            response_type=data.get('response_type', 'conclude_session'),
            session_id=data.get('session_id', ''),
            concluded_at=data.get('concluded_at', ''),
            tuning_summary=tuning_summary,
            hardware_saturation_analysis=hardware_saturation,
            final_report_markdown=data.get('final_report_markdown', ''),
            conclusion_reason=data.get('conclusion_reason', ''),
        )

    @staticmethod
    def detect_response_type(response: str) -> str:
        """
        Detect the type of AI response.

        Returns:
            One of: 'strategy', 'tuning', 'conclusion', 'unknown'
        """
        try:
            data = SDLParser.extract_json(response)

            # Check for response_type field first
            response_type = data.get('response_type', '')
            if response_type == 'conclude_session':
                return 'conclusion'

            # Check for conclusion indicators
            if 'hardware_saturation_analysis' in data or 'conclusion_reason' in data:
                return 'conclusion'

            # Check for tuning proposal indicators
            if 'tuning_chunks' in data or 'bottleneck_type' in data:
                return 'tuning'

            # Check for strategy indicators
            if 'execution_plan' in data or 'hypothesis' in data:
                return 'strategy'

            return 'unknown'
        except ParseError:
            return 'unknown'

    @staticmethod
    def validate_strategy(strategy: StrategyDefinition) -> list:
        """
        Validate a StrategyDefinition for common issues.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        if not strategy.hypothesis:
            issues.append("Warning: No hypothesis specified")

        if strategy.execution_plan:
            ep = strategy.execution_plan
            if ep.clients > 1000:
                issues.append("Warning: Very high client count (>1000)")
            if ep.duration_seconds < 10:
                issues.append("Warning: Very short duration (<10s)")
            if ep.scale < 1:
                issues.append("Error: Scale must be >= 1")

        if strategy.success_criteria:
            sc = strategy.success_criteria
            if sc.target_tps and sc.target_tps > 1000000:
                issues.append("Warning: Unrealistic TPS target (>1M)")

        return issues

    @staticmethod
    def validate_tuning_proposal(proposal: TuningProposal) -> list:
        """
        Validate a TuningProposal for safety issues.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        if not proposal.tuning_chunks:
            issues.append("Error: No tuning chunks provided")
            return issues

        for chunk in proposal.tuning_chunks:
            # Check for dangerous commands
            for cmd in chunk.apply_commands:
                cmd_upper = cmd.upper()
                if 'DROP' in cmd_upper:
                    issues.append(f"Error: DROP command in chunk {chunk.id}")
                if 'TRUNCATE' in cmd_upper:
                    issues.append(f"Error: TRUNCATE command in chunk {chunk.id}")
                if 'DELETE FROM' in cmd_upper and 'WHERE' not in cmd_upper:
                    issues.append(f"Warning: DELETE without WHERE in chunk {chunk.id}")

            # Check for missing rollback
            if chunk.apply_commands and not chunk.rollback_commands:
                issues.append(f"Warning: No rollback commands for chunk {chunk.id}")

            # Check verification
            if not chunk.verification_command:
                issues.append(f"Warning: No verification command for chunk {chunk.id}")

        return issues

    @staticmethod
    def parse_first_sight(response: str) -> FirstSightResponse:
        """
        Parse AI response into FirstSightResponse.

        Args:
            response: Raw AI response text

        Returns:
            FirstSightResponse dataclass
        """
        data = SDLParser.extract_json(response)

        # Parse strategy options
        strategy_options = []
        for opt in data.get('strategy_options', []):
            strategy_options.append(StrategyOption(
                id=opt.get('id', ''),
                name=opt.get('name', ''),
                goal=opt.get('goal', ''),
                hypothesis=opt.get('hypothesis', ''),
                target_kpis=opt.get('target_kpis', {}),
                rationale=opt.get('rationale', ''),
                estimated_duration_minutes=opt.get('estimated_duration_minutes', 5),
                risk_level=opt.get('risk_level', 'LOW'),
            ))

        return FirstSightResponse(
            protocol_version=data.get('protocol_version', 'v2'),
            system_overview=data.get('system_overview', ''),
            schema_overview=data.get('schema_overview', ''),
            key_observations=data.get('key_observations', []),
            warnings=data.get('warnings', []),
            strategy_options=strategy_options,
        )

    @staticmethod
    def parse_round1_config(response: str) -> Round1Config:
        """
        Parse AI response into Round1Config.

        Args:
            response: Raw AI response text

        Returns:
            Round1Config dataclass
        """
        data = SDLParser.extract_json(response)

        # Parse tuning chunks
        tuning_chunks = []
        for i, chunk_data in enumerate(data.get('tuning_chunks', [])):
            chunk = SDLTuningChunk(
                id=chunk_data.get('id', f"chunk-{i:03d}"),
                category=chunk_data.get('category', 'config'),
                name=chunk_data.get('name', f"Tuning {i+1}"),
                description=chunk_data.get('description', ''),
                apply_commands=chunk_data.get('apply_commands', []),
                rollback_commands=chunk_data.get('rollback_commands', []),
                requires_restart=chunk_data.get('requires_restart', False),
                verification_command=chunk_data.get('verification_command'),
                verification_expected=chunk_data.get('verification_expected'),
                priority=chunk_data.get('priority', 'MEDIUM'),
            )
            tuning_chunks.append(chunk)

        # Parse OS tuning
        os_tuning = []
        for i, ot_data in enumerate(data.get('os_tuning', [])):
            ot = OsTuning(
                id=ot_data.get('id', f"os-{i:03d}"),
                name=ot_data.get('name', f"OS Tuning {i+1}"),
                description=ot_data.get('description', ''),
                apply_command=ot_data.get('apply_command', ''),
                rollback_command=ot_data.get('rollback_command', ''),
                persistent_file=ot_data.get('persistent_file'),
                persistent_line=ot_data.get('persistent_line'),
            )
            os_tuning.append(ot)

        return Round1Config(
            protocol_version=data.get('protocol_version', 'v2'),
            response_type=data.get('response_type', 'round1_config'),
            rationale=data.get('rationale', ''),
            tuning_chunks=tuning_chunks,
            os_tuning=os_tuning,
            restart_required=data.get('restart_required', False),
            restart_reason=data.get('restart_reason', ''),
        )
