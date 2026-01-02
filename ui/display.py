"""
ResultDisplay - Formats results for output.

Generates:
- Markdown reports
- JSON exports
- SQL/Bash scripts
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from datetime import datetime


class ResultDisplay:
    """
    Formats and exports diagnostic results.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, summary: Dict[str, Any]) -> str:
        """
        Generate markdown report from session summary.

        Returns:
            Path to generated report
        """
        lines = []

        # Header
        lines.append("# PostgreSQL Diagnostic Report")
        lines.append("")
        lines.append(f"**Session ID:** {summary.get('session_id', 'Unknown')}")
        lines.append(f"**Generated:** {datetime.utcnow().isoformat()}")
        lines.append("")

        # State machine summary
        sm = summary.get('state_machine', {})
        lines.append("## Execution Summary")
        lines.append("")
        lines.append(f"- **Final State:** {sm.get('current_state', 'Unknown')}")
        lines.append(f"- **Iterations:** {sm.get('iteration', 0)}")
        lines.append(f"- **Total Duration:** {sm.get('total_duration_ms', 0) / 1000:.1f}s")
        lines.append("")

        # System context
        context = summary.get('context', {})
        if context:
            lines.append("## System Context")
            lines.append("")

            sys_ctx = context.get('system_context', {})
            if sys_ctx:
                lines.append("### Hardware")
                lines.append("")
                lines.append(f"- **CPU:** {sys_ctx.get('cpu_cores', '?')} cores ({sys_ctx.get('cpu_architecture', '?')})")
                lines.append(f"- **RAM:** {sys_ctx.get('ram_total_gb', '?'):.1f} GB")
                lines.append(f"- **Storage:** {sys_ctx.get('storage_topology', 'Unknown')}")
                lines.append("")

            runtime_ctx = context.get('runtime_context', {})
            if runtime_ctx:
                lines.append("### PostgreSQL Configuration")
                lines.append("")
                lines.append("```")
                active_config = runtime_ctx.get('active_config', {})
                for key, value in list(active_config.items())[:15]:
                    lines.append(f"{key} = {value}")
                if len(active_config) > 15:
                    lines.append(f"... and {len(active_config) - 15} more")
                lines.append("```")
                lines.append("")

        # Strategy
        strategy = summary.get('strategy', {})
        if strategy:
            lines.append("## Diagnostic Strategy")
            lines.append("")
            lines.append(f"**Name:** {strategy.get('name', 'Unknown')}")
            lines.append("")
            lines.append(f"**Hypothesis:** {strategy.get('hypothesis', 'None')}")
            lines.append("")

            plan = strategy.get('execution_plan', {})
            if plan:
                lines.append("### Benchmark Plan")
                lines.append("")
                lines.append(f"- Type: {plan.get('benchmark_type', 'pgbench')}")
                lines.append(f"- Scale: {plan.get('scale', 100)}")
                lines.append(f"- Clients: {plan.get('clients', 10)}")
                lines.append(f"- Duration: {plan.get('duration_seconds', 60)}s")
                lines.append("")

        # Results
        result = summary.get('result', {})
        if result:
            lines.append("## Benchmark Results")
            lines.append("")

            metrics = result.get('metrics', {})
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            if metrics.get('tps'):
                lines.append(f"| TPS | {metrics['tps']:.2f} |")
            if metrics.get('latency_avg_ms'):
                lines.append(f"| Avg Latency | {metrics['latency_avg_ms']:.2f} ms |")
            if metrics.get('latency_max_ms'):
                lines.append(f"| Max Latency | {metrics['latency_max_ms']:.2f} ms |")
            if metrics.get('transactions'):
                lines.append(f"| Transactions | {metrics['transactions']:,} |")
            lines.append("")

        # Proposal
        proposal = summary.get('proposal', {})
        if proposal:
            lines.append("## Tuning Proposal")
            lines.append("")
            lines.append(f"**Analysis:** {proposal.get('analysis_summary', 'None')}")
            lines.append("")
            lines.append(f"**Bottleneck:** {proposal.get('bottleneck_type', 'Unknown')} (confidence: {proposal.get('confidence', 0):.0%})")
            lines.append("")

            chunks = proposal.get('tuning_chunks', [])
            if chunks:
                lines.append("### Recommended Changes")
                lines.append("")

                for i, chunk in enumerate(chunks, 1):
                    lines.append(f"#### {i}. {chunk.get('name', 'Unknown')}")
                    lines.append("")
                    lines.append(f"**Category:** {chunk.get('category', 'Unknown')}")
                    lines.append("")
                    lines.append(f"**Rationale:** {chunk.get('rationale', 'None')}")
                    lines.append("")

                    if chunk.get('requires_restart'):
                        lines.append("> :warning: **Requires PostgreSQL restart**")
                        lines.append("")

                    lines.append("**Commands:**")
                    lines.append("```sql")
                    for cmd in chunk.get('apply_commands', []):
                        lines.append(cmd)
                    lines.append("```")
                    lines.append("")

            improvement = proposal.get('expected_improvement', {})
            if improvement:
                lines.append("### Expected Improvement")
                lines.append("")
                lines.append(f"- TPS increase: +{improvement.get('tps_increase_pct', 0)}%")
                lines.append(f"- Latency reduction: -{improvement.get('latency_reduction_pct', 0)}%")
                lines.append("")

        # Tuning results
        tuning_results = summary.get('tuning_results', [])
        if tuning_results:
            lines.append("## Tuning Results")
            lines.append("")

            for result in tuning_results:
                if result.get('success'):
                    lines.append(f"- :white_check_mark: {result.get('chunk_id', 'Unknown')}: Applied and verified")
                else:
                    lines.append(f"- :x: {result.get('failed_chunk_id', 'Unknown')}: Failed")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by pg_diagnose v2.2*")

        report = '\n'.join(lines)

        # Save report
        report_path = self.output_dir / "report.md"
        with open(report_path, 'w') as f:
            f.write(report)

        return str(report_path)

    def generate_tuning_sql(self, proposal: Dict[str, Any]) -> str:
        """
        Generate SQL script from tuning proposal.

        Returns:
            Path to generated script
        """
        lines = []

        lines.append("-- PostgreSQL Tuning Script")
        lines.append(f"-- Generated: {datetime.utcnow().isoformat()}")
        lines.append("-- Review carefully before executing!")
        lines.append("")
        lines.append("-- IMPORTANT: Some changes require a PostgreSQL restart.")
        lines.append("-- Run: sudo systemctl restart postgresql")
        lines.append("")

        chunks = proposal.get('tuning_chunks', [])
        for chunk in chunks:
            lines.append(f"-- {chunk.get('name', 'Unknown')}")
            lines.append(f"-- Rationale: {chunk.get('rationale', 'None')}")

            if chunk.get('requires_restart'):
                lines.append("-- WARNING: Requires restart")

            for cmd in chunk.get('apply_commands', []):
                # Include SQL commands (exclude shell commands like sudo, sysctl, echo)
                cmd_upper = cmd.upper().strip()
                is_sql = any(kw in cmd_upper for kw in [
                    'ALTER SYSTEM', 'SELECT', 'CREATE EXTENSION', 'DROP EXTENSION',
                    'SET ', 'SHOW ', 'VACUUM', 'ANALYZE', 'REINDEX'
                ])
                is_shell = any(kw in cmd_upper for kw in ['SUDO', 'SYSCTL', 'ECHO', 'SYSTEMCTL'])
                if is_sql and not is_shell:
                    cmd = cmd.rstrip(';')  # Remove trailing semicolon if present
                    lines.append(f"{cmd};")

            lines.append("")

        lines.append("-- Reload configuration")
        lines.append("SELECT pg_reload_conf();")
        lines.append("")
        lines.append("-- Verify changes")

        for chunk in chunks:
            verify_cmd = chunk.get('verification_command', '')
            if verify_cmd:
                verify_cmd = verify_cmd.rstrip(';')  # Remove trailing semicolon if present
                lines.append(f"{verify_cmd};")

        script = '\n'.join(lines)

        # Save script
        script_path = self.output_dir / "tuning" / "tuning.sql"
        script_path.parent.mkdir(parents=True, exist_ok=True)

        with open(script_path, 'w') as f:
            f.write(script)

        return str(script_path)

    def generate_tuning_bash(self, proposal: Dict[str, Any]) -> str:
        """
        Generate bash script for OS-level tuning.

        Returns:
            Path to generated script
        """
        lines = []

        lines.append("#!/bin/bash")
        lines.append("")
        lines.append("# PostgreSQL OS Tuning Script")
        lines.append(f"# Generated: {datetime.utcnow().isoformat()}")
        lines.append("# Review carefully before executing!")
        lines.append("")
        lines.append("set -e")
        lines.append("")

        chunks = proposal.get('tuning_chunks', [])
        has_os_commands = False

        for chunk in chunks:
            for cmd in chunk.get('apply_commands', []):
                # Filter for OS commands
                if any(kw in cmd.lower() for kw in ['sysctl', 'echo', 'systemctl']):
                    if not has_os_commands:
                        lines.append(f"echo '=== {chunk.get('name', 'OS Tuning')} ==='")
                        lines.append("")
                        has_os_commands = True

                    lines.append(f"# {chunk.get('rationale', '')[:60]}")
                    lines.append(cmd)
                    lines.append("")

        if has_os_commands:
            lines.append("echo 'OS tuning complete.'")
        else:
            lines.append("echo 'No OS-level tuning required.'")

        script = '\n'.join(lines)

        # Save script
        script_path = self.output_dir / "tuning" / "tuning.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)

        with open(script_path, 'w') as f:
            f.write(script)

        # Make executable
        script_path.chmod(0o755)

        return str(script_path)

    def export_json(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data as JSON file.

        Returns:
            Path to exported file
        """
        path = self.output_dir / filename

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return str(path)

    def format_for_stdout(self, summary: Dict[str, Any]) -> str:
        """
        Format summary for stdout (no Rich).
        """
        lines = []

        lines.append("=" * 60)
        lines.append(" PostgreSQL Diagnostic Summary")
        lines.append("=" * 60)

        sm = summary.get('state_machine', {})
        lines.append(f"\nFinal State: {sm.get('current_state', 'Unknown')}")
        lines.append(f"Iterations: {sm.get('iteration', 0)}")
        lines.append(f"Duration: {sm.get('total_duration_ms', 0) / 1000:.1f}s")

        result = summary.get('result', {})
        metrics = result.get('metrics', {})
        if metrics:
            lines.append(f"\nBenchmark Results:")
            lines.append(f"  TPS: {metrics.get('tps', 0):.2f}")
            lines.append(f"  Latency: {metrics.get('latency_avg_ms', 0):.2f}ms avg")

        proposal = summary.get('proposal', {})
        if proposal:
            lines.append(f"\nBottleneck: {proposal.get('bottleneck_type', 'Unknown')}")
            lines.append(f"Confidence: {proposal.get('confidence', 0):.0%}")

            chunks = proposal.get('tuning_chunks', [])
            if chunks:
                lines.append(f"\nRecommended changes ({len(chunks)}):")
                for chunk in chunks:
                    restart = " [RESTART]" if chunk.get('requires_restart') else ""
                    lines.append(f"  - {chunk.get('name', 'Unknown')}{restart}")

        lines.append(f"\nOutput: {summary.get('output_dir', 'Unknown')}")
        lines.append("")

        return '\n'.join(lines)
