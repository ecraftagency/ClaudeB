"""
CLI - Command-line interface for pg_diagnose.

Phase 1: Init workflow, DB selection, AI first-sight analysis.
Phase 2: Strategy selection, benchmark execution, result analysis.
"""

import argparse
import sys
import os
import warnings
import time
import threading
import subprocess
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from .commands import SessionState, CommandHandler, create_command_handler
from .dashboard import ProgressTimeline, DiffView, SafetyDisplay
from .workspace import (
    WorkspaceManager, Workspace, Session,
    SessionState as WorkspaceSessionState,
    SessionPhase, SessionStateMachine, InvalidStateTransition
)
from .ui import SlashCommandHandler, CommandResult, StatusLineManager

# Suppress Google API warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
os.environ["GRPC_VERBOSITY"] = "ERROR"

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.markdown import Markdown
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered PostgreSQL Diagnostic & Tuning Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    pg_diagnose -H 10.0.0.21 -U postgres
    pg_diagnose -H dbserver -p 5432 -U postgres

    # Quick health check
    pg_diagnose -H 10.0.0.21 --health

    # Watch mode (live monitoring)
    pg_diagnose -H 10.0.0.21 --watch

    # Dry-run (recommendations only)
    pg_diagnose -H 10.0.0.21 --dry-run --output recs.json

    # Auto mode (non-interactive)
    pg_diagnose -H 10.0.0.21 --auto --target-tps 6000 --max-rounds 5

    # Session management
    pg_diagnose --list-sessions
    pg_diagnose -H 10.0.0.21 --resume my-session
    pg_diagnose -H 10.0.0.21 --save-session my-session

    # Export
    pg_diagnose -H 10.0.0.21 --export-config > postgresql.conf
    pg_diagnose -H 10.0.0.21 --export-ansible > playbook.yml

Environment Variables:
    GEMINI_API_KEY    Gemini API key (required)
    PGPASSWORD        PostgreSQL password
        """,
    )

    # Database connection
    parser.add_argument(
        "-H", "--host",
        help="PostgreSQL host"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=5432,
        help="PostgreSQL port (default: 5432)"
    )
    parser.add_argument(
        "-U", "--user",
        default="postgres",
        help="PostgreSQL user (default: postgres)"
    )
    parser.add_argument(
        "-W", "--password",
        help="PostgreSQL password (or use PGPASSWORD env var)"
    )
    parser.add_argument(
        "-d", "--database",
        help="Database name (skip selection prompt)"
    )

    # SSH for remote operations (optional)
    parser.add_argument(
        "--ssh-host", "--ssh",
        dest="ssh_host",
        help="SSH connection for OS-level tuning (format: user@host or just host, default user: ubuntu)"
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=22,
        help="SSH port (default: 22)"
    )

    # ==================== Operation Modes ====================
    mode_group = parser.add_argument_group('Operation Modes')

    mode_group.add_argument(
        "--health",
        action="store_true",
        help="Quick health check mode"
    )
    mode_group.add_argument(
        "--watch",
        action="store_true",
        help="Live monitoring mode"
    )
    mode_group.add_argument(
        "--watch-interval",
        type=int,
        default=5,
        help="Watch mode refresh interval in seconds (default: 5)"
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show recommendations without applying"
    )
    mode_group.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run analysis and exit (alias for --dry-run)"
    )
    mode_group.add_argument(
        "--dashboard",
        action="store_true",
        help="Use live TUI dashboard mode"
    )

    # ==================== Auto Mode ====================
    auto_group = parser.add_argument_group('Auto Mode (Non-Interactive)')

    auto_group.add_argument(
        "--auto",
        action="store_true",
        help="Automatic tuning mode (non-interactive)"
    )
    auto_group.add_argument(
        "--target-tps",
        type=float,
        default=0,
        help="Target TPS for auto mode"
    )
    auto_group.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum tuning rounds for auto mode (default: 5)"
    )
    auto_group.add_argument(
        "--risk",
        choices=['low', 'medium', 'high'],
        default='low',
        help="Risk level for auto mode (default: low)"
    )

    # ==================== Session Management ====================
    session_group = parser.add_argument_group('Session Management')

    session_group.add_argument(
        "--save-session",
        metavar="NAME",
        help="Save session with given name"
    )
    session_group.add_argument(
        "--resume",
        metavar="NAME",
        help="Resume a saved session"
    )
    session_group.add_argument(
        "--list-sessions",
        action="store_true",
        help="List saved sessions"
    )

    # ==================== Export Options ====================
    export_group = parser.add_argument_group('Export Options')

    export_group.add_argument(
        "--export",
        choices=['markdown', 'json', 'postgresql', 'sql', 'ansible', 'terraform'],
        help="Export session in specified format"
    )
    export_group.add_argument(
        "--export-config",
        action="store_true",
        help="Export PostgreSQL configuration (shorthand for --export postgresql)"
    )
    export_group.add_argument(
        "--export-ansible",
        action="store_true",
        help="Export Ansible playbook (shorthand for --export ansible)"
    )
    export_group.add_argument(
        "-o", "--output",
        help="Output file for exports (default: stdout)"
    )

    # ==================== Test Mode ====================
    test_group = parser.add_argument_group('Test Mode (For Automated Testing)')

    test_group.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode with structured JSON output"
    )
    test_group.add_argument(
        "--mock-benchmark",
        action="store_true",
        help="Use mock benchmark results (no actual pgbench)"
    )
    test_group.add_argument(
        "--mock-ai",
        action="store_true",
        help="Use mock AI agent (no actual Gemini API calls)"
    )
    test_group.add_argument(
        "--test-scenario",
        choices=['balanced_tps', 'wal_optimized', 'error_scenario'],
        default='balanced_tps',
        help="Test scenario for mock data (default: balanced_tps)"
    )
    test_group.add_argument(
        "--input-file",
        metavar="FILE",
        help="Read commands from file instead of stdin"
    )
    test_group.add_argument(
        "--output-json",
        action="store_true",
        help="Output all responses as JSON (for test verification)"
    )
    test_group.add_argument(
        "--state-dump",
        action="store_true",
        help="Dump workspace/session state after each command"
    )
    test_group.add_argument(
        "--test-workspace",
        metavar="PATH",
        help="Use specific workspace directory for testing"
    )

    return parser.parse_args()


class DiagnoseUI:
    """UI for PostgreSQL Diagnostic Tool."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self._live = None
        self.command_handler: Optional[CommandHandler] = None
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.status_line_manager: Optional[StatusLineManager] = None
        self.slash_handler: Optional[SlashCommandHandler] = None

    def set_command_handler(self, handler: CommandHandler):
        """Set the command handler for slash command support."""
        self.command_handler = handler

    def set_workspace_manager(self, manager: WorkspaceManager):
        """Set workspace manager and initialize related UI components."""
        self.workspace_manager = manager
        self.status_line_manager = StatusLineManager(self.console)
        self.slash_handler = SlashCommandHandler(manager, self.console)

    def show_status_line(self):
        """Show the workspace status line."""
        if self.status_line_manager and self.workspace_manager:
            self.status_line_manager.update_from_workspace(self.workspace_manager)
            self.status_line_manager.show()

    def prompt(self, message: str, allow_commands: bool = True) -> str:
        """
        Get user input with optional slash command support.

        If input starts with '/', it's processed as a command and
        this method prompts again for actual input.
        """
        while True:
            user_input = input(message).strip()

            # Check for slash commands
            if allow_commands and user_input.startswith('/'):
                # First try workspace slash commands
                if self.slash_handler and self.slash_handler.is_command(user_input):
                    result, msg = self.slash_handler.execute(user_input)
                    self.print(msg)

                    if result == CommandResult.EXIT:
                        sys.exit(0)

                    continue  # Prompt again after command

                # Fall back to legacy command handler
                if self.command_handler:
                    self.command_handler.parse_and_execute(user_input)
                    continue  # Prompt again after command

            return user_input

    def print(self, message: str = ""):
        """Print a message."""
        if self.console:
            self.console.print(message)
        else:
            print(message)

    def print_banner(self):
        """Print application banner."""
        if self.console:
            banner = "[bold cyan]PostgreSQL Diagnostic Tool[/] [dim]v2.2[/]\n[dim]AI-Powered Performance Analysis & Tuning[/]"
            self.console.print(Panel(banner, border_style="cyan"))
        else:
            print("=" * 50)
            print("PostgreSQL Diagnostic Tool v2.2")
            print("AI-Powered Performance Analysis & Tuning")
            print("=" * 50)
        self.print()

    def print_error(self, message: str):
        """Print error message."""
        if self.console:
            self.console.print(f"[bold red]Error:[/] {message}")
        else:
            print(f"Error: {message}")

    def show_context_menu(self, context: str):
        """
        Display context-sensitive slash commands.

        Contexts:
            workspace - Workspace main menu
            session_list - Session selection
            strategy - Strategy selection
            strategy_custom - Strategy customization (command-first)
            benchmark_confirm - Confirm benchmark execution
            target_confirm - Confirm target TPS
            tuning_apply - Confirm applying tuning changes
            tuning_round - DBA prompt at start/during round (command-first)
            restart_confirm - Confirm PostgreSQL restart
            session_end - Session ending options
        """
        menus = {
            'workspace': [
                ('/new', 'Start new tuning session'),
                ('/sessions', 'List all sessions'),
                ('/resume', 'Resume a paused session'),
                ('/retry', 'Retry a failed session'),
                ('/status', 'Show workspace status'),
                ('/quit', 'Exit tool'),
            ],
            'session_list': [
                ('/resume', 'Resume paused session'),
                ('/retry', 'Retry failed session'),
                ('/back', 'Return to workspace'),
                ('/quit', 'Exit tool'),
            ],
            'strategy': [
                ('/back', 'Return to workspace'),
                ('/refresh', 'Get new AI strategies'),
                ('/quit', 'Exit tool'),
            ],
            'strategy_custom': [
                ('/go', 'Continue with AI recommendations'),
                ('/custom', 'Provide custom instructions'),
                ('/back', 'Return to strategy list'),
                ('/quit', 'Exit tool'),
            ],
            'benchmark_confirm': [
                ('/run', 'Run the benchmark'),
                ('/skip', 'Skip benchmark'),
                ('/back', 'Return to strategy selection'),
                ('/quit', 'Exit tool'),
            ],
            'target_confirm': [
                ('/accept', 'Accept suggested target'),
                ('/set', 'Set custom target (e.g. /set 5000)'),
                ('/back', 'Return to strategy selection'),
            ],
            'tuning_apply': [
                ('/apply', 'Apply these changes'),
                ('/details', 'Show change details'),
                ('/skip', 'Skip this round'),
                ('/stop', 'Pause session'),
            ],
            'tuning_round': [
                ('/go', 'Continue with AI recommendations'),
                ('/custom', 'Provide custom instructions'),
                ('/status', 'Show session status'),
                ('/history', 'Show tuning history'),
                ('/done', 'End session with current results'),
                ('/stop', 'Pause session'),
            ],
            'restart_confirm': [
                ('/restart', 'Restart PostgreSQL now'),
                ('/skip', 'Skip restart (changes pending)'),
                ('/stop', 'Pause session'),
            ],
            'session_end': [
                ('/new', 'Start new session'),
                ('/retry', 'Retry with different strategy'),
                ('/export', 'Export session report'),
                ('/quit', 'Exit tool'),
            ],
            'error_recovery': [
                ('/retry', 'Retry from last checkpoint'),
                ('/skip', 'Skip and continue'),
                ('/stop', 'Pause session'),
                ('/quit', 'Exit tool'),
            ],
        }

        commands = menus.get(context, [])
        if not commands:
            return

        if self.console:
            self.console.print()
            self.console.print("[dim]Commands:[/]", end="  ")
            cmd_strs = [f"[cyan]{cmd}[/][dim]={desc}[/]" for cmd, desc in commands]
            self.console.print("  ".join(cmd_strs))
        else:
            print()
            print("Commands: " + "  ".join([f"{cmd}={desc}" for cmd, desc in commands]))

    def spinner_start(self, message: str):
        """Start a spinner with message."""
        if self.console and RICH_AVAILABLE:
            from rich.status import Status
            self._live = Status(f"[cyan]{message}[/]", spinner="dots", console=self.console)
            self._live.start()
        else:
            print(f"... {message}", end="", flush=True)

    def spinner_stop(self, success_message: str = "Done"):
        """Stop the spinner."""
        if self._live:
            self._live.stop()
            self._live = None
            if self.console:
                self.console.print(f"[green]{success_message}[/]")
        else:
            print(f" {success_message}")

    def display_db_config(self, config: Dict[str, str], title: str = "Database Configuration"):
        """Display database configuration as a 2-column table."""
        if self.console:
            table = Table(title=title, box=box.ROUNDED, show_header=True)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")

            for key, value in config.items():
                table.add_row(key, str(value))

            self.console.print(table)
        else:
            print(f"\n{title}")
            print("-" * 60)
            for key, value in config.items():
                print(f"  {key}: {value}")

    # ===================== Phase 2 Methods =====================

    def select_strategy(self, strategy_options: List) -> int:
        """Let user select a benchmark strategy. Returns index (0-based) or -1 for back."""
        self.print()
        self.show_context_menu('strategy')
        while True:
            try:
                choice = self.prompt("Select strategy number: ").strip()
                # Handle slash commands
                if choice.lower() in ['/quit', '/q', 'q']:
                    sys.exit(0)
                if choice.lower() in ['/back', '/b']:
                    return -1  # Signal to go back
                if choice.lower() in ['/refresh', '/r']:
                    return -2  # Signal to refresh strategies
                idx = int(choice) - 1
                if 0 <= idx < len(strategy_options):
                    return idx
                print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a number.")

    def get_dba_customization(self) -> Optional[str]:
        """Get optional DBA customization message. Returns None for go, 'BACK' for back."""
        self.print()
        if self.console:
            self.console.print("[cyan]Strategy Customization[/]")
            self.console.print("[dim]You can provide custom instructions for the AI to consider.[/]")
        else:
            print("Strategy Customization")
            print("You can provide custom instructions for the AI.")

        self.show_context_menu('strategy_custom')

        while True:
            cmd = self.prompt("> ").strip().lower()

            if cmd in ['/go', '/g', '']:
                return None  # Continue without custom instructions

            if cmd in ['/back', '/b']:
                return 'BACK'

            if cmd in ['/quit', '/q']:
                import sys
                sys.exit(0)

            if cmd.startswith('/custom'):
                # Extract inline text or prompt for it
                inline_text = cmd[7:].strip()  # Remove '/custom'
                if inline_text:
                    return inline_text
                # Prompt for custom text
                if self.console:
                    self.console.print("[dim]Enter your custom instructions:[/]")
                else:
                    print("Enter your custom instructions:")
                custom_text = self.prompt("Instructions: ", allow_commands=False).strip()
                if custom_text:
                    return custom_text
                self.print("[yellow]No instructions provided, continuing...[/]" if self.console else "No instructions provided.")
                return None

            # Unknown command
            self.print(f"[yellow]Unknown command: {cmd}. Type /go to continue or /custom for instructions.[/]" if self.console else f"Unknown command: {cmd}")

    def confirm_benchmark(self, strategy) -> bool:
        """Show benchmark details and confirm execution. Returns True to run, 'BACK', 'SKIP', or False."""
        self.print()
        if self.console:
            self.console.rule("[bold blue]Benchmark Configuration[/]")

            table = Table(box=box.ROUNDED, show_header=False)
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Strategy", strategy.name)
            table.add_row("Hypothesis", strategy.hypothesis)

            if strategy.execution_plan:
                ep = strategy.execution_plan
                table.add_row("Benchmark Type", ep.benchmark_type)
                table.add_row("Scale Factor", str(ep.scale))
                table.add_row("Clients", str(ep.clients))
                table.add_row("Duration", f"{ep.duration_seconds}s")
                if ep.custom_sql:
                    table.add_row("Custom SQL", ep.custom_sql[:100] + "..." if len(ep.custom_sql) > 100 else ep.custom_sql)

            self.console.print(table)
        else:
            print("\n" + "=" * 60)
            print("BENCHMARK CONFIGURATION")
            print("=" * 60)
            print(f"  Strategy: {strategy.name}")
            print(f"  Hypothesis: {strategy.hypothesis}")
            if strategy.execution_plan:
                ep = strategy.execution_plan
                print(f"  Type: {ep.benchmark_type}")
                print(f"  Scale: {ep.scale}")
                print(f"  Clients: {ep.clients}")
                print(f"  Duration: {ep.duration_seconds}s")

        self.print()
        self.show_context_menu('benchmark_confirm')

        while True:
            cmd = self.prompt("> ").strip().lower()

            if cmd in ['/run', '/r', 'y', 'yes']:
                return True

            if cmd in ['/quit', '/q']:
                sys.exit(0)

            if cmd in ['/back', '/b']:
                return 'BACK'

            if cmd in ['/skip', '/s']:
                return 'SKIP'

            if cmd in ['n', 'no', '']:
                return False

            self.print(f"[yellow]Unknown command: {cmd}. Type /run to execute or /skip to skip.[/]" if self.console else f"Unknown command: {cmd}")

    def display_benchmark_config(self, strategy, round_num: int = None):
        """Display benchmark configuration (compact, no confirmation)."""
        if self.console:
            # Compact inline display for tuning rounds
            if strategy.execution_plan:
                ep = strategy.execution_plan
                config_parts = [
                    f"[cyan]Benchmark:[/] {ep.benchmark_type}",
                    f"scale={ep.scale}",
                    f"clients={ep.clients}",
                    f"duration={ep.duration_seconds}s",
                ]
                if ep.custom_sql:
                    sql_preview = ep.custom_sql[:60].replace('\n', ' ')
                    config_parts.append(f"sql=\"{sql_preview}...\"")
                self.console.print("  " + " | ".join(config_parts))
        else:
            if strategy.execution_plan:
                ep = strategy.execution_plan
                self.print(f"  Benchmark: {ep.benchmark_type} | scale={ep.scale} | clients={ep.clients} | duration={ep.duration_seconds}s")

    def display_benchmark_progress(self, duration_seconds: int, status_callback=None):
        """Display benchmark progress with live updates."""
        if self.console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("[cyan]Running benchmark...", total=duration_seconds)

                for i in range(duration_seconds):
                    time.sleep(1)
                    progress.update(task, advance=1)

                    # Update description with live stats if callback provided
                    if status_callback:
                        status = status_callback()
                        if status:
                            progress.update(task, description=f"[cyan]{status}")
        else:
            print(f"Running benchmark ({duration_seconds}s)...", end="", flush=True)
            for i in range(duration_seconds):
                time.sleep(1)
                if (i + 1) % 10 == 0:
                    print(f" {i+1}s", end="", flush=True)
            print(" Done")

    def display_benchmark_result(self, result, target_kpis: Dict[str, Any]):
        """Display benchmark results with KPI comparison."""
        self.print()

        # Show error if benchmark failed
        if hasattr(result, 'error') and result.error:
            if self.console:
                self.console.print(f"[red]Benchmark Error:[/] {result.error}")
            else:
                self.print(f"Benchmark Error: {result.error}")

        # Show raw output snippet if TPS is 0 (helps debug)
        if hasattr(result, 'raw_output') and result.raw_output:
            metrics = getattr(result, 'metrics', None)
            if metrics and getattr(metrics, 'tps', 0) == 0:
                if self.console:
                    self.console.print("[yellow]Debug: Benchmark output (last 500 chars):[/]")
                    self.console.print(f"[dim]{result.raw_output[-500:]}[/]")
                else:
                    self.print("Debug output: " + result.raw_output[-500:])

        # Extract metrics from BenchmarkResult or dict
        if hasattr(result, 'metrics') and result.metrics:
            metrics = result.metrics
            tps = getattr(metrics, 'tps', 0) or 0
            latency = getattr(metrics, 'latency_avg_ms', 0) or 0
            p99 = getattr(metrics, 'latency_max_ms', 0) or 0  # Using max as proxy for p99
            txn = getattr(metrics, 'transactions', 0) or 0
        elif isinstance(result, dict):
            tps = result.get('tps', 0) or 0
            latency = result.get('latency_avg_ms', 0) or 0
            p99 = result.get('latency_p99_ms', 0) or 0
            txn = result.get('transactions', 0) or 0
        else:
            tps = latency = p99 = txn = 0

        target_tps = target_kpis.get('target_tps', 0)
        target_latency = target_kpis.get('max_latency_ms', 0)

        if self.console:
            self.console.rule("[bold blue]Benchmark Results[/]")

            # Main metrics table
            table = Table(title="Performance Metrics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Result", style="green")
            table.add_column("Target", style="yellow")
            table.add_column("Status", style="bold")

            # TPS
            tps_status = "[green]✓ HIT[/]" if tps >= target_tps else "[red]✗ MISS[/]"
            table.add_row("TPS", f"{tps:.2f}", str(target_tps), tps_status)

            # Latency
            if target_latency:
                lat_status = "[green]✓ HIT[/]" if latency <= target_latency else "[red]✗ MISS[/]"
                table.add_row("Avg Latency", f"{latency:.2f}ms", f"<{target_latency}ms", lat_status)

            # P99 Latency
            if p99:
                table.add_row("P99 Latency", f"{p99:.2f}ms", "-", "")

            self.console.print(table)

            # Transaction counts
            if txn:
                self.console.print(f"\n[dim]Total Transactions:[/] {txn:,}")

        else:
            print("\n" + "=" * 60)
            print("BENCHMARK RESULTS")
            print("=" * 60)
            print(f"  TPS: {tps:.2f}")
            print(f"  Target: {target_tps}")
            print(f"  Latency: {latency:.2f}ms")

    def display_ai_analysis(self, analysis):
        """Display AI analysis response."""
        self.print()
        if self.console:
            self.console.rule("[bold blue]AI Analysis[/]")

            if hasattr(analysis, 'analysis_summary'):
                self.console.print(Panel(analysis.analysis_summary, title="Summary", border_style="blue"))

            if hasattr(analysis, 'bottleneck_type'):
                self.console.print(f"\n[cyan]Bottleneck:[/] {analysis.bottleneck_type}")

            if hasattr(analysis, 'confidence'):
                self.console.print(f"[cyan]Confidence:[/] {analysis.confidence * 100:.0f}%")

            if hasattr(analysis, 'tuning_chunks') and analysis.tuning_chunks:
                self.console.print(f"\n[cyan]Recommended Tuning Changes:[/] {len(analysis.tuning_chunks)}")
                for i, chunk in enumerate(analysis.tuning_chunks, 1):
                    self.console.print(f"  {i}. {chunk.name} ({chunk.category})")
        else:
            print("\n" + "=" * 60)
            print("AI ANALYSIS")
            print("=" * 60)
            if hasattr(analysis, 'analysis_summary'):
                print(f"\nSummary: {analysis.analysis_summary}")
            if hasattr(analysis, 'bottleneck_type'):
                print(f"Bottleneck: {analysis.bottleneck_type}")

    def display_round1_config(self, config) -> bool:
        """Display Round 1 configuration and ask for confirmation."""
        self.print()
        if self.console:
            self.console.rule("[bold blue]Round 1 Configuration[/]")
            self.console.print(Panel(config.rationale, title="Rationale", border_style="blue"))

            # PostgreSQL tuning
            if config.tuning_chunks:
                self.console.print("\n[cyan]PostgreSQL Configuration Changes:[/]")
                table = Table(box=box.ROUNDED, show_header=True)
                table.add_column("#", style="cyan", width=3)
                table.add_column("Name", style="green")
                table.add_column("Category", style="yellow")
                table.add_column("Restart", style="red")

                for i, chunk in enumerate(config.tuning_chunks, 1):
                    restart = "[red]Yes[/]" if chunk.requires_restart else "[green]No[/]"
                    table.add_row(str(i), chunk.name, chunk.category, restart)

                self.console.print(table)

                # Show details with specific config flags
                for chunk in config.tuning_chunks:
                    self.console.print(f"\n  [bold]{chunk.name}:[/]")
                    if chunk.description:
                        self.console.print(f"    [dim]{chunk.description}[/]")
                    # Show all ALTER SYSTEM commands prominently
                    for cmd in chunk.apply_commands:
                        if 'ALTER SYSTEM SET' in cmd.upper():
                            self.console.print(f"    [yellow]→ {cmd}[/]")
                        elif 'pg_reload_conf' not in cmd.lower():
                            self.console.print(f"    [dim]→ {cmd}[/]")

            # OS tuning
            if config.os_tuning:
                self.console.print("\n[cyan]OS Configuration Changes:[/]")
                for ot in config.os_tuning:
                    self.console.print(f"  [yellow]•[/] {ot.name}")
                    self.console.print(f"    [dim]{ot.description}[/]")
                    self.console.print(f"    [dim]→[/] {ot.apply_command}")

            # Restart warning
            if config.restart_required:
                self.console.print(f"\n[bold yellow]⚠ PostgreSQL restart required:[/] {config.restart_reason}")

        else:
            print("\n" + "=" * 60)
            print("ROUND 1 CONFIGURATION")
            print("=" * 60)
            print(f"\nRationale: {config.rationale}")

            if config.tuning_chunks:
                print("\nPostgreSQL Changes:")
                for i, chunk in enumerate(config.tuning_chunks, 1):
                    print(f"  {i}. {chunk.name} ({chunk.category})")
                    for cmd in chunk.apply_commands:
                        if 'ALTER SYSTEM SET' in cmd.upper():
                            print(f"     → {cmd}")

            if config.os_tuning:
                print("\nOS Changes:")
                for ot in config.os_tuning:
                    print(f"  • {ot.name}")

            if config.restart_required:
                print(f"\n⚠ Restart required: {config.restart_reason}")

        # Check if there are any changes
        if not config.tuning_chunks and not config.os_tuning:
            self.print("\n[green]No configuration changes needed - current config is optimal for this benchmark.[/]" if self.console else "\nNo configuration changes needed.")
            return True  # Proceed without changes

        self.print()
        confirm = self.prompt("Apply these configurations? [y/N]: ").lower()
        return confirm == 'y'

    def display_databases(self, databases: List[Dict]) -> None:
        """Display database list."""
        if self.console:
            table = Table(title="Available Databases", box=box.ROUNDED)
            table.add_column("#", style="cyan", width=3)
            table.add_column("Database", style="green")
            table.add_column("Size", style="yellow")
            table.add_column("Owner", style="dim")

            for i, db in enumerate(databases, 1):
                table.add_row(str(i), db['name'], db['size'], db['owner'])

            self.console.print(table)
        else:
            print("\nAvailable Databases:")
            print("-" * 50)
            for i, db in enumerate(databases, 1):
                print(f"  {i}. {db['name']} ({db['size']}) - owner: {db['owner']}")

    def select_database(self, databases: List[Dict]) -> str:
        """Interactively select database."""
        if not databases:
            self.print_error("No accessible databases found.")
            sys.exit(1)

        self.display_databases(databases)

        while True:
            try:
                choice = self.prompt("\nSelect database number (or 'q' to quit): ")
                if choice.lower() == 'q':
                    sys.exit(0)
                idx = int(choice) - 1
                if 0 <= idx < len(databases):
                    return databases[idx]['name']
                print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a number.")

    def display_first_sight(self, response) -> None:
        """Display the FirstSightResponse from AI."""
        if self.console:
            self._display_first_sight_rich(response)
        else:
            self._display_first_sight_plain(response)

    def _display_first_sight_rich(self, response) -> None:
        """Rich display of first sight analysis."""
        # System Overview
        self.console.print()
        self.console.rule("[bold blue]System Overview[/]")
        self.console.print(response.system_overview)

        # Schema Overview
        self.console.print()
        self.console.rule("[bold blue]Schema Overview[/]")
        self.console.print(response.schema_overview)

        # Key Observations
        if response.key_observations:
            self.console.print()
            self.console.rule("[bold blue]Key Observations[/]")
            for obs in response.key_observations:
                self.console.print(f"  [green]•[/] {obs}")

        # Warnings
        if response.warnings:
            self.console.print()
            self.console.rule("[bold yellow]Warnings[/]")
            for warning in response.warnings:
                self.console.print(f"  [yellow]⚠[/] {warning}")

        # Strategy Options
        if response.strategy_options:
            self.console.print()
            self.console.rule("[bold blue]Suggested Benchmark Strategies[/]")
            self.console.print()

            for i, opt in enumerate(response.strategy_options, 1):
                # Risk level color
                risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(opt.risk_level, "white")

                tree = Tree(f"[bold cyan]{i}. {opt.name}[/]")
                tree.add(f"[dim]Goal:[/] {opt.goal}")
                tree.add(f"[dim]Hypothesis:[/] {opt.hypothesis}")

                # KPIs
                kpi_str = ", ".join(f"{k}: {v}" for k, v in opt.target_kpis.items())
                tree.add(f"[dim]Target KPIs:[/] {kpi_str}")

                tree.add(f"[dim]Rationale:[/] {opt.rationale}")
                tree.add(f"[dim]Duration:[/] ~{opt.estimated_duration_minutes} minutes")
                tree.add(f"[dim]Risk Level:[/] [{risk_color}]{opt.risk_level}[/]")

                self.console.print(tree)
                self.console.print()

    def _display_first_sight_plain(self, response) -> None:
        """Plain text display of first sight analysis."""
        print("\n" + "=" * 60)
        print("SYSTEM OVERVIEW")
        print("=" * 60)
        print(response.system_overview)

        print("\n" + "=" * 60)
        print("SCHEMA OVERVIEW")
        print("=" * 60)
        print(response.schema_overview)

        if response.key_observations:
            print("\n" + "=" * 60)
            print("KEY OBSERVATIONS")
            print("=" * 60)
            for obs in response.key_observations:
                print(f"  • {obs}")

        if response.warnings:
            print("\n" + "=" * 60)
            print("WARNINGS")
            print("=" * 60)
            for warning in response.warnings:
                print(f"  ! {warning}")

        if response.strategy_options:
            print("\n" + "=" * 60)
            print("SUGGESTED BENCHMARK STRATEGIES")
            print("=" * 60)
            for i, opt in enumerate(response.strategy_options, 1):
                print(f"\n{i}. {opt.name}")
                print(f"   Goal: {opt.goal}")
                print(f"   Hypothesis: {opt.hypothesis}")
                print(f"   Target KPIs: {opt.target_kpis}")
                print(f"   Rationale: {opt.rationale}")
                print(f"   Duration: ~{opt.estimated_duration_minutes} minutes")
                print(f"   Risk Level: {opt.risk_level}")


def create_connection(host: str, port: int, user: str, password: str, database: str):
    """Create PostgreSQL connection."""
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2 is required. Install with: pip install psycopg2-binary")
        sys.exit(1)

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=database,
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        sys.exit(1)


def list_databases(host: str, port: int, user: str, password: str) -> List[Dict]:
    """List all accessible databases."""
    conn = create_connection(host, port, user, password, "postgres")

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    d.datname,
                    pg_size_pretty(pg_database_size(d.datname)) as size,
                    pg_database_size(d.datname) as size_bytes,
                    u.usename as owner
                FROM pg_database d
                JOIN pg_user u ON d.datdba = u.usesysid
                WHERE d.datistemplate = false
                ORDER BY pg_database_size(d.datname) DESC
            """)
            databases = []
            for row in cur.fetchall():
                databases.append({
                    'name': row[0],
                    'size': row[1],
                    'size_bytes': row[2],
                    'owner': row[3]
                })
            return databases
    finally:
        conn.close()


def run_discovery(conn, ssh_config: Optional[Dict] = None):
    """Run discovery phase to gather system and schema context."""
    from .discovery.system import SystemScanner, SystemScannerConfig
    from .discovery.schema import SchemaScanner
    from .discovery.runtime import RuntimeScanner
    from .protocol.context import ContextPacket
    from datetime import datetime

    # System scanner
    sys_config = None
    if ssh_config:
        sys_config = SystemScannerConfig(
            ssh_host=ssh_config.get('host'),
            ssh_port=ssh_config.get('port', 22),
            ssh_user=ssh_config.get('user'),
            ssh_key=ssh_config.get('key'),
        )
    system_scanner = SystemScanner(config=sys_config)
    system_context = system_scanner.scan()

    # Schema scanner
    schema_scanner = SchemaScanner(conn)
    schema_context = schema_scanner.scan()

    # Runtime scanner
    runtime_scanner = RuntimeScanner(conn, system_scanner)
    runtime_context = runtime_scanner.scan()

    # Build context packet
    context_packet = ContextPacket(
        protocol_version="v2",
        timestamp=datetime.utcnow().isoformat(),
        session_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        system_context=system_context,
        runtime_context=runtime_context,
        schema_context=schema_context,
    )

    return context_packet


def get_db_config(conn) -> Dict[str, str]:
    """Get key PostgreSQL configuration values."""
    key_params = [
        'max_connections',
        'shared_buffers',
        'effective_cache_size',
        'work_mem',
        'maintenance_work_mem',
        'wal_buffers',
        'checkpoint_completion_target',
        'random_page_cost',
        'effective_io_concurrency',
        'max_worker_processes',
        'max_parallel_workers',
    ]

    config = {}
    try:
        with conn.cursor() as cur:
            for param in key_params:
                cur.execute(f"SHOW {param}")
                result = cur.fetchone()
                if result:
                    config[param] = result[0]
    except Exception:
        pass

    return config


def run_benchmark_with_telemetry(
    conn,
    strategy,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
    ssh_config: Optional[Dict],
    ui: 'DiagnoseUI',
    session_state: Optional['SessionState'] = None,
    round_num: int = 0,
) -> Dict[str, Any]:
    """Run benchmark with concurrent telemetry collection.

    Args:
        conn: Database connection
        strategy: Strategy with execution plan
        db_host, db_port, db_name, db_user, db_password: DB connection details
        ssh_config: SSH config for telemetry
        ui: UI instance
        session_state: Optional session state (for test mode)
        round_num: Current tuning round number (for mock TPS progression)
    """
    from .telemetry.collector import TelemetryCollector

    # Check if we should use mock benchmark
    use_mock_benchmark = (
        session_state and
        getattr(session_state, 'mock_benchmark', False)
    )

    if use_mock_benchmark:
        from .tests.mocks import MockBenchmarkRunner
        scenario = getattr(session_state, 'test_scenario', 'balanced_tps') or 'balanced_tps'
        runner = MockBenchmarkRunner(scenario=scenario)
        runner.set_round(round_num)
    else:
        from .runner.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
        )

    duration = strategy.execution_plan.duration_seconds if strategy.execution_plan else 60
    result = {'tps': 0, 'latency_avg_ms': 0, 'latency_p99_ms': 0, 'transactions': 0}
    telemetry_data = {'pg_stats': [], 'iostat': [], 'vmstat': []}

    # Start telemetry collection in background
    telemetry_stop = threading.Event()
    telemetry_thread = None

    def collect_telemetry():
        collector = TelemetryCollector(connection=conn, ssh_config=ssh_config)
        while not telemetry_stop.is_set():
            try:
                snapshot = collector.collect_snapshot()
                telemetry_data['pg_stats'].append(snapshot.get('pg_stats', {}))
                telemetry_data['iostat'].append(snapshot.get('iostat', {}))
                telemetry_data['vmstat'].append(snapshot.get('vmstat', {}))
            except Exception:
                pass
            time.sleep(1)

    try:
        # Start telemetry thread (skip in mock mode - no real telemetry needed)
        if not use_mock_benchmark:
            telemetry_thread = threading.Thread(target=collect_telemetry, daemon=True)
            telemetry_thread.start()

        # Show progress while benchmark runs
        if ui.console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=ui.console,
            ) as progress:
                task = progress.add_task("[cyan]Running benchmark...", total=duration)

                # Start benchmark in background
                bench_result = [None]
                bench_error = [None]

                def run_bench():
                    try:
                        bench_result[0] = runner.run(strategy)
                    except Exception as e:
                        bench_error[0] = e

                bench_thread = threading.Thread(target=run_bench)
                bench_thread.start()

                # Update progress
                start_time = time.time()
                while bench_thread.is_alive():
                    elapsed = time.time() - start_time
                    progress.update(task, completed=min(elapsed, duration))
                    time.sleep(0.5)

                bench_thread.join()
                progress.update(task, completed=duration)

                if bench_error[0]:
                    raise bench_error[0]
                result = bench_result[0]
        else:
            ui.print(f"Running benchmark ({duration}s)...")
            result = runner.run(strategy)

    finally:
        telemetry_stop.set()
        if telemetry_thread:
            telemetry_thread.join(timeout=2)

    return {
        'benchmark_result': result,
        'telemetry': telemetry_data,
    }


def format_telemetry_summary(telemetry: Dict) -> str:
    """
    Format telemetry data into a summary string for AI.

    Includes both aggregated stats AND time series data so AI can see patterns
    like performance cliffs, checkpoint spikes, etc.
    """
    lines = []

    # --- Aggregated Summary ---
    lines.append("=== AGGREGATED METRICS ===")

    pg_stats = telemetry.get('pg_stats', [])
    if pg_stats:
        avg_cache_hit = sum(s.get('cache_hit_ratio', 0) for s in pg_stats) / len(pg_stats) if pg_stats else 0
        min_cache_hit = min((s.get('cache_hit_ratio', 100) for s in pg_stats), default=0)
        lines.append(f"PG Cache Hit: avg={avg_cache_hit:.2f}%, min={min_cache_hit:.2f}%")

    iostat = telemetry.get('iostat', [])
    if iostat:
        avg_util = sum(s.get('util', 0) for s in iostat) / len(iostat) if iostat else 0
        max_util = max((s.get('util', 0) for s in iostat), default=0)
        avg_await = sum(s.get('await', 0) for s in iostat) / len(iostat) if iostat else 0
        max_await = max((s.get('await', 0) for s in iostat), default=0)
        lines.append(f"IO: avg_util={avg_util:.1f}%, max_util={max_util:.1f}%, avg_await={avg_await:.2f}ms, max_await={max_await:.2f}ms")

    vmstat = telemetry.get('vmstat', [])
    if vmstat:
        avg_cpu_user = sum(s.get('cpu_user', 0) for s in vmstat) / len(vmstat) if vmstat else 0
        avg_cpu_sys = sum(s.get('cpu_sys', 0) for s in vmstat) / len(vmstat) if vmstat else 0
        avg_cpu_wait = sum(s.get('cpu_wait', 0) for s in vmstat) / len(vmstat) if vmstat else 0
        max_cpu_wait = max((s.get('cpu_wait', 0) for s in vmstat), default=0)
        lines.append(f"CPU: avg_user={avg_cpu_user:.1f}%, avg_sys={avg_cpu_sys:.1f}%, avg_wait={avg_cpu_wait:.1f}%, max_wait={max_cpu_wait:.1f}%")

    # --- Time Series Data (for pattern detection) ---
    lines.append("\n=== TIME SERIES DATA (5-second intervals) ===")
    lines.append("Format: [T+0s, T+5s, T+10s, ...] - values at each interval")

    # IO utilization time series
    if iostat:
        util_series = [f"{s.get('util', 0):.0f}" for s in iostat]
        await_series = [f"{s.get('await', 0):.1f}" for s in iostat]
        read_series = [f"{s.get('r_per_sec', 0):.0f}" for s in iostat]
        write_series = [f"{s.get('w_per_sec', 0):.0f}" for s in iostat]
        lines.append(f"IO Utilization %: [{', '.join(util_series)}]")
        lines.append(f"IO Await ms: [{', '.join(await_series)}]")
        lines.append(f"Read IOPS: [{', '.join(read_series)}]")
        lines.append(f"Write IOPS: [{', '.join(write_series)}]")

    # CPU time series
    if vmstat:
        cpu_user_series = [f"{s.get('cpu_user', 0):.0f}" for s in vmstat]
        cpu_sys_series = [f"{s.get('cpu_sys', 0):.0f}" for s in vmstat]
        cpu_wait_series = [f"{s.get('cpu_wait', 0):.0f}" for s in vmstat]
        cpu_idle_series = [f"{s.get('cpu_idle', 0):.0f}" for s in vmstat]
        lines.append(f"CPU User %: [{', '.join(cpu_user_series)}]")
        lines.append(f"CPU System %: [{', '.join(cpu_sys_series)}]")
        lines.append(f"CPU Wait %: [{', '.join(cpu_wait_series)}]")
        lines.append(f"CPU Idle %: [{', '.join(cpu_idle_series)}]")

    # Memory/swap time series
    if vmstat:
        swap_in_series = [f"{s.get('swap_in', 0):.0f}" for s in vmstat]
        swap_out_series = [f"{s.get('swap_out', 0):.0f}" for s in vmstat]
        if any(int(x) > 0 for x in swap_in_series) or any(int(x) > 0 for x in swap_out_series):
            lines.append(f"Swap In: [{', '.join(swap_in_series)}]")
            lines.append(f"Swap Out: [{', '.join(swap_out_series)}]")

    # PG stats time series (cache hit ratio over time)
    if pg_stats:
        cache_hit_series = [f"{s.get('cache_hit_ratio', 0):.1f}" for s in pg_stats]
        lines.append(f"PG Cache Hit %: [{', '.join(cache_hit_series)}]")

    # Detect and highlight anomalies
    lines.append("\n=== ANOMALY DETECTION ===")
    anomalies = []

    # Detect IO spikes
    if iostat:
        utils = [s.get('util', 0) for s in iostat]
        avg_util = sum(utils) / len(utils)
        for i, u in enumerate(utils):
            if u > avg_util * 1.5 and u > 80:
                anomalies.append(f"IO spike at T+{i*5}s: {u:.0f}% (avg: {avg_util:.0f}%)")

        awaits = [s.get('await', 0) for s in iostat]
        avg_await = sum(awaits) / len(awaits)
        for i, a in enumerate(awaits):
            if a > avg_await * 2 and a > 10:
                anomalies.append(f"IO latency spike at T+{i*5}s: {a:.1f}ms (avg: {avg_await:.1f}ms)")

    # Detect CPU wait spikes (indicates IO bottleneck)
    if vmstat:
        waits = [s.get('cpu_wait', 0) for s in vmstat]
        for i, w in enumerate(waits):
            if w > 20:
                anomalies.append(f"High CPU wait at T+{i*5}s: {w:.0f}%")

    # Detect cache hit drops
    if pg_stats:
        hits = [s.get('cache_hit_ratio', 100) for s in pg_stats]
        for i, h in enumerate(hits):
            if h < 95:
                anomalies.append(f"Low cache hit at T+{i*5}s: {h:.1f}%")

    if anomalies:
        for anomaly in anomalies[:10]:  # Limit to 10 anomalies
            lines.append(f"  ⚠ {anomaly}")
    else:
        lines.append("  No significant anomalies detected")

    return "\n".join(lines) if lines else "No telemetry data collected"


def apply_round1_config(
    conn,
    config,
    ssh_config: Optional[Dict],
    ui: 'DiagnoseUI',
) -> Dict[str, Any]:
    """
    Apply Round 1 configuration changes.

    Returns dict with:
      - success: True if all changes applied successfully
      - applied_changes: List of applied changes with commands
    """
    applied_pg = []
    applied_os = []
    applied_changes = []  # Track for summary
    restart_needed = False

    # Apply PostgreSQL tuning chunks
    if config.tuning_chunks:
        ui.print()
        ui.spinner_start("Applying PostgreSQL configuration...")

        # ALTER SYSTEM requires autocommit mode
        # First, commit any pending transaction
        conn.rollback()
        old_autocommit = conn.autocommit
        conn.autocommit = True

        try:
            for chunk in config.tuning_chunks:
                try:
                    with conn.cursor() as cur:
                        for cmd in chunk.apply_commands:
                            # Skip pg_reload_conf if we need a restart anyway
                            if 'pg_reload_conf' in cmd.lower() and config.restart_required:
                                continue
                            cur.execute(cmd)

                    applied_pg.append(chunk.name)

                    # Track for summary - extract ALTER SYSTEM commands
                    pg_configs = [cmd for cmd in chunk.apply_commands
                                 if 'ALTER SYSTEM SET' in cmd.upper()]
                    applied_changes.append({
                        'round': 0,  # Round 0 = initial config
                        'name': chunk.name,
                        'category': chunk.category,
                        'pg_configs': pg_configs,
                        'type': 'postgresql',
                    })

                    if chunk.requires_restart:
                        restart_needed = True

                except Exception as e:
                    ui.spinner_stop(f"Failed on {chunk.name}")
                    ui.print_error(f"Failed to apply {chunk.name}: {e}")
                    conn.autocommit = old_autocommit
                    return {'success': False, 'applied_changes': applied_changes}

            ui.spinner_stop(f"Applied {len(applied_pg)} PostgreSQL changes")
        finally:
            conn.autocommit = old_autocommit

    # Apply OS tuning via SSH (if available)
    if config.os_tuning and ssh_config:
        ui.print()
        ui.spinner_start("Applying OS configuration...")

        for ot in config.os_tuning:
            try:
                ssh_cmd = [
                    "ssh",
                    "-t", "-t",  # Force pseudo-TTY for sudo
                    "-o", "StrictHostKeyChecking=no",
                    "-p", str(ssh_config.get('port', 22)),
                    f"{ssh_config.get('user', 'ubuntu')}@{ssh_config.get('host')}",
                    ot.apply_command
                ]
                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    stdin=subprocess.DEVNULL,
                )

                if result.returncode == 0:
                    applied_os.append(ot.name)
                    # Track OS changes for summary
                    applied_changes.append({
                        'round': 0,
                        'name': ot.name,
                        'category': 'os',
                        'os_command': ot.apply_command,
                        'type': 'os',
                    })
                else:
                    # Show warning but don't fail - OS tuning is optional
                    ui.print(f"[yellow]Warning: {ot.name}: {result.stderr.strip()}[/]" if ui.console else f"Warning: {ot.name}: {result.stderr.strip()}")

            except Exception as e:
                ui.print(f"[yellow]Warning: {ot.name}: {e}[/]" if ui.console else f"Warning: {ot.name}: {e}")

        if applied_os:
            ui.spinner_stop(f"Applied {len(applied_os)} OS changes")
        else:
            ui.spinner_stop("OS changes skipped")

    elif config.os_tuning and not ssh_config:
        ui.print()
        ui.print("[yellow]OS tuning skipped (no SSH config provided)[/]" if ui.console else "OS tuning skipped (no SSH config)")

    # Handle restart if needed
    if restart_needed or config.restart_required:
        ui.print()
        if ui.console:
            ui.console.print("[bold yellow]PostgreSQL restart required to apply changes.[/]")
        else:
            print("PostgreSQL restart required to apply changes.")

        restart_confirm = ui.prompt("Restart PostgreSQL now? [y/N]: ").lower()
        if restart_confirm == 'y':
            if ssh_config:
                if not restart_postgresql_with_retry(ssh_config, ui, max_retries=5, timeout_sec=5):
                    return {'success': False, 'applied_changes': applied_changes}
            else:
                ui.print("Please restart PostgreSQL manually and press Enter to continue...")
                ui.prompt("", allow_commands=False)
        else:
            ui.print("[yellow]Configuration applied but restart skipped. Some changes won't take effect.[/]" if ui.console else "Restart skipped.")

    # Verify changes
    ui.print()
    ui.spinner_start("Verifying configuration...")

    verified = 0
    for chunk in config.tuning_chunks:
        if chunk.verification_command:
            try:
                with conn.cursor() as cur:
                    cur.execute(chunk.verification_command)
                    result = cur.fetchone()
                    if result and str(result[0]) == chunk.verification_expected:
                        verified += 1
            except Exception:
                pass

    ui.spinner_stop(f"Verified {verified}/{len(config.tuning_chunks)} changes")

    return {'success': True, 'applied_changes': applied_changes}


def restart_postgresql_with_retry(ssh_config: Dict, ui: 'DiagnoseUI', max_retries: int = 5, timeout_sec: int = 5) -> bool:
    """
    Restart PostgreSQL and wait for it to be ready.
    Also restarts PgBouncer if it's running.

    Args:
        ssh_config: SSH configuration dict
        ui: UI instance
        max_retries: Maximum number of retries (default: 5)
        timeout_sec: Timeout for each retry in seconds (default: 5)

    Returns:
        True if restart successful and PostgreSQL is ready, False otherwise
    """
    ui.spinner_start("Restarting PostgreSQL...")

    try:
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-p", str(ssh_config.get('port', 22)),
            f"{ssh_config.get('user', 'ubuntu')}@{ssh_config.get('host')}",
            "sudo systemctl restart postgresql"
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            ui.spinner_stop("Restart command failed")
            ui.print_error(f"Restart failed: {result.stderr}")
            return False

        ui.spinner_stop("Restart initiated, waiting for PostgreSQL to be ready...")

        # Wait for PostgreSQL to be ready with retry
        pg_ready = False
        for attempt in range(1, max_retries + 1):
            ui.print(f"  Checking readiness (attempt {attempt}/{max_retries})..." if not ui.console else f"  [dim]Checking readiness (attempt {attempt}/{max_retries})...[/dim]")

            try:
                ready_cmd = [
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    "-o", f"ConnectTimeout={timeout_sec}",
                    "-p", str(ssh_config.get('port', 22)),
                    f"{ssh_config.get('user', 'ubuntu')}@{ssh_config.get('host')}",
                    "pg_isready -q"
                ]
                ready_result = subprocess.run(ready_cmd, capture_output=True, timeout=timeout_sec + 5)

                if ready_result.returncode == 0:
                    ui.print("[green]PostgreSQL is ready[/]" if ui.console else "PostgreSQL is ready")
                    pg_ready = True
                    break

            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

            if attempt < max_retries:
                time.sleep(timeout_sec)

        if not pg_ready:
            ui.print("[red]PostgreSQL failed to become ready after all retries[/]" if ui.console else "PostgreSQL failed to become ready")
            return False

        # Restart PgBouncer if it exists (to re-establish connection pool)
        ui.print("[dim]Checking for PgBouncer...[/dim]" if ui.console else "Checking for PgBouncer...")
        try:
            pgb_check_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-p", str(ssh_config.get('port', 22)),
                f"{ssh_config.get('user', 'ubuntu')}@{ssh_config.get('host')}",
                "systemctl is-active pgbouncer 2>/dev/null || echo 'not-found'"
            ]
            pgb_result = subprocess.run(pgb_check_cmd, capture_output=True, text=True, timeout=10)

            if pgb_result.returncode == 0 and 'active' in pgb_result.stdout:
                ui.print("[dim]Restarting PgBouncer...[/dim]" if ui.console else "Restarting PgBouncer...")
                pgb_restart_cmd = [
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    "-p", str(ssh_config.get('port', 22)),
                    f"{ssh_config.get('user', 'ubuntu')}@{ssh_config.get('host')}",
                    "sudo systemctl restart pgbouncer"
                ]
                subprocess.run(pgb_restart_cmd, capture_output=True, timeout=30)
                time.sleep(2)  # Give PgBouncer time to establish connections
                ui.print("[green]PgBouncer restarted[/]" if ui.console else "PgBouncer restarted")
        except Exception:
            pass  # PgBouncer restart is optional

        return True

    except Exception as e:
        ui.spinner_stop("Restart failed")
        ui.print_error(f"Restart failed: {e}")
        return False


def reconnect_with_retry(
    host: str, port: int, user: str, password: str, database: str,
    ui: 'DiagnoseUI', max_retries: int = 5, timeout_sec: int = 5
):
    """
    Reconnect to PostgreSQL with retry logic.

    Args:
        max_retries: Maximum number of retries (default: 5)
        timeout_sec: Timeout between retries in seconds (default: 5)

    Returns:
        Connection object if successful

    Raises:
        Exception if all retries fail
    """
    ui.spinner_start("Reconnecting to database...")

    for attempt in range(1, max_retries + 1):
        try:
            conn = create_connection(host, port, user, password, database)
            ui.spinner_stop("Reconnected")
            return conn
        except Exception as e:
            if attempt < max_retries:
                ui.print(f"  [dim]Retry {attempt}/{max_retries} in {timeout_sec}s...[/dim]" if ui.console else f"  Retry {attempt}/{max_retries}...")
                time.sleep(timeout_sec)
            else:
                ui.spinner_stop("Reconnection failed")
                raise Exception(f"Could not reconnect after {max_retries} attempts: {e}")


def display_final_summary(
    ui: 'DiagnoseUI',
    tuning_history: Dict[str, Any],
    target_tps: float,
    target_hit: bool,
):
    """Display final tuning summary with sweet spot and hardware suggestions."""
    ui.print()
    if RICH_AVAILABLE and ui.console:
        ui.console.rule("[bold green]Tuning Session Complete[/]")
    else:
        ui.print("=" * 50)
        ui.print("Tuning Session Complete")
        ui.print("=" * 50)

    # Summary table
    ui.print()
    baseline = tuning_history['baseline_tps']
    best = tuning_history['best_tps']
    improvement = ((best - baseline) / baseline * 100) if baseline > 0 else 0

    if ui.console:
        summary_table = Table(title="Performance Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Baseline TPS", f"{baseline:.0f}")
        summary_table.add_row("Best TPS", f"{best:.0f}")
        summary_table.add_row("Improvement", f"{improvement:.1f}%")
        summary_table.add_row("Rounds Completed", str(tuning_history['iterations_completed']))
        summary_table.add_row("Target TPS", str(int(target_tps)) if target_tps > 0 else "N/A")

        if target_tps > 0:
            if target_hit:
                summary_table.add_row("Target Status", "[bold green]✓ ACHIEVED[/]")
            else:
                gap = ((target_tps - best) / target_tps * 100)
                summary_table.add_row("Target Status", f"[yellow]✗ MISSED by {gap:.1f}%[/]")

        ui.console.print(summary_table)

        # Sweet Spot Suggestion - Show actual config commands
        if tuning_history['applied_changes']:
            ui.print()
            ui.console.print("[bold cyan]Sweet Spot Configuration:[/]")
            ui.console.print("  The following changes contributed to the best performance:")

            # Separate PostgreSQL and OS changes
            pg_changes = []
            os_changes = []
            for change in tuning_history['applied_changes']:
                if change.get('type') == 'os' or change.get('category') == 'os':
                    os_changes.append(change)
                else:
                    pg_changes.append(change)

            # Display PostgreSQL configuration changes
            if pg_changes:
                ui.console.print("\n  [yellow]PostgreSQL Configuration:[/]")
                for change in pg_changes:
                    round_label = f"[dim](round {change.get('round', '?')})[/dim]"
                    ui.console.print(f"    [green]• {change['name']}[/] {round_label}")
                    # Show actual ALTER SYSTEM commands
                    pg_configs = change.get('pg_configs', [])
                    if pg_configs:
                        for cmd in pg_configs:
                            ui.console.print(f"      [cyan]→ {cmd}[/]")
                    # Also show apply_commands if no pg_configs (for tuning rounds)
                    elif 'apply_commands' in change:
                        for cmd in change['apply_commands']:
                            if 'ALTER SYSTEM SET' in cmd.upper():
                                ui.console.print(f"      [cyan]→ {cmd}[/]")

            # Display OS configuration changes
            if os_changes:
                ui.console.print("\n  [yellow]OS Configuration:[/]")
                for change in os_changes:
                    round_label = f"[dim](round {change.get('round', '?')})[/dim]"
                    ui.console.print(f"    [green]• {change['name']}[/] {round_label}")
                    # Show actual OS command
                    os_cmd = change.get('os_command', '')
                    if os_cmd:
                        ui.console.print(f"      [cyan]→ {os_cmd}[/]")

            ui.console.print(f"\n  [dim]These settings achieved {best:.0f} TPS ({improvement:.1f}% improvement)[/dim]")

        # Hardware suggestions if target not hit
        if not target_hit and target_tps > 0:
            ui.print()
            ui.console.print("[bold yellow]Hardware Scaling Recommendations:[/]")

            gap_pct = ((target_tps - best) / best * 100) if best > 0 else 100

            if gap_pct > 50:
                ui.console.print("  [red]Large performance gap detected. Significant hardware upgrade needed:[/]")
                ui.console.print("  • Consider migrating to a larger instance class")
                ui.console.print("  • Add more CPU cores (current workload appears CPU-bound)")
                ui.console.print("  • Increase RAM significantly for larger shared_buffers")
                ui.console.print("  • Switch to NVMe RAID for I/O-intensive workloads")
            elif gap_pct > 20:
                ui.console.print("  [yellow]Moderate hardware improvements recommended:[/]")
                ui.console.print("  • Add 2-4 more CPU cores for better parallelism")
                ui.console.print("  • Increase RAM by 50-100% for better caching")
                ui.console.print("  • Consider faster storage (NVMe over SSD)")
            else:
                ui.console.print("  [green]Minor improvements may help:[/]")
                ui.console.print("  • Fine-tune connection pooling (PgBouncer)")
                ui.console.print("  • Consider read replicas for read-heavy workloads")
                ui.console.print("  • Review application query patterns")

            ui.console.print(f"\n  [dim]Gap to target: {gap_pct:.1f}% ({int(target_tps - best)} TPS)[/dim]")
    else:
        ui.print(f"\nBaseline TPS: {baseline:.0f}")
        ui.print(f"Best TPS: {best:.0f}")
        ui.print(f"Improvement: {improvement:.1f}%")
        ui.print(f"Rounds: {tuning_history['iterations_completed']}")

        if target_hit:
            ui.print("Target: ACHIEVED")
        elif target_tps > 0:
            ui.print(f"Target: MISSED (needed {target_tps:.0f})")

        # Show applied config changes in plain text
        if tuning_history['applied_changes']:
            ui.print("\nApplied Configuration Changes:")

            # Separate PostgreSQL and OS changes
            pg_changes = [c for c in tuning_history['applied_changes']
                         if c.get('type') != 'os' and c.get('category') != 'os']
            os_changes = [c for c in tuning_history['applied_changes']
                         if c.get('type') == 'os' or c.get('category') == 'os']

            if pg_changes:
                ui.print("  PostgreSQL:")
                for change in pg_changes:
                    ui.print(f"    • {change['name']} (round {change.get('round', '?')})")
                    for cmd in change.get('pg_configs', []):
                        ui.print(f"      → {cmd}")

            if os_changes:
                ui.print("  OS:")
                for change in os_changes:
                    ui.print(f"    • {change['name']} (round {change.get('round', '?')})")
                    if change.get('os_command'):
                        ui.print(f"      → {change['os_command']}")

        if not target_hit and target_tps > 0:
            ui.print("\nConsider hardware upgrades:")
            ui.print("  - More CPU cores")
            ui.print("  - More RAM")
            ui.print("  - Faster storage (NVMe)")


# ==================== Special Mode Functions ====================

def run_health_check(args, password: str, ssh_config: Optional[Dict], ui: 'DiagnoseUI'):
    """Run health check mode."""
    from .modes import HealthCheck

    # Connect to database
    database = args.database or 'postgres'
    try:
        conn = create_connection(args.host, args.port, args.user, password, database)
    except Exception as e:
        ui.print_error(f"Connection failed: {e}")
        sys.exit(1)

    try:
        health = HealthCheck(conn, ssh_config)
        results = health.run_all()
        health.display(results)
    finally:
        conn.close()


def run_watch_mode(args, password: str, ssh_config: Optional[Dict], ui: 'DiagnoseUI', interval: int = 5):
    """Run watch mode."""
    from .modes import WatchMode

    # Connect to database
    database = args.database or 'postgres'
    try:
        conn = create_connection(args.host, args.port, args.user, password, database)
    except Exception as e:
        ui.print_error(f"Connection failed: {e}")
        sys.exit(1)

    try:
        target_tps = args.target_tps if hasattr(args, 'target_tps') else 0
        watch = WatchMode(conn, ssh_config, interval)
        watch.run(target_tps)
    finally:
        conn.close()


def run_dry_run_mode(args, password: str, ssh_config: Optional[Dict], ui: 'DiagnoseUI', use_mock: bool):
    """Run dry-run / analyze-only mode."""
    from .modes import DryRunMode

    # Connect to database
    database = args.database or 'postgres'
    try:
        conn = create_connection(args.host, args.port, args.user, password, database)
    except Exception as e:
        ui.print_error(f"Connection failed: {e}")
        sys.exit(1)

    try:
        # Initialize agent
        if use_mock:
            from .agent.client import MockGeminiAgent
            agent = MockGeminiAgent()
        else:
            from .agent.client import GeminiAgent
            agent = GeminiAgent()

        dry_run = DryRunMode(conn, agent, ssh_config)
        recommendations = dry_run.run(args.output)
        dry_run.display(recommendations)

        # If output file specified, confirm
        if args.output:
            ui.print(f"\n[green]Recommendations saved to {args.output}[/]" if ui.console else f"\nSaved to {args.output}")
    finally:
        conn.close()


def run_export_mode(session_data: Dict, args, ui: 'DiagnoseUI'):
    """Handle export operations."""
    from .export import export_session

    # Determine format
    if args.export_config:
        format_type = 'postgresql'
    elif args.export_ansible:
        format_type = 'ansible'
    elif args.export:
        format_type = args.export
    else:
        return  # No export requested

    try:
        content = export_session(session_data, format_type, args.output)

        if args.output:
            ui.print(f"[green]Exported to {args.output}[/]" if ui.console else f"Exported to {args.output}")
        else:
            print(content)
    except Exception as e:
        ui.print_error(f"Export failed: {e}")


def main():
    """Main entry point."""
    args = parse_args()
    password = args.password or os.environ.get('PGPASSWORD')

    # ==================== Test Mode Setup ====================
    test_handler = None
    if args.test_mode:
        from .tests.test_mode import setup_test_mode, TestModeHandler
        test_handler = setup_test_mode(args)

        # In test mode, also enable mocks by default
        if args.mock_ai or args.mock_benchmark:
            pass  # Will be handled below

    # Initialize UI
    ui = DiagnoseUI()

    # Connect test handler to UI if in test mode
    if test_handler:
        ui.test_handler = test_handler

    # ==================== Handle Session List (no connection needed) ====================
    if args.list_sessions:
        from .session import SessionManager, display_sessions
        manager = SessionManager()
        sessions = manager.list_sessions()
        display_sessions(sessions, ui.console)
        return

    # ==================== Validate Host for Connection Modes ====================
    if not args.host:
        ui.print_error("Host (-H) is required")
        sys.exit(1)

    # Initialize session state for slash commands
    session_state = SessionState(
        db_host=args.host,
        db_port=args.port,
        db_user=args.user,
        db_password=password or "",
    )

    # Create command handler and link to UI
    cmd_handler = create_command_handler(session_state)
    ui.set_command_handler(cmd_handler)
    session_state.ui = ui

    # Check for Gemini API key (use mock mode via env var for development)
    # Not needed for health/watch modes, or when using mock AI
    use_mock = os.environ.get('PG_DIAGNOSE_MOCK', '').lower() in ('1', 'true', 'yes')
    use_mock = use_mock or (args.test_mode and args.mock_ai)
    needs_ai = not (args.health or args.watch)
    if needs_ai and not use_mock and not os.environ.get('GEMINI_API_KEY'):
        ui.print_error("GEMINI_API_KEY environment variable is required")
        ui.print("Set it with: export GEMINI_API_KEY=your-api-key")
        sys.exit(1)

    # Store test mode config in session state
    session_state.test_mode = args.test_mode
    session_state.mock_ai = args.mock_ai if args.test_mode else False
    session_state.mock_benchmark = args.mock_benchmark if args.test_mode else False
    session_state.test_scenario = args.test_scenario if args.test_mode else None
    session_state.test_handler = test_handler

    # SSH config - parse user@host format
    ssh_config = None
    ssh_user = "ubuntu"  # default
    ssh_host = None
    if args.ssh_host:
        if '@' in args.ssh_host:
            ssh_user, ssh_host = args.ssh_host.split('@', 1)
        else:
            ssh_host = args.ssh_host
        ssh_config = {
            'host': ssh_host,
            'port': args.ssh_port,
            'user': ssh_user,
        }
        session_state.ssh_config = ssh_config

    # ==================== Handle Special Modes ====================

    # Health Check Mode
    if args.health:
        ui.print_banner()
        run_health_check(args, password, ssh_config, ui)
        return

    # Watch Mode
    if args.watch:
        ui.print_banner()
        run_watch_mode(args, password, ssh_config, ui, args.watch_interval)
        return

    # Dry-run / Analyze-only Mode
    if args.dry_run or args.analyze_only:
        ui.print_banner()
        run_dry_run_mode(args, password, ssh_config, ui, use_mock)
        return

    # ==================== Normal Interactive Mode ====================

    # Print banner
    ui.print_banner()
    ui.print("[dim]Type /help for available commands[/]" if RICH_AVAILABLE else "Type /help for available commands")
    ui.print()

    # ===================== WORKSPACE CHECK (IDE-like) =====================
    # Check for existing workspaces BEFORE database selection
    workspace_manager = WorkspaceManager()
    ui.set_workspace_manager(workspace_manager)
    existing_workspaces = workspace_manager.list_workspaces()

    # Cleanup stale sessions (ACTIVE sessions with no activity for 24+ hours)
    # This runs at startup to mark abandoned sessions
    for ws_info in existing_workspaces:
        try:
            ws = workspace_manager.open(ws_info['name'])
            if ws:
                abandoned = workspace_manager.cleanup_stale_sessions(max_age_hours=24)
                if abandoned:
                    ui.print(f"[dim]Marked {len(abandoned)} stale session(s) as abandoned in {ws_info['name']}[/]" if RICH_AVAILABLE else f"Marked {len(abandoned)} stale session(s) as abandoned")
                workspace_manager.close()
        except Exception:
            pass  # Ignore cleanup errors - non-critical

    database = None
    workspace = None
    resumed_session = None

    # If no -d flag and workspaces exist, show them first
    if not args.database and existing_workspaces:
        workspace_selection_done = False

        while not workspace_selection_done:
            ui.print()
            if RICH_AVAILABLE and ui.console:
                ui.console.rule("[bold cyan]Recent Workspaces[/]")
            else:
                ui.print("=" * 50)
                ui.print("Recent Workspaces")
                ui.print("=" * 50)

            # Display workspaces
            from .workspace import display_workspaces
            display_workspaces(existing_workspaces, ui.console)

            ui.print()
            ui.print("[dim]Enter workspace number to resume, or 'n' for new database[/]" if RICH_AVAILABLE else "Enter workspace number to resume, or 'n' for new database")
            ui.show_context_menu('workspace')

            while True:
                ws_choice = ui.prompt("Choice: ").strip().lower()

                # Handle workspace-level slash commands
                if ws_choice in ['/quit', '/q', 'quit', 'q']:
                    ui.print("Goodbye!")
                    return
                if ws_choice in ['/status', '/s']:
                    ui.print(f"Workspaces: {len(existing_workspaces)}")
                    continue
                if ws_choice in ['/new', '/n']:
                    ws_choice = 'n'
                break  # Valid choice, exit the inner loop

            if ws_choice == 'n' or ws_choice == 'new':
                workspace_selection_done = True
                continue

            try:
                ws_idx = int(ws_choice) - 1
                if 0 <= ws_idx < len(existing_workspaces):
                    ws_info = existing_workspaces[ws_idx]
                    workspace = workspace_manager.open(ws_info['name'])
                    if workspace:
                        database = workspace.data.db_name
                        ui.print(f"\n[green]Resuming workspace: {workspace.name}[/]" if RICH_AVAILABLE else f"\nResuming workspace: {workspace.name}")

                        # Check for paused sessions
                        sessions = workspace.list_sessions()
                        paused = [s for s in sessions if s.get('state') == 'paused']
                        active = [s for s in sessions if s.get('state') == 'active']

                        if paused or active:
                            resumable_sessions = active + paused
                            ui.print()
                            ui.print("[cyan]Sessions in this workspace:[/]" if RICH_AVAILABLE else "Sessions in this workspace:")
                            for idx, s in enumerate(resumable_sessions, 1):
                                state_str = "[yellow]ACTIVE[/]" if s['state'] == 'active' else "[blue]PAUSED[/]"
                                if RICH_AVAILABLE:
                                    ui.print(f"  {idx}. {s['strategy'][:35]} {state_str} R{s.get('rounds', 0)} {s.get('best_tps', 0):.0f} TPS")
                                else:
                                    ui.print(f"  {idx}. {s['strategy'][:35]} [{s['state'].upper()}] R{s.get('rounds', 0)}")

                            ui.print()
                            ui.show_context_menu('session_list')
                            resume_choice = ui.prompt("Enter session number to resume, or 'n' for new: ").strip().lower()

                            # Handle session-level slash commands
                            if resume_choice in ['/back', '/b']:
                                workspace_manager.close()
                                workspace = None
                                database = None
                                continue  # Back to workspace selection
                            if resume_choice in ['/quit', '/q']:
                                ui.print("Goodbye!")
                                return

                            if resume_choice != 'n' and resume_choice:
                                try:
                                    session_idx = int(resume_choice) - 1
                                    if 0 <= session_idx < len(resumable_sessions):
                                        session_name = resumable_sessions[session_idx]['name']
                                        workspace.open_session(session_name)
                                        if workspace.current_session:
                                            if workspace.current_session.state.value == 'paused':
                                                workspace.current_session.resume()
                                            resumed_session = workspace.current_session
                                            ui.print(f"[green]Resumed session: {resumable_sessions[session_idx]['strategy']}[/]" if RICH_AVAILABLE else f"Resumed: {session_name}")
                                    else:
                                        ui.print("[yellow]Invalid session number[/]" if RICH_AVAILABLE else "Invalid session number")
                                except ValueError:
                                    ui.print("[yellow]Invalid input, starting new session[/]" if RICH_AVAILABLE else "Invalid input, starting new session")
                                except Exception as e:
                                    ui.print(f"[yellow]Could not resume: {e}[/]" if RICH_AVAILABLE else f"Could not resume: {e}")

                        workspace_selection_done = True
                else:
                    ui.print("[yellow]Invalid workspace number[/]" if RICH_AVAILABLE else "Invalid workspace number")
            except ValueError:
                ui.print("[yellow]Invalid input[/]" if RICH_AVAILABLE else "Invalid input")

    # ===================== PHASE 1: Discovery =====================
    # Step 1: Connect to PostgreSQL server
    ui.spinner_start("Connecting to PostgreSQL server...")
    try:
        databases = list_databases(args.host, args.port, args.user, password)
        ui.spinner_stop("Connected")
    except Exception as e:
        ui.spinner_stop("Failed")
        ui.print_error(str(e))
        sys.exit(1)

    # Step 2: Select database (if not already set from workspace resume)
    if args.database:
        database = args.database
        ui.print(f"Using database: [cyan]{database}[/]" if RICH_AVAILABLE else f"Using database: {database}")
    elif not database:
        # No workspace resumed, show database list
        database = ui.select_database(databases)

    # Step 3: Connect to selected database
    ui.print()
    ui.spinner_start(f"Loading {database} configuration...")
    conn = create_connection(args.host, args.port, args.user, password, database)
    db_config = get_db_config(conn)
    ui.spinner_stop("Loaded")

    # Update session state with connection info
    session_state.db_name = database
    session_state.conn = conn
    session_state.baseline_config = db_config.copy()

    # ===================== WORKSPACE INITIALIZATION =====================
    # Create or find workspace for this database (if not already resumed)
    if not workspace:
        workspace = workspace_manager.find_or_create(
            db_host=args.host,
            db_port=args.port,
            db_name=database,
            db_user=args.user,
            ssh_host=ssh_host or "",
            ssh_user=ssh_user,
        )

    ui.print()
    if RICH_AVAILABLE:
        ui.print(f"[dim]Workspace: {workspace.name}[/]")
    else:
        ui.print(f"Workspace: {workspace.name}")

    # Show DB configuration
    ui.print()
    ui.display_db_config(db_config, f"PostgreSQL Configuration ({database})")

    # Initialize ws_session before try block to ensure it's always available
    ws_session = None

    try:
        # Initialize AI agent
        # Priority: test mode mock > general mock > real agent
        if session_state.mock_ai:
            # Use test mode's MockGeminiAgent with golden_data
            from .tests.mocks import MockGeminiAgent as TestMockAgent
            scenario = session_state.test_scenario or 'balanced_tps'
            agent = TestMockAgent(scenario=scenario)
        elif use_mock:
            # Use basic mock from agent.client
            from .agent.client import MockGeminiAgent
            agent = MockGeminiAgent()
        else:
            from .agent.client import GeminiAgent
            agent = GeminiAgent()

        # Store agent reference for slash commands
        session_state.agent = agent

        # ===================== HANDLE RESUMED SESSION =====================
        # Resume flow control flags
        skip_to_baseline = False
        skip_to_tuning_loop = False
        resume_tuning_history = None
        resume_round_num = None
        resume_strategy = None
        resume_chosen_strategy_dict = None
        resume_target_tps = None
        resume_true_baseline_tps = None

        if resumed_session:
            ui.print()
            if RICH_AVAILABLE:
                ui.console.rule("[bold green]Resuming Session[/]")
            else:
                ui.print("=" * 50)
                ui.print("Resuming Session")
                ui.print("=" * 50)

            # Show session summary
            rs_data = resumed_session.data
            ui.print(f"Strategy: [cyan]{rs_data.strategy_name}[/]" if RICH_AVAILABLE else f"Strategy: {rs_data.strategy_name}")
            ui.print(f"Rounds completed: {len(rs_data.rounds)}")

            if rs_data.baseline:
                ui.print(f"Baseline TPS: {rs_data.baseline.get('tps', 0):.0f}")
            if rs_data.best_tps > 0:
                ui.print(f"Best TPS: {rs_data.best_tps:.0f}")
            if rs_data.target_tps > 0:
                ui.print(f"Target TPS: {rs_data.target_tps:.0f}")

            ui.print()
            ui.print("[dim]Running discovery to refresh system context...[/]" if RICH_AVAILABLE else "Refreshing system context...")

            # Run discovery for fresh context
            ui.spinner_start("Analyzing system...")
            context_packet = run_discovery(conn, ssh_config)
            ui.spinner_stop("Ready")

            # Use resumed session as ws_session
            ws_session = resumed_session

            # Restore session state variables
            resume_true_baseline_tps = rs_data.baseline.get('tps', 0) if rs_data.baseline else 0
            resume_target_tps = rs_data.target_tps
            session_state.baseline_tps = resume_true_baseline_tps
            session_state.target_tps = resume_target_tps
            session_state.strategy_name = rs_data.strategy_name

            # Build strategy object for AI calls
            from .protocol.sdl import StrategyDefinition, ExecutionPlan, SuccessCriteria
            resume_strategy = StrategyDefinition(
                id=rs_data.strategy_id or "resumed",
                name=rs_data.strategy_name,
                hypothesis=rs_data.strategy_rationale or "",
                execution_plan=ExecutionPlan(
                    benchmark_type="pgbench",
                    scale=100,
                    clients=50,
                    duration_seconds=120,
                ),
                success_criteria=SuccessCriteria(
                    target_tps=resume_target_tps,
                ),
            )

            resume_chosen_strategy_dict = {
                'id': rs_data.strategy_id or "resumed",
                'name': rs_data.strategy_name,
                'goal': rs_data.strategy_rationale or "",
                'hypothesis': rs_data.strategy_rationale or "",
                'target_kpis': {'target_tps': resume_target_tps},
            }

            # Determine where to resume based on session state
            if not rs_data.baseline:
                # No baseline yet - skip to baseline
                ui.print()
                ui.print("[cyan]Session saved at strategy selection. Resuming to baseline...[/]" if RICH_AVAILABLE else "Resuming to baseline...")
                skip_to_baseline = True
            elif len(rs_data.rounds) == 0:
                # Has baseline but no rounds - skip to Round 1
                ui.print()
                ui.print("[cyan]Session saved after baseline. Resuming to Round 1...[/]" if RICH_AVAILABLE else "Resuming to Round 1...")
                skip_to_baseline = True  # Will fall through to Round 1 after baseline check
            else:
                # Has rounds completed - skip to tuning loop
                tps_history = [resume_true_baseline_tps] + [r.get('tps', 0) for r in rs_data.rounds]
                best_tps = rs_data.best_tps or max(tps_history) if tps_history else 0

                resume_tuning_history = {
                    'baseline_tps': resume_true_baseline_tps,
                    'best_tps': best_tps,
                    'tps_history': tps_history,
                    'iterations_completed': len(rs_data.rounds),
                    'applied_changes': rs_data.sweet_spot_changes.copy(),
                    'no_improvement_rounds': 0,
                }
                resume_round_num = len(rs_data.rounds) + 1

                ui.print()
                ui.print(f"[cyan]Resuming from Round {resume_round_num}...[/]" if RICH_AVAILABLE else f"Resuming from Round {resume_round_num}...")
                skip_to_tuning_loop = True

            ui.show_status_line()

            # Confirm resume
            ui.print()
            continue_choice = ui.prompt("Continue? [Y/n]: ").strip().lower()
            if continue_choice in ['n', 'no']:
                ui.print("Session remains paused.")
                return

        # Step 3: Discovery (skip if resuming - already done above)
        if not resumed_session:
            ui.print()
            ui.spinner_start("Analyzing system and schema...")
            context_packet = run_discovery(conn, ssh_config)
            ui.spinner_stop("Analysis complete")

        # Step 4: First Sight Analysis (skip if resuming with baseline)
        first_sight = None
        if not skip_to_baseline and not skip_to_tuning_loop:
            ui.print()
            ui.spinner_start("Consulting AI for recommendations...")
            first_sight = agent.first_sight_analysis(context_packet)
            ui.spinner_stop("AI analysis complete")

            # Display results
            ui.display_first_sight(first_sight)

        # ===================== MAIN LOOP: Strategy -> Tuning -> Benchmark =====================
        # This loop allows returning to strategy selection after a tuning session

        while True:
            # ===== SKIP STRATEGY SELECTION IF RESUMING =====
            if skip_to_baseline or skip_to_tuning_loop:
                # Use saved strategy from resume
                strategy = resume_strategy
                chosen_strategy_dict = resume_chosen_strategy_dict
                dba_message = ""  # No customization on resume

                ui.print()
                ui.print(f"[cyan]Using saved strategy: {strategy.name}[/]" if RICH_AVAILABLE else f"Using saved strategy: {strategy.name}")
            else:
                ui.print()
                if RICH_AVAILABLE and ui.console:
                    ui.console.rule("[bold blue]Strategy Selection[/]")
                else:
                    ui.print("=" * 50)
                    ui.print("Strategy Selection")
                    ui.print("=" * 50)

                # Step 5: Strategy Selection
                strategy_idx = ui.select_strategy(first_sight.strategy_options)
                selected_option = first_sight.strategy_options[strategy_idx]

                # Update session state with strategy info
                session_state.strategy_name = selected_option.name
                session_state.strategy_id = selected_option.id
                session_state.target_tps = selected_option.target_kpis.get('target_tps', 0)

                # ===== CREATE WORKSPACE SESSION =====
                # Auto-create session when strategy is selected (unless resuming)
                if ws_session is None or ws_session.state.value == 'archived':
                    ws_session = workspace.create_session(
                        strategy_name=selected_option.name,
                        strategy_id=selected_option.id,
                        rationale=selected_option.hypothesis,
                    )
                    # Set phase to STRATEGY_SELECTION
                    ws_session.set_phase(SessionPhase.STRATEGY_SELECTION)
                    # Save first sight analysis
                    ws_session.set_first_sight(
                        analysis=first_sight.executive_summary if hasattr(first_sight, 'executive_summary') else '',
                        bottleneck=first_sight.bottleneck if hasattr(first_sight, 'bottleneck') else '',
                        confidence=first_sight.confidence if hasattr(first_sight, 'confidence') else 0.0,
                    )
                ui.show_status_line()

                ui.print()
                ui.print(f"Selected: [cyan]{selected_option.name}[/]" if RICH_AVAILABLE else f"Selected: {selected_option.name}")

                # Step 6: DBA Customization (optional)
                dba_message = ui.get_dba_customization()

                # Step 7: Generate benchmark plan from AI (needed for baseline)
                ui.print()
                ui.spinner_start("Generating benchmark plan...")

                chosen_strategy_dict = {
                    'id': selected_option.id,
                    'name': selected_option.name,
                    'goal': selected_option.goal,
                    'hypothesis': selected_option.hypothesis,
                    'target_kpis': selected_option.target_kpis,
                }

                strategy = agent.generate_strategy({
                    'system_context': context_packet.system_context,
                    'runtime_context': context_packet.runtime_context,
                    'schema_context': context_packet.schema_context,
                    'selected_strategy': chosen_strategy_dict,
                    'dba_customization': dba_message,
                })
                ui.spinner_stop("Plan ready")

            # ===== SKIP BASELINE IF RESUMING WITH ROUNDS =====
            if skip_to_tuning_loop:
                # Use saved values from resume
                true_baseline_tps = resume_true_baseline_tps
                target_tps = resume_target_tps
                ui.print()
                ui.print(f"[dim]Baseline TPS: {true_baseline_tps:,.0f} | Target: {target_tps:,.0f}[/]" if RICH_AVAILABLE else f"Baseline: {true_baseline_tps:.0f} | Target: {target_tps:.0f}")
            else:
                # Step 8: Run BASELINE benchmark (before any config changes)
                # Set phase to BASELINE
                if ws_session:
                    ws_session.set_phase(SessionPhase.BASELINE)
                    ws_session.save_checkpoint("Before baseline benchmark")

                ui.print()
                if RICH_AVAILABLE and ui.console:
                    ui.console.rule("[bold yellow]Baseline Benchmark[/]")
                    ui.console.print("[dim]Running benchmark with CURRENT configuration (no changes applied yet)[/]")
                else:
                    ui.print("=" * 50)
                    ui.print("Baseline Benchmark (before any changes)")
                    ui.print("=" * 50)

                if not ui.confirm_benchmark(strategy):
                    ui.print("Benchmark cancelled.")
                    ws_session = None  # Reset for new session on next strategy
                    skip_to_baseline = False  # Reset skip flags
                    skip_to_tuning_loop = False
                    continue  # Go back to strategy selection

                ui.print()
                baseline_run_result = run_benchmark_with_telemetry(
                    conn=conn,
                    strategy=strategy,
                    db_host=args.host,
                    db_port=args.port,
                    db_name=database,
                    db_user=args.user,
                    db_password=password,
                    ssh_config=ssh_config,
                    ui=ui,
                    session_state=session_state,
                    round_num=0,  # Baseline is round 0
                )

                baseline_benchmark_result = baseline_run_result['benchmark_result']
                baseline_telemetry = baseline_run_result['telemetry']
                true_baseline_tps = baseline_benchmark_result.metrics.tps if baseline_benchmark_result.metrics else 0

                # ===== CHECKPOINT: Save baseline to workspace session =====
                baseline_latency_avg = baseline_benchmark_result.metrics.latency_avg_ms if baseline_benchmark_result.metrics else 0
                baseline_latency_p99 = getattr(baseline_benchmark_result.metrics, 'latency_p99_ms', 0) if baseline_benchmark_result.metrics else 0
                ws_session.save_baseline(
                    tps=true_baseline_tps,
                    latency_avg=baseline_latency_avg,
                    latency_p99=baseline_latency_p99,
                    metrics=baseline_telemetry,
                    raw_output=baseline_benchmark_result.raw_output if hasattr(baseline_benchmark_result, 'raw_output') else '',
                )
                ui.show_status_line()

                # Display baseline results (without target comparison yet)
                ui.print()
                if ui.console:
                    ui.console.print(f"[cyan]Baseline TPS:[/] [bold]{true_baseline_tps:.0f}[/]")
                else:
                    ui.print(f"Baseline TPS: {true_baseline_tps:.0f}")

                # ===== ERROR HANDLING: 0 TPS baseline =====
                if true_baseline_tps == 0:
                    ui.print()
                    if ui.console:
                        ui.console.print("[red]ERROR: Baseline benchmark returned 0 TPS[/]")
                        ui.console.print("[yellow]This usually means the benchmark failed to run properly.[/]")
                        ui.console.print("[dim]Check the debug output above for error details.[/]")
                    else:
                        ui.print("ERROR: Baseline benchmark returned 0 TPS")
                        ui.print("Check the debug output above for error details.")

                    # Set error state
                    if ws_session:
                        ws_session.set_error(
                            error_type="benchmark_failed",
                            message="Baseline benchmark returned 0 TPS. Check benchmark configuration.",
                            recoverable=True
                        )

                    ui.show_context_menu('session_end')
                    error_choice = ui.prompt("Options: /retry to try again, /skip to continue anyway, Enter to quit: ").strip().lower()

                    if error_choice in ['/retry', 'retry', 'r']:
                        if ws_session:
                            ws_session.resume(SessionPhase.BASELINE)
                        continue  # Restart strategy selection
                    elif error_choice in ['/skip', 'skip', 's']:
                        ui.print("[yellow]Continuing with 0 TPS baseline (results may be unreliable)[/]" if ui.console else "Continuing with 0 TPS baseline")
                        true_baseline_tps = 1  # Avoid division by zero
                    else:
                        ui.print("Exiting due to benchmark failure.")
                        if ws_session:
                            ws_session.fail("Baseline benchmark failed with 0 TPS")
                        break

                # ===== STEP 8b: AI suggests target based on baseline =====
                ui.print()
                ui.spinner_start("AI analyzing baseline and suggesting target...")

                try:
                    # Ask AI to suggest target based on baseline
                    target_suggestion = agent.suggest_target(
                        baseline_tps=true_baseline_tps,
                        context=context_packet,
                        strategy=chosen_strategy_dict,
                    )
                    ui.spinner_stop("Target suggestion ready")

                    suggested_target = target_suggestion.get('target_tps', int(true_baseline_tps * 1.2))
                    improvement_pct = ((suggested_target - true_baseline_tps) / true_baseline_tps * 100) if true_baseline_tps > 0 else 20
                    rationale = target_suggestion.get('rationale', 'Based on system capacity analysis')

                except Exception as e:
                    ui.spinner_stop("Using default target")
                    # Fallback: suggest 20% improvement
                    suggested_target = int(true_baseline_tps * 1.2)
                    improvement_pct = 20
                    rationale = "Default 20% improvement target"

                # Display suggestion
                ui.print()
                if ui.console:
                    ui.console.print(f"[bold]Target Suggestion:[/]")
                    ui.console.print(f"  Baseline: [cyan]{true_baseline_tps:,.0f}[/] TPS")
                    ui.console.print(f"  Suggested Target: [green]{suggested_target:,.0f}[/] TPS ([green]+{improvement_pct:.0f}%[/])")
                    ui.console.print(f"  [dim]Rationale: {rationale}[/]")
                else:
                    ui.print(f"Target Suggestion:")
                    ui.print(f"  Baseline: {true_baseline_tps:,.0f} TPS")
                    ui.print(f"  Suggested Target: {suggested_target:,.0f} TPS (+{improvement_pct:.0f}%)")
                    ui.print(f"  Rationale: {rationale}")

                # Ask DBA to confirm or set custom target - command-first pattern
                ui.print()
                ui.show_context_menu('target_confirm')

                target_tps = None
                while True:
                    target_cmd = ui.prompt("> ").strip()

                    if target_cmd.lower() in ['/accept', '/a', 'y', 'yes', '']:
                        target_tps = suggested_target
                        ui.print(f"[green]Target accepted: {target_tps:,.0f} TPS[/]" if ui.console else f"Target: {target_tps:.0f} TPS")
                        break

                    if target_cmd.lower() in ['/back', '/b', 'n', 'no']:
                        ui.print("[yellow]Target declined. Returning to strategy selection.[/]" if ui.console else "Target declined.")
                        target_tps = None
                        break

                    if target_cmd.lower().startswith('/set'):
                        # Extract value: /set 5000 or /set5000
                        value_str = target_cmd[4:].strip()
                        if value_str:
                            try:
                                target_tps = float(value_str)
                                ui.print(f"[green]Custom target set: {target_tps:,.0f} TPS[/]" if ui.console else f"Custom target: {target_tps:.0f} TPS")
                                break
                            except ValueError:
                                ui.print(f"[yellow]Invalid number: {value_str}[/]" if ui.console else f"Invalid number: {value_str}")
                                continue
                        # Prompt for value
                        value_input = ui.prompt("Enter target TPS: ", allow_commands=False).strip()
                        try:
                            target_tps = float(value_input)
                            ui.print(f"[green]Custom target set: {target_tps:,.0f} TPS[/]" if ui.console else f"Custom target: {target_tps:.0f} TPS")
                            break
                        except ValueError:
                            ui.print(f"[yellow]Invalid number. Using suggested: {suggested_target:,.0f}[/]" if ui.console else f"Using suggested: {suggested_target:.0f}")
                            target_tps = suggested_target
                            break

                    # Try to parse as a number directly
                    try:
                        target_tps = float(target_cmd)
                        ui.print(f"[green]Custom target set: {target_tps:,.0f} TPS[/]" if ui.console else f"Custom target: {target_tps:.0f} TPS")
                        break
                    except ValueError:
                        pass

                    ui.print(f"[yellow]Unknown command: {target_cmd}. Type /accept or /set <value>.[/]" if ui.console else f"Unknown command: {target_cmd}")

                if target_tps is None:
                    ws_session = None  # Reset for new session on next strategy
                    skip_to_baseline = False
                    skip_to_tuning_loop = False
                    continue  # Go back to strategy selection

                # Update session state with confirmed target
                session_state.target_tps = target_tps

                # Update workspace session with confirmed target
                ws_session.data.target_tps = target_tps
                ws_session.save()
                ui.show_status_line()

            # ===== SKIP TO TUNING LOOP IF RESUMING WITH ROUNDS =====
            resumed_to_tuning_loop = False  # Track if we took the resume path

            if skip_to_tuning_loop:
                resumed_to_tuning_loop = True

                # Use saved tuning history from resume
                tuning_history = resume_tuning_history
                round_num = resume_round_num

                # Show resume summary
                ui.print()
                if RICH_AVAILABLE and ui.console:
                    ui.console.rule(f"[bold green]Resuming at Round {round_num}[/]")
                else:
                    ui.print("=" * 50)
                    ui.print(f"Resuming at Round {round_num}")
                    ui.print("=" * 50)

                ui.print(f"Previous rounds: {tuning_history['iterations_completed']}")
                ui.print(f"Best TPS so far: {tuning_history['best_tps']:,.0f}")

                # Sync session state
                session_state.baseline_tps = true_baseline_tps
                session_state.current_tps = tuning_history['best_tps']
                session_state.best_tps = tuning_history['best_tps']
                session_state.tps_history = tuning_history['tps_history']
                session_state.applied_changes = tuning_history['applied_changes']

                # Get AI recommendations for next round
                ui.print()
                ui.spinner_start(f"AI preparing Round {round_num} recommendations...")
                analysis = agent.analyze_results(
                    strategy=strategy,
                    result=None,  # No result yet - resuming
                    telemetry_summary="Resuming session",
                    current_config=db_config,
                    target_kpis={'target_tps': target_tps},
                    human_feedback={'message': f'Resuming at Round {round_num}', 'round': round_num},
                )
                ui.spinner_stop("Ready")

                # Display AI analysis
                ui.display_ai_analysis(analysis)

                # Clear skip flags for next loop iteration
                skip_to_baseline = False
                skip_to_tuning_loop = False

            if not resumed_to_tuning_loop:
                # Run Round 1 and initial analysis (normal flow or resume from baseline)

                # Check if baseline already meets target
                if true_baseline_tps >= target_tps:
                    ui.print()
                    if ui.console:
                        ui.console.print(f"[bold yellow]⚠ Baseline ({true_baseline_tps:,.0f} TPS) already meets target ({target_tps:,.0f} TPS)![/]")
                        ui.console.print("[dim]Continue tuning to prevent regression and explore further optimization?[/]")
                    else:
                        ui.print(f"Warning: Baseline ({true_baseline_tps:.0f}) already meets target ({target_tps:.0f})!")
                        ui.print("Continue tuning to prevent regression?")

                    continue_choice = ui.prompt("Continue tuning? [Y/n]: ").strip().lower()
                    if continue_choice in ['n', 'no']:
                        ui.print("[green]Session complete. Baseline is optimal.[/]" if ui.console else "Session complete.")

                        # Display summary and ask for next strategy
                        tuning_history = {
                            'baseline_tps': true_baseline_tps,
                            'best_tps': true_baseline_tps,
                            'tps_history': [true_baseline_tps],
                            'iterations_completed': 0,
                            'applied_changes': [],
                        }
                        display_final_summary(ui, tuning_history, target_tps, True)

                        continue_strategy = ui.prompt("Try a different strategy? [y/N]: ").strip().lower()
                        if continue_strategy != 'y':
                            break
                        ws_session = None  # Reset for new session on next strategy
                        skip_to_baseline = False
                        skip_to_tuning_loop = False
                        continue  # Go to strategy selection

                # Step 9: Get Round 1 configuration from AI (using baseline telemetry)
                ui.print()
                ui.spinner_start("AI generating initial configuration based on baseline...")

                round1_config = agent.get_round1_config(
                    context=context_packet,
                    first_sight=first_sight,
                    chosen_strategy=chosen_strategy_dict,
                    dba_message=dba_message,
                )
                ui.spinner_stop("Initial configuration ready")

                # Step 10: Display and confirm Round 1 config
                initial_applied_changes = []  # Track round 0 changes
                if ui.display_round1_config(round1_config):
                    # Apply the configuration
                    if round1_config.tuning_chunks or round1_config.os_tuning:
                        apply_result = apply_round1_config(
                            conn=conn,
                            config=round1_config,
                            ssh_config=ssh_config,
                            ui=ui,
                        )
                        if not apply_result['success']:
                            ui.print_error("Failed to apply initial configuration. Aborting.")
                            sys.exit(1)

                        # Store round 0 applied changes
                        initial_applied_changes = apply_result.get('applied_changes', [])

                        # Reconnect if restart happened
                        if round1_config.restart_required:
                            conn.close()
                            try:
                                conn = reconnect_with_retry(
                                    args.host, args.port, args.user, password, database,
                                    ui, max_retries=5, timeout_sec=5
                                )
                                session_state.conn = conn
                            except Exception as e:
                                # Connection lost after restart - set ERROR state
                                if ws_session:
                                    ws_session.set_error(
                                        error_type="connection_lost",
                                        message=f"Failed to reconnect after config restart: {e}",
                                        recoverable=True
                                    )
                                ui.print_error(f"Reconnection failed: {e}")
                                ui.print("Session saved with error state. Use /retry to continue.")
                                continue  # Back to strategy selection

                # Step 11: Run Round 1 benchmark (after config applied)
                ui.print()
                if RICH_AVAILABLE and ui.console:
                    ui.console.rule("[bold green]Round 1 Benchmark[/]")
                    ui.console.print("[dim]Running benchmark with tuned configuration[/]")
                else:
                    ui.print("=" * 50)
                    ui.print("Round 1 Benchmark (after tuning)")
                    ui.print("=" * 50)

                ui.print()
                run_result = run_benchmark_with_telemetry(
                    conn=conn,
                    strategy=strategy,
                    db_host=args.host,
                    db_port=args.port,
                    db_name=database,
                    db_user=args.user,
                    db_password=password,
                    ssh_config=ssh_config,
                    ui=ui,
                    session_state=session_state,
                    round_num=1,  # Round 1
                )

                benchmark_result = run_result['benchmark_result']
                telemetry = run_result['telemetry']

                # ===== CHECKPOINT: Save Round 1 to workspace session =====
                round1_metrics_tps = benchmark_result.metrics.tps if benchmark_result.metrics else 0
                round1_latency_avg = benchmark_result.metrics.latency_avg_ms if benchmark_result.metrics else 0
                round1_latency_p99 = getattr(benchmark_result.metrics, 'latency_p99_ms', 0) if benchmark_result.metrics else 0
                ws_session.save_round(
                    round_num=1,
                    tps=round1_metrics_tps,
                    changes=initial_applied_changes,
                    latency_avg=round1_latency_avg,
                    latency_p99=round1_latency_p99,
                    metrics=telemetry,
                    raw_output=benchmark_result.raw_output if hasattr(benchmark_result, 'raw_output') else '',
                )
                ui.show_status_line()

                # Step 12: Display benchmark results
                ui.display_benchmark_result(benchmark_result, chosen_strategy_dict.get('target_kpis', {}))

                # Step 13: Get DBA feedback on Round 1 results
                ui.print()
                if ui.console:
                    ui.console.print("[cyan]DBA Feedback on Round 1:[/]")
                    ui.console.print("[dim]Provide feedback or instructions for AI to consider in next tuning round.[/]")
                    ui.console.print("[dim]Examples: 'enable synchronous_commit', 'focus on WAL tuning', 'skip memory changes'[/]")
                else:
                    ui.print("DBA Feedback on Round 1:")
                    ui.print("(e.g., 'enable synchronous_commit', 'focus on WAL tuning')")

                dba_feedback = ui.prompt("Custom instructions (Enter to skip): ").strip()

                human_feedback = None
                if dba_feedback:
                    human_feedback = {
                        'message': dba_feedback,
                        'round': 1,  # Feedback on Round 1
                    }
                    ui.print(f"[green]Feedback noted[/]" if ui.console else "Feedback noted")

                # Step 14: Send results to AI for analysis
                ui.print()
                ui.spinner_start("AI analyzing results...")

                telemetry_summary = format_telemetry_summary(telemetry)
                analysis = agent.analyze_results(
                    strategy=strategy,
                    result=benchmark_result,
                    telemetry_summary=telemetry_summary,
                    current_config=db_config,
                    target_kpis=chosen_strategy_dict.get('target_kpis', {}),
                    human_feedback=human_feedback,
                )
                ui.spinner_stop("Analysis complete")

                # Step 15: Display AI analysis
                ui.display_ai_analysis(analysis)

                # ===================== TUNING ROUNDS =====================
                # Track tuning history
                round1_tps = benchmark_result.metrics.tps if benchmark_result.metrics else 0
                round1_improvement = ((round1_tps - true_baseline_tps) / true_baseline_tps * 100) if true_baseline_tps > 0 else 0

                # Show Round 1 improvement from baseline
                ui.print()
                if ui.console:
                    if round1_improvement > 0:
                        ui.console.print(f"[bold green]Round 1 Result: {round1_tps:.0f} TPS (+{round1_improvement:.1f}% from baseline)[/]")
                    else:
                        ui.console.print(f"[yellow]Round 1 Result: {round1_tps:.0f} TPS ({round1_improvement:.1f}% from baseline)[/]")
                else:
                    ui.print(f"Round 1 Result: {round1_tps:.0f} TPS ({round1_improvement:+.1f}% from baseline)")

                tuning_history = {
                    'iterations_completed': 1,  # Round 1 is complete
                    'baseline_tps': true_baseline_tps,  # TRUE baseline (before any config changes)
                    'tps_history': [true_baseline_tps, round1_tps],  # [baseline, round1]
                    'applied_changes': initial_applied_changes.copy(),  # Include round 0 changes
                    'best_tps': round1_tps,  # Best TPS after tuning
                    'best_config_snapshot': {},
                    'no_improvement_rounds': 0,
                }

                # Sync session state for slash commands
                session_state.baseline_tps = true_baseline_tps
                session_state.current_tps = round1_tps
                session_state.best_tps = round1_tps

                # Clear skip flags
                skip_to_baseline = False
                skip_to_tuning_loop = False

                # Set round_num for tuning loop
                round_num = 2  # Next round is Round 2 (Round 1 was the initial benchmark)

                # Check if target already hit after Round 1
                if target_tps > 0 and round1_tps >= target_tps:
                    ui.print()
                    ui.print(f"[bold green]🎉 Target TPS {target_tps:.0f} achieved in Round 1! (current: {round1_tps:.0f}, +{round1_improvement:.1f}% from baseline)[/]" if ui.console else f"Target TPS {target_tps:.0f} achieved in Round 1! (+{round1_improvement:.1f}%)")
                    ui.print(f"[green]Skipping further tuning rounds - target already met.[/]" if ui.console else "Skipping further tuning - target met.")

                    # Jump to final summary
                    target_hit_final = True
                    display_final_summary(ui, tuning_history, target_tps, target_hit_final)

                    # ===== ARCHIVE WORKSPACE SESSION (R1 success) =====
                    conclusion = f"Target achieved in Round 1! Best TPS: {round1_tps:.0f}"
                    ws_session.archive(conclusion)
                    ui.show_status_line()

                    if RICH_AVAILABLE and tuning_history['tps_history']:
                        timeline = ProgressTimeline(ui.console)
                        timeline.display(tuning_history['tps_history'], target_tps,
                                       tuning_history['iterations_completed'], target_achieved=True)

                # Session save and continue prompt
                if args.save_session:
                    from .session import SessionManager, session_from_state
                    manager = SessionManager()
                    session_state.current_round = tuning_history['iterations_completed']
                    session_state.tps_history = tuning_history['tps_history']
                    session_state.applied_changes = tuning_history['applied_changes']
                    session_state.best_tps = tuning_history['best_tps']
                    session_data = session_from_state(session_state)
                    session_data.status = 'completed'
                    manager.save(session_data, args.save_session)
                    ui.print(f"\n[green]Session saved as '{args.save_session}'[/]" if ui.console else f"\nSession saved as '{args.save_session}'")

                ui.print()
                ui.show_context_menu('session_end')
                continue_choice = ui.prompt("Continue with a different strategy? [y/N]: ").strip().lower()

                # Handle slash commands
                if continue_choice in ['/quit', '/q', 'n', 'no', '']:
                    ui.print()
                    if ui.console:
                        ui.console.print("[bold green]Thank you for using PostgreSQL Diagnostic Tool![/]")
                    else:
                        ui.print("Thank you for using PostgreSQL Diagnostic Tool!")
                    break
                if continue_choice in ['/export', '/e']:
                    if ws_session:
                        ui.print(f"Session data saved in workspace: {workspace.name}")
                    continue
                if continue_choice not in ['y', 'yes', '/new', '/n']:
                    break

                # ===== GET NEW STRATEGIES FROM AI (after R1 success) =====
                previous_session_summary = {
                    'baseline_tps': tuning_history['baseline_tps'],
                    'best_tps': tuning_history['best_tps'],
                    'improvement_pct': round1_improvement,
                    'rounds_completed': 1,
                    'target_tps': target_tps,
                    'target_achieved': True,
                    'applied_changes': tuning_history['applied_changes'],
                }

                excluded_strategy_info = {
                    'id': selected_option.id,
                    'name': selected_option.name,
                    'goal': selected_option.goal,
                    'hypothesis': selected_option.hypothesis,
                    'target_kpis': selected_option.target_kpis,
                }

                ui.print()
                ui.spinner_start("AI analyzing successful session and generating new strategies...")

                try:
                    first_sight = agent.get_next_strategies(
                        context=context_packet,
                        previous_session=previous_session_summary,
                        excluded_strategy=excluded_strategy_info,
                    )
                    ui.spinner_stop("New strategies ready")
                    ui.display_first_sight(first_sight)
                except Exception as e:
                    ui.spinner_stop("Failed to get new strategies")
                    ui.print_error(f"Error getting new strategies: {e}")
                    ui.print("Falling back to original strategy options...")
                    ui.display_first_sight(first_sight)

                ws_session = None  # Reset for new session on next strategy
                continue  # Go to strategy selection with new strategies

            # ===== TUNING LOOP (accessible from both resume and normal paths) =====
            # Initialize tuning loop tracking variables
            max_no_target_hit = 3  # Break after 3 rounds not hitting target
            consecutive_hits_required = 2  # Need 2 consecutive hits to confirm success
            no_target_hit_count = 0
            consecutive_target_hits = 0

            # Check if first analysis is already a conclusion
            from .protocol.conclusion import SessionConclusion

            while True:
                # Set phase to TUNING and save checkpoint at start of each round
                if ws_session:
                    ws_session.set_phase(SessionPhase.TUNING)
                    ws_session.save_checkpoint(f"Starting tuning round {round_num}")

                ui.print()
                if RICH_AVAILABLE and ui.console:
                    ui.console.rule(f"[cyan]Tuning Round {round_num}[/]")
                else:
                    ui.print(f"\n--- Tuning Round {round_num} ---")

                # DBA command prompt at the start of each round
                ui.print()
                if ui.console:
                    ui.console.print("[cyan]Round Action[/]")
                else:
                    ui.print("Round Action:")
                ui.show_context_menu('tuning_round')

                round_custom_input = None
                while True:
                    cmd = ui.prompt("> ").strip().lower()

                    if cmd in ['/go', '/g', '']:
                        break  # Continue without custom input

                    if cmd in ['/stop', '/pause', '/quit', '/q']:
                        ui.print()
                        if ui.console:
                            ui.console.print("[yellow]Pausing session...[/]")
                        else:
                            ui.print("Pausing session...")
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused at Round {round_num}. Use /resume to continue later.")
                        break

                    if cmd in ['/status', '/s']:
                        ui.print(f"Round: {round_num} | Best TPS: {tuning_history['best_tps']:.0f} | Target: {target_tps:.0f}")
                        continue

                    if cmd in ['/history', '/h']:
                        ui.print("Tuning History:")
                        for change in tuning_history.get('applied_changes', []):
                            ui.print(f"  R{change['round']}: {change['name']}")
                        continue

                    if cmd in ['/done', '/d']:
                        ui.print("Ending session with current results...")
                        round_custom_input = 'DONE'
                        break

                    if cmd.startswith('/custom'):
                        # Extract inline text or prompt for it
                        inline_text = cmd[7:].strip()
                        if inline_text:
                            round_custom_input = inline_text
                            break
                        if ui.console:
                            ui.console.print("[dim]Enter your custom instructions:[/]")
                        else:
                            print("Enter your custom instructions:")
                        custom_text = ui.prompt("Instructions: ", allow_commands=False).strip()
                        if custom_text:
                            round_custom_input = custom_text
                        break

                    ui.print(f"[yellow]Unknown command: {cmd}. Type /go to continue or /custom for instructions.[/]" if ui.console else f"Unknown command: {cmd}")

                # Handle /stop and /done exits
                if cmd in ['/stop', '/pause', '/quit', '/q']:
                    break  # Exit tuning loop

                if round_custom_input == 'DONE':
                    break  # Exit tuning loop with current results

                # If user provided custom input, regenerate AI analysis with feedback
                if round_custom_input:
                    ui.print()
                    ui.spinner_start(f"AI regenerating Round {round_num} recommendations with your input...")
                    analysis = agent.analyze_results(
                        strategy=strategy,
                        result=None,
                        telemetry_summary="Regenerating with DBA feedback",
                        current_config=db_config,
                        target_kpis=chosen_strategy_dict.get('target_kpis', {}),
                        tuning_history=tuning_history,
                        human_feedback={'message': round_custom_input, 'round': round_num},
                    )
                    ui.spinner_stop("Ready")

                # Check analysis type
                if hasattr(analysis, 'conclusion_reason') or isinstance(analysis, SessionConclusion):
                    # AI concluded - hardware saturated or success
                    ui.print()
                    ui.print("[yellow]AI concluded tuning session[/]" if ui.console else "AI concluded tuning session")
                    if hasattr(analysis, 'conclusion_reason'):
                        ui.print(f"Reason: {analysis.conclusion_reason}")
                    break

                # It's a TuningProposal - show and confirm
                if not hasattr(analysis, 'tuning_chunks') or not analysis.tuning_chunks:
                    ui.print("[yellow]No tuning recommendations from AI[/]" if ui.console else "No tuning recommendations")
                    break

                # Display tuning proposal
                ui.print()
                ui.print(f"[cyan]AI Recommendation:[/] {len(analysis.tuning_chunks)} changes" if ui.console else f"AI Recommendation: {len(analysis.tuning_chunks)} changes")

                for i, chunk in enumerate(analysis.tuning_chunks, 1):
                    if ui.console:
                        ui.console.print(f"  {i}. [green]{chunk.name}[/] ({chunk.category})")
                        # Show specific config changes
                        if hasattr(chunk, 'apply_commands') and chunk.apply_commands:
                            for cmd in chunk.apply_commands:
                                # Extract the setting from ALTER SYSTEM SET commands
                                if 'ALTER SYSTEM SET' in cmd.upper():
                                    ui.console.print(f"     [yellow]→ {cmd}[/]")
                                elif 'pg_reload_conf' not in cmd.lower():
                                    ui.console.print(f"     [dim]→ {cmd}[/]")
                        if chunk.rationale:
                            ui.console.print(f"     [dim]{chunk.rationale[:100]}...[/]" if len(chunk.rationale) > 100 else f"     [dim]{chunk.rationale}[/]")
                    else:
                        ui.print(f"  {i}. {chunk.name} ({chunk.category})")
                        if hasattr(chunk, 'apply_commands') and chunk.apply_commands:
                            for cmd in chunk.apply_commands:
                                if 'ALTER SYSTEM SET' in cmd.upper():
                                    ui.print(f"     → {cmd}")

                # Ask to apply - command-first pattern
                ui.print()
                ui.show_context_menu('tuning_apply')

                apply_action = None
                while True:
                    apply_cmd = ui.prompt("> ").strip().lower()

                    if apply_cmd in ['/apply', '/a', 'y', 'yes']:
                        apply_action = 'apply'
                        break

                    if apply_cmd in ['/stop', '/pause']:
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused at Round {round_num}.")
                        apply_action = 'stop'
                        break

                    if apply_cmd in ['/skip', '/s']:
                        ui.print("Skipping this round...")
                        apply_action = 'skip'
                        break

                    if apply_cmd in ['/details', '/d']:
                        for chunk in analysis.tuning_chunks:
                            ui.print(f"\n{chunk.name}:")
                            ui.print(f"  Rationale: {chunk.rationale}")
                            for apply_c in chunk.apply_commands:
                                ui.print(f"  → {apply_c}")
                        continue

                    if apply_cmd in ['n', 'no', '']:
                        ui.print("Tuning session ended by user.")
                        apply_action = 'end'
                        break

                    ui.print(f"[yellow]Unknown command: {apply_cmd}. Type /apply to apply or /skip to skip.[/]" if ui.console else f"Unknown command: {apply_cmd}")

                if apply_action == 'stop':
                    break
                if apply_action == 'skip':
                    round_num += 1
                    continue
                if apply_action == 'end':
                    break
                if apply_action != 'apply':
                    continue

                # Apply the tuning
                ui.spinner_start(f"Applying tuning round {round_num}...")

                conn.rollback()
                old_autocommit = conn.autocommit
                conn.autocommit = True

                applied_this_round = []
                restart_needed = False

                try:
                    for chunk in analysis.tuning_chunks:
                        try:
                            with conn.cursor() as cur:
                                for cmd in chunk.apply_commands:
                                    cur.execute(cmd)
                            applied_this_round.append(chunk.name)
                            # Extract ALTER SYSTEM commands for summary
                            pg_configs = [cmd for cmd in chunk.apply_commands
                                         if 'ALTER SYSTEM SET' in cmd.upper()]
                            tuning_history['applied_changes'].append({
                                'round': round_num,
                                'name': chunk.name,
                                'category': chunk.category,
                                'pg_configs': pg_configs,
                            })
                            if chunk.requires_restart:
                                restart_needed = True
                        except Exception as e:
                            ui.print(f"[yellow]Warning: {chunk.name} failed: {e}[/]" if ui.console else f"Warning: {chunk.name} failed: {e}")
                finally:
                    conn.autocommit = old_autocommit

                ui.spinner_stop(f"Applied {len(applied_this_round)} changes")

                # Handle restart if needed
                if restart_needed and ssh_config:
                    restart_confirm = ui.prompt("PostgreSQL restart required. Restart now? [y/N]: ").lower()
                    if restart_confirm == 'y':
                        if restart_postgresql_with_retry(ssh_config, ui, max_retries=5, timeout_sec=5):
                            conn.close()
                            try:
                                conn = reconnect_with_retry(
                                    args.host, args.port, args.user, password, database,
                                    ui, max_retries=5, timeout_sec=5
                                )
                                session_state.conn = conn
                            except Exception as e:
                                # Connection lost - set ERROR state
                                if ws_session:
                                    ws_session.set_error(
                                        error_type="connection_lost",
                                        message=f"Failed to reconnect after restart: {e}",
                                        recoverable=True
                                    )
                                ui.print_error(f"Reconnection failed: {e}")
                                break
                        else:
                            # Restart failed - set ERROR state
                            if ws_session:
                                ws_session.set_error(
                                    error_type="restart_failed",
                                    message="PostgreSQL restart failed",
                                    recoverable=True
                                )
                            ui.print_error("PostgreSQL restart failed. Session saved with error state.")
                            break

                # Run benchmark for this round
                ui.print()
                ui.display_benchmark_config(strategy, round_num)
                run_result = run_benchmark_with_telemetry(
                    conn=conn,
                    strategy=strategy,
                    db_host=args.host,
                    db_port=args.port,
                    db_name=database,
                    db_user=args.user,
                    db_password=password,
                    ssh_config=ssh_config,
                    ui=ui,
                    session_state=session_state,
                    round_num=round_num,
                )

                benchmark_result = run_result['benchmark_result']
                telemetry = run_result['telemetry']
                current_tps = benchmark_result.metrics.tps if benchmark_result.metrics else 0

                # ===== ERROR HANDLING: 0 TPS in tuning round =====
                if current_tps == 0:
                    ui.print()
                    if ui.console:
                        ui.console.print(f"[red]ERROR: Round {round_num} benchmark returned 0 TPS[/]")
                        ui.console.print("[yellow]The benchmark may have failed. Check debug output above.[/]")
                    else:
                        ui.print(f"ERROR: Round {round_num} benchmark returned 0 TPS")

                    # Record error but don't transition to ERROR state yet
                    # Allow user to decide what to do
                    ui.show_context_menu('tuning_dba_post')
                    error_action = ui.prompt("Options: /retry to retry this round, /stop to pause, Enter to continue: ").strip().lower()

                    if error_action in ['/retry', 'retry', 'r']:
                        ui.print("Retrying round...")
                        continue  # Retry the same round
                    elif error_action in ['/stop', '/pause', 'stop']:
                        if ws_session:
                            ws_session.set_error(
                                error_type="benchmark_failed",
                                message=f"Round {round_num} benchmark returned 0 TPS",
                                recoverable=True
                            )
                        break  # Exit tuning loop

                # ===== CHECKPOINT: Save round to workspace session =====
                if ws_session:
                    round_latency_avg = benchmark_result.metrics.latency_avg_ms if benchmark_result.metrics else 0
                    round_latency_p99 = getattr(benchmark_result.metrics, 'latency_p99_ms', 0) if benchmark_result.metrics else 0
                    # Get changes applied this round from tuning_history
                    round_changes = [c for c in tuning_history.get('applied_changes', []) if c.get('round') == round_num]
                    ws_session.save_round(
                        round_num=round_num,
                        tps=current_tps,
                        changes=round_changes,
                        latency_avg=round_latency_avg,
                        latency_p99=round_latency_p99,
                        metrics=telemetry,
                        raw_output=benchmark_result.raw_output if hasattr(benchmark_result, 'raw_output') else '',
                    )
                    ui.show_status_line()

                # Display results
                ui.display_benchmark_result(benchmark_result, chosen_strategy_dict.get('target_kpis', {}))

                # Track progress
                tuning_history['tps_history'].append(current_tps)
                tuning_history['iterations_completed'] = round_num

                # Sync session state for slash commands
                session_state.current_tps = current_tps
                session_state.current_round = round_num
                session_state.tps_history = tuning_history['tps_history']
                session_state.applied_changes = tuning_history['applied_changes']

                # Check improvement
                prev_tps = tuning_history['tps_history'][-2] if len(tuning_history['tps_history']) > 1 else 0
                improvement_pct = ((current_tps - prev_tps) / prev_tps * 100) if prev_tps > 0 else 0

                if current_tps > tuning_history['best_tps']:
                    tuning_history['best_tps'] = current_tps
                    session_state.best_tps = current_tps
                    ui.print(f"[green]New best TPS: {current_tps:.0f} (+{improvement_pct:.1f}%)[/]" if ui.console else f"New best TPS: {current_tps:.0f} (+{improvement_pct:.1f}%)")

                # Check if target hit (with ±10% permissible range)
                target_hit = False
                if target_tps > 0:
                    target_low = target_tps * 0.9   # -10%
                    target_high = target_tps * 1.1  # +10%

                    if current_tps >= target_low:
                        target_hit = True
                        no_target_hit_count = 0
                        consecutive_target_hits += 1

                        if current_tps >= target_tps:
                            ui.print(f"[bold green]🎉 Target TPS {target_tps:.0f} achieved! (current: {current_tps:.0f}) [{consecutive_target_hits}/{consecutive_hits_required} confirmations][/]" if ui.console else f"Target TPS {target_tps:.0f} achieved! [{consecutive_target_hits}/{consecutive_hits_required}]")
                        else:
                            ui.print(f"[green]Within target range ({target_low:.0f}-{target_high:.0f} TPS) [{consecutive_target_hits}/{consecutive_hits_required} confirmations][/]" if ui.console else f"Within target range [{consecutive_target_hits}/{consecutive_hits_required}]")

                        # Only break after consecutive_hits_required confirmations
                        if consecutive_target_hits >= consecutive_hits_required:
                            ui.print(f"[bold green]Target confirmed with {consecutive_target_hits} consecutive hits![/]" if ui.console else f"Target confirmed!")
                            break
                        # Continue to next round for confirmation
                    else:
                        consecutive_target_hits = 0  # Reset on miss
                        no_target_hit_count += 1
                        remaining = max_no_target_hit - no_target_hit_count
                        ui.print(f"[yellow]Target not hit ({no_target_hit_count}/{max_no_target_hit}). Need {target_low:.0f}+ TPS[/]" if ui.console else f"Target not hit ({no_target_hit_count}/{max_no_target_hit})")

                        # Check if we should break - end game scenario
                        if no_target_hit_count >= max_no_target_hit:
                            ui.print()
                            if ui.console:
                                ui.console.print()
                                ui.console.rule("[bold yellow]Session Ending[/]")
                                ui.console.print(f"[yellow]Target not achieved after {max_no_target_hit} consecutive misses.[/]")
                                ui.console.print(f"[dim]Best TPS: {tuning_history['best_tps']:.0f} | Target: {target_tps:.0f}[/]")
                                ui.console.print()
                                ui.console.print("[cyan]Consider trying a different strategy.[/]")
                            else:
                                ui.print(f"Target not achieved after {max_no_target_hit} rounds.")
                                ui.print(f"Best TPS: {tuning_history['best_tps']:.0f} | Target: {target_tps:.0f}")

                            # Mark session as FAILED (not just save - proper state transition)
                            if ws_session:
                                conclusion = f"Target not achieved after {max_no_target_hit} consecutive misses. Best TPS: {tuning_history['best_tps']:.0f}"
                                ws_session.fail(conclusion)
                                ui.show_status_line()
                            break

                # Get DBA action for next round - command-first pattern
                ui.print()
                if ui.console:
                    ui.console.print("[cyan]Next Round Action[/]")
                else:
                    ui.print("Next Round Action:")
                ui.show_context_menu('tuning_round')

                human_feedback = None
                next_action = None
                while True:
                    next_cmd = ui.prompt("> ").strip().lower()

                    if next_cmd in ['/go', '/g', '']:
                        next_action = 'continue'
                        break

                    if next_cmd in ['/stop', '/pause', '/quit', '/q']:
                        ui.print()
                        ui.print("[yellow]Pausing session...[/]" if ui.console else "Pausing session...")
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"[green]Session paused. Use /resume to continue later.[/]" if ui.console else "Session paused.")
                        next_action = 'stop'
                        break

                    if next_cmd in ['/status', '/s']:
                        ui.print(f"Round: {round_num} | Best TPS: {tuning_history['best_tps']:.0f} | Target: {target_tps:.0f}")
                        continue

                    if next_cmd in ['/history', '/h']:
                        ui.print("Tuning History:")
                        for change in tuning_history.get('applied_changes', []):
                            ui.print(f"  R{change['round']}: {change['name']}")
                        continue

                    if next_cmd in ['/done', '/d']:
                        ui.print("Ending session with current results...")
                        next_action = 'done'
                        break

                    if next_cmd.startswith('/custom'):
                        # Extract inline text or prompt for it
                        inline_text = next_cmd[7:].strip()
                        if inline_text:
                            human_feedback = {'message': inline_text, 'round': round_num}
                            ui.print(f"[green]Custom instructions noted for Round {round_num + 1}[/]" if ui.console else f"Instructions noted")
                            next_action = 'continue'
                            break
                        if ui.console:
                            ui.console.print("[dim]Enter your custom instructions:[/]")
                        else:
                            print("Enter your custom instructions:")
                        custom_text = ui.prompt("Instructions: ", allow_commands=False).strip()
                        if custom_text:
                            human_feedback = {'message': custom_text, 'round': round_num}
                            ui.print(f"[green]Custom instructions noted for Round {round_num + 1}[/]" if ui.console else f"Instructions noted")
                        next_action = 'continue'
                        break

                    ui.print(f"[yellow]Unknown command: {next_cmd}. Type /go to continue or /custom for instructions.[/]" if ui.console else f"Unknown command: {next_cmd}")

                if next_action == 'stop':
                    break
                if next_action == 'done':
                    break

                # Get AI analysis for next round
                ui.print()
                ui.spinner_start("AI analyzing for next round...")

                telemetry_summary = format_telemetry_summary(telemetry)
                analysis = agent.analyze_results(
                    strategy=strategy,
                    result=benchmark_result,
                    telemetry_summary=telemetry_summary,
                    current_config=db_config,
                    target_kpis=chosen_strategy_dict.get('target_kpis', {}),
                    tuning_history=tuning_history,
                    human_feedback=human_feedback,
                )
                ui.spinner_stop("Analysis ready")

                ui.display_ai_analysis(analysis)
                round_num += 1

            # ===================== FINAL SUMMARY =====================
            # Set phase to EVALUATING before final check
            if ws_session:
                ws_session.set_phase(SessionPhase.EVALUATING)

            target_hit_final = target_tps > 0 and tuning_history['best_tps'] >= (target_tps * 0.9)
            display_final_summary(ui, tuning_history, target_tps, target_hit_final)

            # Set phase to COMPLETED before archiving/failing
            if ws_session:
                ws_session.set_phase(SessionPhase.COMPLETED)

            # ===== ARCHIVE/FAIL WORKSPACE SESSION =====
            if ws_session:
                conclusion = f"Target {'achieved' if target_hit_final else 'not achieved'} after {tuning_history['iterations_completed']} rounds. Best TPS: {tuning_history['best_tps']:.0f}"
                if target_hit_final:
                    ws_session.archive(conclusion)
                else:
                    # Mark as FAILED when target not achieved
                    ws_session.fail(conclusion)
                ui.show_status_line()

            # Display progress timeline
            if RICH_AVAILABLE and tuning_history['tps_history']:
                timeline = ProgressTimeline(ui.console)
                timeline.display(tuning_history['tps_history'], target_tps,
                               tuning_history['iterations_completed'], target_achieved=target_hit_final)

            # Session save option
            if args.save_session:
                from .session import SessionManager, session_from_state
                manager = SessionManager()

                # Update session state from tuning history
                session_state.current_round = tuning_history['iterations_completed']
                session_state.tps_history = tuning_history['tps_history']
                session_state.applied_changes = tuning_history['applied_changes']
                session_state.best_tps = tuning_history['best_tps']

                session_data = session_from_state(session_state)
                session_data.status = 'completed' if target_hit_final else 'paused'

                manager.save(session_data, args.save_session)
                ui.print(f"\n[green]Session saved as '{args.save_session}'[/]" if ui.console else f"\nSession saved as '{args.save_session}'")

            # ===================== CONTINUE OR EXIT =====================
            ui.print()
            ui.show_context_menu('session_end')
            continue_choice = ui.prompt("Continue with a different strategy? [y/N]: ").strip().lower()

            # Handle slash commands
            if continue_choice in ['/quit', '/q', 'n', 'no', '']:
                ui.print()
                if ui.console:
                    ui.console.print("[bold green]Thank you for using PostgreSQL Diagnostic Tool![/]")
                    ui.console.print("[dim]Goodbye![/dim]")
                else:
                    ui.print("Thank you for using PostgreSQL Diagnostic Tool!")
                    ui.print("Goodbye!")
                break
            if continue_choice in ['/export', '/e']:
                if ws_session:
                    ui.print(f"Session data saved in workspace: {workspace.name}")
                continue
            if continue_choice not in ['y', 'yes', '/new', '/n']:
                break

            # ===================== GET NEW STRATEGIES FROM AI =====================
            # Build previous session summary for AI context
            previous_session_summary = {
                'baseline_tps': tuning_history['baseline_tps'],
                'best_tps': tuning_history['best_tps'],
                'improvement_pct': ((tuning_history['best_tps'] - tuning_history['baseline_tps']) / tuning_history['baseline_tps'] * 100) if tuning_history['baseline_tps'] > 0 else 0,
                'rounds_completed': tuning_history['iterations_completed'],
                'target_tps': target_tps,
                'target_achieved': target_hit_final,
                'applied_changes': tuning_history['applied_changes'],
            }

            # Build excluded strategy info
            excluded_strategy_info = {
                'id': chosen_strategy_dict.get('id', ''),
                'name': chosen_strategy_dict.get('name', ''),
                'goal': chosen_strategy_dict.get('goal', ''),
                'hypothesis': chosen_strategy_dict.get('hypothesis', ''),
                'target_kpis': chosen_strategy_dict.get('target_kpis', {}),
            }

            ui.print()
            ui.spinner_start("AI analyzing previous session and generating new strategies...")

            try:
                # Get new strategy options from AI (excluding previous strategy)
                first_sight = agent.get_next_strategies(
                    context=context_packet,
                    previous_session=previous_session_summary,
                    excluded_strategy=excluded_strategy_info,
                )
                ui.spinner_stop("New strategies ready")

                # Display the full first sight response including strategy options
                ui.display_first_sight(first_sight)

            except Exception as e:
                ui.spinner_stop("Failed to get new strategies")
                ui.print_error(f"Error getting new strategies: {e}")
                ui.print("Falling back to original strategy options...")
                # Keep the original first_sight if AI call fails
                # Display original strategies again
                ui.display_first_sight(first_sight)

            # Reset ws_session so new one is created for next strategy
            ws_session = None

            # Continue - loop back to strategy selection with new strategies

    except KeyboardInterrupt:
        ui.print("\nInterrupted by user")
        # Save session state before exiting
        if ws_session and ws_session.state == WorkspaceSessionState.ACTIVE:
            ui.print("[yellow]Saving session state...[/]" if ui.console else "Saving session state...")
            ws_session.pause()
            ui.print(f"[green]Session paused. Use /resume to continue later.[/]" if ui.console else "Session paused.")
        if workspace_manager:
            workspace_manager.close()
        sys.exit(130)

    except Exception as e:
        ui.print_error(str(e))
        import traceback
        traceback.print_exc()
        # Save error state before exiting
        if ws_session and ws_session.state == WorkspaceSessionState.ACTIVE:
            ws_session.set_error(
                error_type="unexpected_error",
                message=str(e),
                recoverable=True
            )
            ui.print(f"[yellow]Session saved with error state. Use /retry to continue.[/]" if ui.console else "Session saved with error state.")
        if workspace_manager:
            workspace_manager.close()
        sys.exit(1)

    finally:
        # Ensure workspace is saved and connection closed
        if workspace_manager:
            try:
                workspace_manager.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
