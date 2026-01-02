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
import readline
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


# ===== Tab Completion for Commands =====
class CommandCompleter:
    """Tab completion for slash commands."""

    def __init__(self, commands: List[str] = None):
        self.commands = commands or []
        self.matches = []

    def set_commands(self, commands: List[str]):
        """Update available commands for current context."""
        self.commands = commands

    def complete(self, text: str, state: int):
        """Readline completer function."""
        if state == 0:
            # First call - build match list
            if text.startswith('/'):
                # Match against commands
                self.matches = [c for c in self.commands if c.startswith(text)]
            elif text == '':
                # Show all commands on empty tab
                self.matches = self.commands.copy()
            else:
                self.matches = []

        # Return match or None
        if state < len(self.matches):
            return self.matches[state]
        return None


# Global completer instance
_completer = CommandCompleter()

def setup_tab_completion(commands: List[str]):
    """Setup readline tab completion with given commands."""
    _completer.set_commands(commands)
    readline.set_completer(_completer.complete)
    readline.set_completer_delims(' \t\n')
    readline.parse_and_bind('tab: complete')

from .commands import SessionState, CommandHandler, create_command_handler
from .dashboard import ProgressTimeline, DiffView, SafetyDisplay
from .workspace import (
    WorkspaceManager, Workspace, Session,
    SessionState as WorkspaceSessionState,
    SessionPhase, SessionStateMachine, InvalidStateTransition
)
from .ui import SlashCommandHandler, CommandResult, StatusLineManager
from .config import Config

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

Config File:
    pg_diagnose automatically loads config.toml from:
    1. ./pg_diagnose/config.toml (project root)
    2. ./config.toml (current directory)
    3. ~/.pg_diagnose/config.toml
    4. ~/.config/pg_diagnose/config.toml
        """,
    )

    # Config file
    parser.add_argument(
        "-c", "--config",
        help="Path to TOML config file (auto-detected if not specified)"
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

    def prompt(self, message: str, allow_commands: bool = True, tab_commands: List[str] = None, default: str = None) -> str:
        """
        Get user input with optional slash command support and tab completion.

        If input starts with '/', it's processed as a command and
        this method prompts again for actual input.

        Unrecognized commands are returned to the caller for handling.

        Args:
            message: Prompt message to display
            allow_commands: Whether to process slash commands
            tab_commands: List of commands for tab completion (e.g., ['/go', '/done', '/custom'])
            default: Default command to return if user presses Enter (shown in prompt)
        """
        # Setup tab completion if commands provided
        if tab_commands:
            setup_tab_completion(tab_commands)
        else:
            # Default commands for general prompts
            setup_tab_completion(['/help', '/status', '/quit', '/pause'])

        # Build prompt message with default hint
        if default:
            display_message = f"[{default}]: "
        else:
            display_message = message

        while True:
            try:
                user_input = input(display_message).strip()
            except EOFError:
                return '/quit'

            # Return default if user just presses Enter
            if not user_input and default:
                return default

            # Check for slash commands
            if allow_commands and user_input.startswith('/'):
                # First try workspace slash commands
                if self.slash_handler and self.slash_handler.is_command(user_input):
                    result, msg = self.slash_handler.execute(user_input)
                    self.print(msg)

                    if result == CommandResult.EXIT:
                        sys.exit(0)

                    continue  # Prompt again after command

                # Fall back to legacy command handler (only if it recognizes the command)
                if self.command_handler and self.command_handler.has_command(user_input):
                    self.command_handler.parse_and_execute(user_input)
                    continue  # Prompt again after command

                # Unrecognized command - return to caller for context-specific handling

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
            cmd = self.prompt("> ", default="/go").strip().lower()

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
            cmd = self.prompt("> ", default="/run").strip().lower()

            if cmd in ['/run', '/r', 'y', 'yes', '']:
                return True

            if cmd in ['/quit', '/q']:
                sys.exit(0)

            if cmd in ['/back', '/b']:
                return 'BACK'

            if cmd in ['/skip', '/s']:
                return 'SKIP'

            if cmd in ['n', 'no']:
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
                table = Table(box=box.ROUNDED, show_header=True, expand=True, width=120)
                table.add_column("#", style="cyan", width=3)
                table.add_column("Name", style="green", width=30)
                table.add_column("Category", style="yellow", width=10)
                table.add_column("SQL Commands", style="cyan", width=65, overflow="fold")
                table.add_column("Restart", style="red", width=7)

                for i, chunk in enumerate(config.tuning_chunks, 1):
                    restart = "[red]Yes[/]" if chunk.requires_restart else "[green]No[/]"
                    # Extract ALTER SYSTEM commands
                    sql_cmds = [cmd for cmd in chunk.apply_commands if 'ALTER SYSTEM SET' in cmd.upper()]
                    sql_text = "\n".join(sql_cmds) if sql_cmds else "-"
                    table.add_row(str(i), chunk.name, chunk.category, sql_text, restart)

                self.console.print(table)

                # Show descriptions below table
                for chunk in config.tuning_chunks:
                    if chunk.description:
                        self.console.print(f"\n  [bold]{chunk.name}:[/] [dim]{chunk.description}[/]")

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

        # Key Observations (issues/areas needing attention)
        if response.key_observations:
            self.console.print()
            self.console.rule("[bold yellow]Key Observations[/]")
            for obs in response.key_observations:
                self.console.print(f"  [yellow]•[/] {obs}")

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

    # Get PostgreSQL version first (needed for system context)
    runtime_scanner_temp = RuntimeScanner(conn, None)
    pg_version, pg_version_full = runtime_scanner_temp.get_version()

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
    system_context = system_scanner.scan(pg_version=pg_version, pg_version_full=pg_version_full)

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


def create_live_benchmark_table(
    elapsed: float,
    duration: float,
    tps: float,
    latency: float,
    cpu_user: float,
    cpu_sys: float,
    cpu_wait: float,
    mem_free_mb: float,
    disk_util: float,
    disk_await: float,
    cache_hit_ratio: float,
    buffers_backend: int,
    buffers_clean: int,
    buffers_checkpoint: int,
) -> Table:
    """Create a live benchmark metrics table."""
    from rich.table import Table
    from rich import box

    # Progress percentage
    pct = min(elapsed / duration * 100, 100) if duration > 0 else 0
    remaining = max(0, duration - elapsed)

    # Main table
    table = Table(
        title=f"[bold cyan]Benchmark[/] [dim]({elapsed:.0f}s / {duration:.0f}s)[/] [{'green' if pct >= 100 else 'yellow'}]{pct:.0f}%[/]",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("Category", style="dim", width=14)
    table.add_column("Metric", width=45)

    # TPS & Latency row
    tps_color = "green" if tps > 0 else "dim"
    lat_color = "green" if latency > 0 and latency < 20 else "yellow" if latency < 50 else "red"
    table.add_row(
        "Performance",
        f"[bold {tps_color}]TPS: {tps:,.0f}[/]  [dim]|[/]  [bold {lat_color}]Latency: {latency:.1f}ms[/]"
    )

    # CPU row
    cpu_total = cpu_user + cpu_sys
    cpu_color = "green" if cpu_total < 70 else "yellow" if cpu_total < 90 else "red"
    wait_color = "green" if cpu_wait < 10 else "yellow" if cpu_wait < 30 else "red"
    table.add_row(
        "CPU",
        f"[{cpu_color}]User: {cpu_user:.0f}%  Sys: {cpu_sys:.0f}%[/]  [dim]|[/]  [{wait_color}]Wait: {cpu_wait:.0f}%[/]"
    )

    # Disk row
    disk_color = "green" if disk_util < 70 else "yellow" if disk_util < 90 else "red"
    await_color = "green" if disk_await < 5 else "yellow" if disk_await < 20 else "red"
    table.add_row(
        "Disk I/O",
        f"[{disk_color}]Util: {disk_util:.0f}%[/]  [dim]|[/]  [{await_color}]Await: {disk_await:.1f}ms[/]"
    )

    # Cache row
    cache_color = "green" if cache_hit_ratio > 95 else "yellow" if cache_hit_ratio > 80 else "red"
    cache_miss = 100 - cache_hit_ratio
    table.add_row(
        "Buffer Cache",
        f"[{cache_color}]Hit: {cache_hit_ratio:.1f}%  Miss: {cache_miss:.1f}%[/]"
    )

    # Buffer writer row
    backend_color = "green" if buffers_backend == 0 else "yellow" if buffers_backend < 100 else "red"
    table.add_row(
        "Buffer Writers",
        f"[{backend_color}]Backend: {buffers_backend}[/]  [dim]|[/]  BGWriter: {buffers_clean}  [dim]|[/]  Checkpoint: {buffers_checkpoint}"
    )

    return table


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
    """Run benchmark with concurrent telemetry collection and live display.

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
    import re

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

    # Live metrics state
    live_metrics = {
        'tps': 0.0,
        'latency': 0.0,
        'cpu_user': 0.0,
        'cpu_sys': 0.0,
        'cpu_wait': 0.0,
        'mem_free_mb': 0.0,
        'disk_util': 0.0,
        'disk_await': 0.0,
        'cache_hit_ratio': 99.9,
        'buffers_backend': 0,
        'buffers_clean': 0,
        'buffers_checkpoint': 0,
        # Track baseline for delta
        'baseline_bgwriter': None,
    }

    # Start telemetry collection in background
    telemetry_stop = threading.Event()
    telemetry_thread = None

    def collect_telemetry():
        # Debug: log ssh_config
        import sys
        if ssh_config:
            print(f"[TELEMETRY] SSH config: host={ssh_config.get('host')}, user={ssh_config.get('user')}", file=sys.stderr, flush=True)
        else:
            print("[TELEMETRY] WARNING: ssh_config is None!", file=sys.stderr, flush=True)

        collector = TelemetryCollector(connection=conn, ssh_config=ssh_config)
        collection_errors = []
        debug_count = 0
        while not telemetry_stop.is_set():
            try:
                snapshot = collector.collect_snapshot()

                # Update live metrics
                pg_stats = snapshot.get('pg_stats', {})
                iostat = snapshot.get('iostat', {})
                vmstat = snapshot.get('vmstat', {})

                # Debug first 2 collections
                if debug_count < 2:
                    print(f"[TELEMETRY] vmstat={vmstat}", file=sys.stderr, flush=True)
                    print(f"[TELEMETRY] iostat keys={list(iostat.keys())[:5]}", file=sys.stderr, flush=True)
                    debug_count += 1

                # CPU from vmstat (nested structure: vmstat.cpu.user_pct)
                vmstat_cpu = vmstat.get('cpu', {})
                live_metrics['cpu_user'] = vmstat_cpu.get('user_pct', 0)
                live_metrics['cpu_sys'] = vmstat_cpu.get('system_pct', 0)
                live_metrics['cpu_wait'] = vmstat_cpu.get('wait_pct', 0)
                vmstat_mem = vmstat.get('memory', {})
                live_metrics['mem_free_mb'] = vmstat_mem.get('free_kb', 0) / 1024

                # Disk from iostat - returns dict of {device_name: {util_pct, await_ms, ...}}
                if isinstance(iostat, dict) and iostat:
                    # Get max utilization across all devices
                    max_util = 0
                    max_await = 0
                    for dev_name, dev_stats in iostat.items():
                        if isinstance(dev_stats, dict):
                            util = dev_stats.get('util_pct', 0)
                            await_ms = dev_stats.get('await_ms', 0)
                            if util > max_util:
                                max_util = util
                                max_await = await_ms
                    live_metrics['disk_util'] = max_util
                    live_metrics['disk_await'] = max_await

                # Cache hit from pg_stat_database
                db_stats = pg_stats.get('database', {})
                live_metrics['cache_hit_ratio'] = db_stats.get('cache_hit_ratio', 99.9)

                # BGWriter stats - track delta (cumulative counters)
                bgwriter = pg_stats.get('bgwriter', {})
                current_backend = bgwriter.get('buffers_backend', 0) or 0
                current_clean = bgwriter.get('buffers_clean', 0) or 0
                current_checkpoint = bgwriter.get('buffers_checkpoint', 0) or 0

                # Initialize or reset baseline (handles PostgreSQL restarts)
                if live_metrics['baseline_bgwriter'] is None:
                    live_metrics['baseline_bgwriter'] = bgwriter.copy()

                base = live_metrics['baseline_bgwriter']
                base_backend = base.get('buffers_backend', 0) or 0
                base_clean = base.get('buffers_clean', 0) or 0
                base_checkpoint = base.get('buffers_checkpoint', 0) or 0

                # Detect counter reset (current < baseline means PG restarted)
                if current_backend < base_backend or current_clean < base_clean or current_checkpoint < base_checkpoint:
                    # Reset baseline to current values
                    live_metrics['baseline_bgwriter'] = bgwriter.copy()
                    base_backend = current_backend
                    base_clean = current_clean
                    base_checkpoint = current_checkpoint

                live_metrics['buffers_backend'] = current_backend - base_backend
                live_metrics['buffers_clean'] = current_clean - base_clean
                live_metrics['buffers_checkpoint'] = current_checkpoint - base_checkpoint

                # Store for later
                telemetry_data['pg_stats'].append(pg_stats)
                telemetry_data['iostat'].append(iostat)
                telemetry_data['vmstat'].append(vmstat)
            except Exception as e:
                collection_errors.append(str(e))
            time.sleep(1)
        # Store errors for debugging
        telemetry_data['_errors'] = collection_errors[:5]  # Keep first 5 errors

    try:
        # Start telemetry thread (skip in mock mode - no real telemetry needed)
        if not use_mock_benchmark:
            telemetry_thread = threading.Thread(target=collect_telemetry, daemon=True)
            telemetry_thread.start()

        # Show live metrics while benchmark runs
        if ui.console and RICH_AVAILABLE:
            from rich.live import Live

            # Callback for real-time TPS updates from pgbench progress
            def on_progress(tps: float, latency: float):
                live_metrics['tps'] = tps
                live_metrics['latency'] = latency

            # Start benchmark in background with streaming callback
            bench_result = [None]
            bench_error = [None]

            def run_bench():
                try:
                    # Pass callback for real-time TPS updates
                    if not use_mock_benchmark:
                        bench_result[0] = runner.run(strategy, progress_callback=on_progress)
                    else:
                        bench_result[0] = runner.run(strategy)
                except Exception as e:
                    bench_error[0] = e

            bench_thread = threading.Thread(target=run_bench)
            bench_thread.start()

            # Live display with metrics
            start_time = time.time()

            with Live(console=ui.console, refresh_per_second=2) as live:
                while bench_thread.is_alive():
                    elapsed = time.time() - start_time

                    # Create and update table
                    table = create_live_benchmark_table(
                        elapsed=elapsed,
                        duration=duration,
                        tps=live_metrics['tps'],
                        latency=live_metrics['latency'],
                        cpu_user=live_metrics['cpu_user'],
                        cpu_sys=live_metrics['cpu_sys'],
                        cpu_wait=live_metrics['cpu_wait'],
                        mem_free_mb=live_metrics['mem_free_mb'],
                        disk_util=live_metrics['disk_util'],
                        disk_await=live_metrics['disk_await'],
                        cache_hit_ratio=live_metrics['cache_hit_ratio'],
                        buffers_backend=live_metrics['buffers_backend'],
                        buffers_clean=live_metrics['buffers_clean'],
                        buffers_checkpoint=live_metrics['buffers_checkpoint'],
                    )
                    live.update(table)
                    time.sleep(0.5)

            bench_thread.join()

            if bench_error[0]:
                raise bench_error[0]
            result = bench_result[0]

            # Show final result
            if result and hasattr(result, 'metrics') and result.metrics:
                final_tps = result.metrics.tps or 0
                final_lat = result.metrics.latency_avg_ms or 0
                ui.console.print(f"[bold green]✓ Benchmark complete[/] - TPS: {final_tps:,.0f}  Latency: {final_lat:.1f}ms")

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
    Format telemetry data into a comprehensive summary string for AI.

    Includes ALL collected metrics as time series for pattern detection:
    - OS: CPU, Memory, Disk I/O, Swap
    - PostgreSQL: Cache, BGWriter, WAL, Connections, Activity, Locks
    """
    lines = []

    pg_stats_list = telemetry.get('pg_stats', [])
    iostat_list = telemetry.get('iostat', [])
    vmstat_list = telemetry.get('vmstat', [])

    # Helper to safely get nested dict value
    def get_nested(d, *keys, default=0):
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, {})
            else:
                return default
        return d if d != {} else default

    # Helper to format series
    def fmt_series(values, fmt=".0f"):
        return "[" + ", ".join(f"{v:{fmt}}" for v in values) + "]"

    # =========================================================================
    # SECTION 1: OS/HARDWARE METRICS
    # =========================================================================
    lines.append("=" * 60)
    lines.append("OS/HARDWARE METRICS (1-second intervals)")
    lines.append("=" * 60)

    # --- CPU Metrics ---
    if vmstat_list:
        lines.append("\n[CPU]")
        cpu_user = [get_nested(s, 'cpu', 'user_pct', default=s.get('cpu_user', 0)) for s in vmstat_list]
        cpu_sys = [get_nested(s, 'cpu', 'system_pct', default=s.get('cpu_sys', 0)) for s in vmstat_list]
        cpu_wait = [get_nested(s, 'cpu', 'wait_pct', default=s.get('cpu_wait', 0)) for s in vmstat_list]
        cpu_idle = [get_nested(s, 'cpu', 'idle_pct', default=s.get('cpu_idle', 0)) for s in vmstat_list]

        lines.append(f"  User %:   {fmt_series(cpu_user)}")
        lines.append(f"  System %: {fmt_series(cpu_sys)}")
        lines.append(f"  Wait %:   {fmt_series(cpu_wait)}")
        lines.append(f"  Idle %:   {fmt_series(cpu_idle)}")
        lines.append(f"  Summary: avg_user={sum(cpu_user)/len(cpu_user):.1f}%, avg_sys={sum(cpu_sys)/len(cpu_sys):.1f}%, max_wait={max(cpu_wait):.1f}%")

    # --- Memory Metrics ---
    if vmstat_list:
        lines.append("\n[MEMORY]")
        mem_free = [get_nested(s, 'memory', 'free_kb', default=s.get('free_kb', 0)) / 1024 for s in vmstat_list]
        mem_buffer = [get_nested(s, 'memory', 'buffer_kb', default=s.get('buffer_kb', 0)) / 1024 for s in vmstat_list]
        mem_cache = [get_nested(s, 'memory', 'cache_kb', default=s.get('cache_kb', 0)) / 1024 for s in vmstat_list]
        mem_swap = [get_nested(s, 'memory', 'swapped_kb', default=s.get('swapped_kb', 0)) / 1024 for s in vmstat_list]

        lines.append(f"  Free MB:   {fmt_series(mem_free)}")
        lines.append(f"  Buffer MB: {fmt_series(mem_buffer)}")
        lines.append(f"  Cache MB:  {fmt_series(mem_cache)}")
        if max(mem_swap) > 0:
            lines.append(f"  Swap MB:   {fmt_series(mem_swap)}")

        # Swap activity
        swap_in = [get_nested(s, 'swap', 'in_per_sec', default=s.get('swap_in', 0)) for s in vmstat_list]
        swap_out = [get_nested(s, 'swap', 'out_per_sec', default=s.get('swap_out', 0)) for s in vmstat_list]
        if max(swap_in) > 0 or max(swap_out) > 0:
            lines.append(f"  Swap In/s:  {fmt_series(swap_in)}")
            lines.append(f"  Swap Out/s: {fmt_series(swap_out)}")

    # --- Disk I/O Metrics ---
    if iostat_list:
        lines.append("\n[DISK I/O]")
        # Handle both dict-of-devices and flat dict formats
        def get_io_val(s, key):
            if isinstance(s, dict):
                # Check if it's nested by device
                for dev in ['nvme0n1', 'sda', 'xvda', 'vda']:
                    if dev in s:
                        return s[dev].get(key, 0)
                # Flat dict
                return s.get(key, 0)
            return 0

        io_util = [get_io_val(s, 'util_pct') or get_io_val(s, 'util') for s in iostat_list]
        io_await = [get_io_val(s, 'await_ms') or get_io_val(s, 'await') for s in iostat_list]
        io_r_await = [get_io_val(s, 'r_await_ms') or get_io_val(s, 'r_await') for s in iostat_list]
        io_w_await = [get_io_val(s, 'w_await_ms') or get_io_val(s, 'w_await') for s in iostat_list]
        io_read = [get_io_val(s, 'r_per_sec') for s in iostat_list]
        io_write = [get_io_val(s, 'w_per_sec') for s in iostat_list]
        io_rkb = [get_io_val(s, 'rkb_per_sec') / 1024 for s in iostat_list]  # Convert to MB
        io_wkb = [get_io_val(s, 'wkb_per_sec') / 1024 for s in iostat_list]

        lines.append(f"  Util %:       {fmt_series(io_util)}")
        lines.append(f"  Await ms:     {fmt_series(io_await, '.1f')}")
        lines.append(f"  R_Await ms:   {fmt_series(io_r_await, '.1f')}")
        lines.append(f"  W_Await ms:   {fmt_series(io_w_await, '.1f')}")
        lines.append(f"  Read IOPS:    {fmt_series(io_read)}")
        lines.append(f"  Write IOPS:   {fmt_series(io_write)}")
        lines.append(f"  Read MB/s:    {fmt_series(io_rkb, '.1f')}")
        lines.append(f"  Write MB/s:   {fmt_series(io_wkb, '.1f')}")
        lines.append(f"  Summary: avg_util={sum(io_util)/len(io_util):.1f}%, max_util={max(io_util):.1f}%, avg_await={sum(io_await)/len(io_await):.1f}ms, max_await={max(io_await):.1f}ms")

    # --- Context Switches & Interrupts ---
    if vmstat_list:
        lines.append("\n[SYSTEM]")
        ctx_switches = [get_nested(s, 'system', 'context_switches', default=s.get('cs', 0)) for s in vmstat_list]
        interrupts = [get_nested(s, 'system', 'interrupts', default=s.get('in', 0)) for s in vmstat_list]
        procs_run = [get_nested(s, 'procs', 'runnable', default=s.get('r', 0)) for s in vmstat_list]
        procs_block = [get_nested(s, 'procs', 'blocked', default=s.get('b', 0)) for s in vmstat_list]

        lines.append(f"  Context Switches/s: {fmt_series(ctx_switches)}")
        lines.append(f"  Interrupts/s:       {fmt_series(interrupts)}")
        lines.append(f"  Procs Runnable:     {fmt_series(procs_run)}")
        lines.append(f"  Procs Blocked:      {fmt_series(procs_block)}")

    # =========================================================================
    # SECTION 2: POSTGRESQL METRICS
    # =========================================================================
    lines.append("\n" + "=" * 60)
    lines.append("POSTGRESQL METRICS (1-second intervals)")
    lines.append("=" * 60)

    if pg_stats_list:
        # --- Buffer Cache ---
        lines.append("\n[BUFFER CACHE]")
        cache_hit = [get_nested(s, 'database', 'cache_hit_ratio', default=0) for s in pg_stats_list]
        blks_hit = [get_nested(s, 'database', 'blks_hit', default=0) for s in pg_stats_list]
        blks_read = [get_nested(s, 'database', 'blks_read', default=0) for s in pg_stats_list]

        lines.append(f"  Cache Hit %:  {fmt_series(cache_hit, '.1f')}")
        lines.append(f"  Blocks Hit:   {fmt_series(blks_hit)}")
        lines.append(f"  Blocks Read:  {fmt_series(blks_read)}")
        lines.append(f"  Summary: avg_hit={sum(cache_hit)/len(cache_hit):.2f}%, min_hit={min(cache_hit):.2f}%")

        # --- BGWriter Stats (cumulative -> compute deltas) ---
        lines.append("\n[BGWRITER]")
        buf_backend = [get_nested(s, 'bgwriter', 'buffers_backend', default=0) for s in pg_stats_list]
        buf_clean = [get_nested(s, 'bgwriter', 'buffers_clean', default=0) for s in pg_stats_list]
        buf_checkpoint = [get_nested(s, 'bgwriter', 'buffers_checkpoint', default=0) for s in pg_stats_list]
        buf_alloc = [get_nested(s, 'bgwriter', 'buffers_alloc', default=0) for s in pg_stats_list]
        maxwritten = [get_nested(s, 'bgwriter', 'maxwritten_clean', default=0) for s in pg_stats_list]
        backend_fsync = [get_nested(s, 'bgwriter', 'buffers_backend_fsync', default=0) for s in pg_stats_list]

        # Compute deltas (cumulative counters)
        def to_deltas(values):
            if len(values) < 2:
                return values
            return [0] + [max(0, values[i] - values[i-1]) for i in range(1, len(values))]

        buf_backend_d = to_deltas(buf_backend)
        buf_clean_d = to_deltas(buf_clean)
        buf_checkpoint_d = to_deltas(buf_checkpoint)
        buf_alloc_d = to_deltas(buf_alloc)

        lines.append(f"  Backend Writes/s:    {fmt_series(buf_backend_d)} (BAD if >0, means shared_buffers too small)")
        lines.append(f"  BGWriter Cleans/s:   {fmt_series(buf_clean_d)}")
        lines.append(f"  Checkpoint Writes/s: {fmt_series(buf_checkpoint_d)}")
        lines.append(f"  Buffers Alloc/s:     {fmt_series(buf_alloc_d)}")
        lines.append(f"  Cumulative: backend={buf_backend[-1] - buf_backend[0]}, bgwriter={buf_clean[-1] - buf_clean[0]}, checkpoint={buf_checkpoint[-1] - buf_checkpoint[0]}")
        if max(maxwritten) > 0:
            lines.append(f"  MaxWritten (bgwriter stopped): {fmt_series(to_deltas(maxwritten))} (BAD if >0)")
        if max(backend_fsync) > 0:
            lines.append(f"  Backend Fsync: {fmt_series(to_deltas(backend_fsync))} (VERY BAD if >0)")

        # --- Checkpoint Stats ---
        lines.append("\n[CHECKPOINTS]")
        ckpt_timed = [get_nested(s, 'bgwriter', 'checkpoints_timed', default=0) for s in pg_stats_list]
        ckpt_req = [get_nested(s, 'bgwriter', 'checkpoints_req', default=0) for s in pg_stats_list]
        ckpt_write_time = [get_nested(s, 'bgwriter', 'checkpoint_write_time_ms', default=0) for s in pg_stats_list]
        ckpt_sync_time = [get_nested(s, 'bgwriter', 'checkpoint_sync_time_ms', default=0) for s in pg_stats_list]

        ckpt_req_d = to_deltas(ckpt_req)
        lines.append(f"  Requested Checkpoints: {fmt_series(ckpt_req_d)} (BAD if >0, means max_wal_size too small)")
        lines.append(f"  Cumulative: timed={ckpt_timed[-1]}, requested={ckpt_req[-1]}")
        lines.append(f"  Total Write Time: {ckpt_write_time[-1] - ckpt_write_time[0]:.0f}ms, Sync Time: {ckpt_sync_time[-1] - ckpt_sync_time[0]:.0f}ms")

        # --- WAL Stats (PG 14+) ---
        wal_data = [get_nested(s, 'wal', default=None) for s in pg_stats_list]
        if wal_data and wal_data[0]:
            lines.append("\n[WAL]")
            wal_bytes = [get_nested(s, 'wal', 'wal_bytes', default=0) for s in pg_stats_list]
            wal_buffers_full = [get_nested(s, 'wal', 'wal_buffers_full', default=0) for s in pg_stats_list]
            wal_write = [get_nested(s, 'wal', 'wal_write', default=0) for s in pg_stats_list]
            wal_sync = [get_nested(s, 'wal', 'wal_sync', default=0) for s in pg_stats_list]
            wal_fpi = [get_nested(s, 'wal', 'wal_fpi', default=0) for s in pg_stats_list]

            wal_bytes_d = to_deltas(wal_bytes)
            wal_mb_d = [b / (1024*1024) for b in wal_bytes_d]
            wal_buffers_full_d = to_deltas(wal_buffers_full)

            lines.append(f"  WAL MB/s:         {fmt_series(wal_mb_d, '.1f')}")
            lines.append(f"  WAL Buffers Full: {fmt_series(wal_buffers_full_d)} (BAD if >0, increase wal_buffers)")
            lines.append(f"  WAL Writes:       {fmt_series(to_deltas(wal_write))}")
            lines.append(f"  WAL Syncs:        {fmt_series(to_deltas(wal_sync))}")
            lines.append(f"  Full Page Images: {fmt_series(to_deltas(wal_fpi))}")
            lines.append(f"  Total WAL: {(wal_bytes[-1] - wal_bytes[0]) / (1024*1024):.1f} MB")

        # --- Connections ---
        lines.append("\n[CONNECTIONS]")
        conn_total = [get_nested(s, 'connections', 'total_connections', default=0) for s in pg_stats_list]
        conn_pct = [get_nested(s, 'connections', 'connection_pct', default=0) for s in pg_stats_list]

        lines.append(f"  Total Connections: {fmt_series(conn_total)}")
        lines.append(f"  Connection %:      {fmt_series(conn_pct, '.1f')}")

        # --- Activity ---
        lines.append("\n[ACTIVITY]")
        active = [get_nested(s, 'activity', 'active', default=0) for s in pg_stats_list]
        idle = [get_nested(s, 'activity', 'idle', default=0) for s in pg_stats_list]
        idle_txn = [get_nested(s, 'activity', 'idle_in_transaction', default=0) for s in pg_stats_list]
        wait_lock = [get_nested(s, 'activity', 'waiting_on_lock', default=0) for s in pg_stats_list]
        wait_io = [get_nested(s, 'activity', 'waiting_on_io', default=0) for s in pg_stats_list]

        lines.append(f"  Active:              {fmt_series(active)}")
        lines.append(f"  Idle:                {fmt_series(idle)}")
        lines.append(f"  Idle in Transaction: {fmt_series(idle_txn)} (BAD if high)")
        lines.append(f"  Waiting on Lock:     {fmt_series(wait_lock)} (BAD if >0)")
        lines.append(f"  Waiting on I/O:      {fmt_series(wait_io)}")

        # --- Locks ---
        lock_data = [get_nested(s, 'locks', default={}) for s in pg_stats_list]
        if lock_data and lock_data[0]:
            lines.append("\n[LOCKS]")
            lock_waiting = [get_nested(s, 'locks', 'total_waiting', default=0) for s in pg_stats_list]
            lines.append(f"  Total Waiting: {fmt_series(lock_waiting)} (BAD if >0)")

        # --- Transaction Stats ---
        lines.append("\n[TRANSACTIONS]")
        xact_commit = [get_nested(s, 'database', 'xact_commit', default=0) for s in pg_stats_list]
        xact_rollback = [get_nested(s, 'database', 'xact_rollback', default=0) for s in pg_stats_list]
        deadlocks = [get_nested(s, 'database', 'deadlocks', default=0) for s in pg_stats_list]
        temp_files = [get_nested(s, 'database', 'temp_files', default=0) for s in pg_stats_list]
        temp_bytes = [get_nested(s, 'database', 'temp_bytes', default=0) for s in pg_stats_list]

        xact_commit_d = to_deltas(xact_commit)
        xact_rollback_d = to_deltas(xact_rollback)
        deadlocks_d = to_deltas(deadlocks)
        temp_files_d = to_deltas(temp_files)

        lines.append(f"  Commits/s:   {fmt_series(xact_commit_d)}")
        lines.append(f"  Rollbacks/s: {fmt_series(xact_rollback_d)}")
        if max(deadlocks_d) > 0:
            lines.append(f"  Deadlocks:   {fmt_series(deadlocks_d)} (BAD!)")
        if max(temp_files_d) > 0:
            temp_mb = [(temp_bytes[i] - temp_bytes[i-1]) / (1024*1024) if i > 0 else 0 for i in range(len(temp_bytes))]
            lines.append(f"  Temp Files:  {fmt_series(temp_files_d)} (BAD - increase work_mem)")
            lines.append(f"  Temp MB:     {fmt_series(temp_mb, '.1f')}")

        # --- Tuple Operations ---
        lines.append("\n[TUPLE OPS]")
        tup_returned = [get_nested(s, 'database', 'tup_returned', default=0) for s in pg_stats_list]
        tup_fetched = [get_nested(s, 'database', 'tup_fetched', default=0) for s in pg_stats_list]
        tup_inserted = [get_nested(s, 'database', 'tup_inserted', default=0) for s in pg_stats_list]
        tup_updated = [get_nested(s, 'database', 'tup_updated', default=0) for s in pg_stats_list]
        tup_deleted = [get_nested(s, 'database', 'tup_deleted', default=0) for s in pg_stats_list]

        lines.append(f"  Returned/s: {fmt_series(to_deltas(tup_returned))}")
        lines.append(f"  Fetched/s:  {fmt_series(to_deltas(tup_fetched))}")
        lines.append(f"  Inserted/s: {fmt_series(to_deltas(tup_inserted))}")
        lines.append(f"  Updated/s:  {fmt_series(to_deltas(tup_updated))}")
        lines.append(f"  Deleted/s:  {fmt_series(to_deltas(tup_deleted))}")

        # --- Block I/O Timing ---
        blk_read_time = [get_nested(s, 'database', 'blk_read_time_ms', default=0) for s in pg_stats_list]
        blk_write_time = [get_nested(s, 'database', 'blk_write_time_ms', default=0) for s in pg_stats_list]
        if max(blk_read_time) > 0 or max(blk_write_time) > 0:
            lines.append("\n[BLOCK I/O TIMING]")
            lines.append(f"  Read Time ms:  {fmt_series(to_deltas(blk_read_time), '.1f')}")
            lines.append(f"  Write Time ms: {fmt_series(to_deltas(blk_write_time), '.1f')}")

    # =========================================================================
    # SECTION 3: ANOMALY DETECTION
    # =========================================================================
    lines.append("\n" + "=" * 60)
    lines.append("ANOMALY DETECTION")
    lines.append("=" * 60)
    anomalies = []

    # Detect IO spikes
    if iostat_list:
        io_util = [get_io_val(s, 'util_pct') or get_io_val(s, 'util') for s in iostat_list]
        io_await = [get_io_val(s, 'await_ms') or get_io_val(s, 'await') for s in iostat_list]
        avg_util = sum(io_util) / len(io_util) if io_util else 0
        avg_await = sum(io_await) / len(io_await) if io_await else 0

        for i, u in enumerate(io_util):
            if u > 95:
                anomalies.append(f"CRITICAL: Disk saturated at T+{i}s: {u:.0f}% util")
            elif u > avg_util * 1.5 and u > 80:
                anomalies.append(f"IO spike at T+{i}s: {u:.0f}% (avg: {avg_util:.0f}%)")

        for i, a in enumerate(io_await):
            if a > 50:
                anomalies.append(f"CRITICAL: High IO latency at T+{i}s: {a:.1f}ms")
            elif a > avg_await * 2 and a > 10:
                anomalies.append(f"IO latency spike at T+{i}s: {a:.1f}ms (avg: {avg_await:.1f}ms)")

    # Detect CPU issues
    if vmstat_list:
        cpu_wait = [get_nested(s, 'cpu', 'wait_pct', default=s.get('cpu_wait', 0)) for s in vmstat_list]
        for i, w in enumerate(cpu_wait):
            if w > 30:
                anomalies.append(f"CRITICAL: CPU I/O wait at T+{i}s: {w:.0f}%")
            elif w > 15:
                anomalies.append(f"High CPU wait at T+{i}s: {w:.0f}%")

    # Detect PG issues
    if pg_stats_list:
        cache_hit = [get_nested(s, 'database', 'cache_hit_ratio', default=100) for s in pg_stats_list]
        for i, h in enumerate(cache_hit):
            if h < 90:
                anomalies.append(f"CRITICAL: Low cache hit at T+{i}s: {h:.1f}%")
            elif h < 95:
                anomalies.append(f"Cache hit drop at T+{i}s: {h:.1f}%")

        # Backend writes (very bad)
        buf_backend = [get_nested(s, 'bgwriter', 'buffers_backend', default=0) for s in pg_stats_list]
        buf_backend_d = to_deltas(buf_backend)
        for i, b in enumerate(buf_backend_d):
            if b > 100:
                anomalies.append(f"CRITICAL: Backend buffer writes at T+{i}s: {b} (increase shared_buffers!)")
            elif b > 0:
                anomalies.append(f"Backend writes at T+{i}s: {b}")

        # Requested checkpoints (bad)
        ckpt_req = [get_nested(s, 'bgwriter', 'checkpoints_req', default=0) for s in pg_stats_list]
        ckpt_req_d = to_deltas(ckpt_req)
        for i, c in enumerate(ckpt_req_d):
            if c > 0:
                anomalies.append(f"Forced checkpoint at T+{i}s (increase max_wal_size!)")

    if anomalies:
        for anomaly in anomalies[:20]:  # Show more anomalies
            lines.append(f"  ⚠ {anomaly}")
    else:
        lines.append("  ✓ No significant anomalies detected")

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


def verify_applied_settings(conn, changed_params: List[str], ui: 'DiagnoseUI') -> Dict[str, Any]:
    """
    Query pg_settings to verify that settings were applied correctly.

    Args:
        conn: Database connection
        changed_params: List of parameter names that were changed
        ui: UI instance

    Returns:
        Dict with verified settings and any mismatches
    """
    if not changed_params:
        return {'verified': [], 'mismatches': []}

    results = {'verified': [], 'mismatches': []}

    try:
        # Query all changed params at once
        param_list = ', '.join(f"'{p.lower()}'" for p in changed_params)
        query = f"""
            SELECT name, setting, unit, context, pending_restart
            FROM pg_settings
            WHERE name IN ({param_list})
            ORDER BY name
        """

        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

        if ui.console and RICH_AVAILABLE:
            from rich.table import Table
            from rich import box

            table = Table(
                title="[bold green]✓ Applied Settings Verification[/]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Parameter", style="cyan", width=30)
            table.add_column("Value", width=20)
            table.add_column("Unit", width=8)
            table.add_column("Status", width=15)

            for row in rows:
                name, setting, unit, context, pending = row
                unit_str = unit or ''

                # Check if pending restart
                if pending:
                    status = "[yellow]Pending Restart[/]"
                else:
                    status = "[green]Active[/]"

                # Format value with unit
                if unit_str:
                    value_str = f"{setting} {unit_str}"
                else:
                    value_str = setting

                table.add_row(name, value_str, unit_str, status)
                results['verified'].append({
                    'name': name,
                    'value': setting,
                    'unit': unit,
                    'pending_restart': pending,
                })

            ui.console.print()
            ui.console.print(table)

        else:
            ui.print("\n=== Applied Settings Verification ===")
            for row in rows:
                name, setting, unit, context, pending = row
                unit_str = unit or ''
                status = "(Pending Restart)" if pending else "(Active)"
                ui.print(f"  {name}: {setting} {unit_str} {status}")

    except Exception as e:
        ui.print(f"[yellow]Warning: Could not verify settings: {e}[/]" if ui.console else f"Warning: Could not verify settings: {e}")

    return results


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


def capture_config_snapshot(conn, applied_changes: List[Dict]) -> Dict[str, str]:
    """
    Capture current PostgreSQL config values for params that were modified.

    Returns dict of {param_name: current_value} for rollback purposes.
    """
    snapshot = {}

    # Extract param names from applied changes
    params_to_capture = set()
    for change in applied_changes:
        for cmd in change.get('pg_configs', []):
            # Parse ALTER SYSTEM SET param = value
            if 'ALTER SYSTEM SET' in cmd.upper():
                # Extract param name (e.g., "shared_buffers" from "ALTER SYSTEM SET shared_buffers = '8GB'")
                parts = cmd.split('=')[0].upper().replace('ALTER SYSTEM SET', '').strip()
                param_name = parts.lower()
                params_to_capture.add(param_name)

    if not params_to_capture:
        return snapshot

    # Query current values from pg_settings
    try:
        with conn.cursor() as cur:
            placeholders = ','.join(['%s'] * len(params_to_capture))
            cur.execute(f"""
                SELECT name, setting
                FROM pg_settings
                WHERE name IN ({placeholders})
            """, tuple(params_to_capture))

            for row in cur.fetchall():
                snapshot[row[0]] = row[1]
    except Exception:
        pass  # Non-critical, return partial snapshot

    return snapshot


def rollback_to_snapshot(
    conn,
    target_snapshot: Dict[str, str],
    current_changes: List[Dict],
    ui: 'DiagnoseUI',
    ssh_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Rollback PostgreSQL config to a previous snapshot state.

    Strategy:
    1. For each param in current_changes not in target_snapshot: ALTER SYSTEM RESET
    2. For each param in target_snapshot: ALTER SYSTEM SET to snapshot value

    Returns dict with:
      - success: True if rollback completed
      - reset_params: List of params that were reset
      - set_params: List of params that were set to snapshot values
      - restart_required: True if any restart-requiring param was changed
    """
    reset_params = []
    set_params = []
    restart_required = False

    # Params requiring restart if changed
    restart_params = {'shared_buffers', 'max_connections', 'wal_buffers',
                      'max_worker_processes', 'max_parallel_workers'}

    # Get params from current changes that might need rollback
    current_params = set()
    for change in current_changes:
        for cmd in change.get('pg_configs', []):
            if 'ALTER SYSTEM SET' in cmd.upper():
                parts = cmd.split('=')[0].upper().replace('ALTER SYSTEM SET', '').strip()
                param_name = parts.lower()
                current_params.add(param_name)

    ui.spinner_start("Rolling back configuration...")

    conn.rollback()
    old_autocommit = conn.autocommit
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            # Reset params not in target snapshot
            for param in current_params:
                if param not in target_snapshot:
                    try:
                        cur.execute(f"ALTER SYSTEM RESET {param}")
                        reset_params.append(param)
                        if param in restart_params:
                            restart_required = True
                    except Exception as e:
                        ui.print(f"[yellow]Warning: Could not reset {param}: {e}[/]" if ui.console else f"Warning: Could not reset {param}")

            # Set params to snapshot values
            for param, value in target_snapshot.items():
                try:
                    # Handle special cases for value formatting
                    if value.isdigit():
                        cur.execute(f"ALTER SYSTEM SET {param} = {value}")
                    else:
                        cur.execute(f"ALTER SYSTEM SET {param} = '{value}'")
                    set_params.append(param)
                    if param in restart_params:
                        restart_required = True
                except Exception as e:
                    ui.print(f"[yellow]Warning: Could not set {param}={value}: {e}[/]" if ui.console else f"Warning: Could not set {param}={value}")

            # Reload config
            cur.execute("SELECT pg_reload_conf()")

        ui.spinner_stop("Configuration rolled back")

    finally:
        conn.autocommit = old_autocommit

    return {
        'success': True,
        'reset_params': reset_params,
        'set_params': set_params,
        'restart_required': restart_required,
    }


def display_round_summary(
    ui: 'DiagnoseUI',
    round_num: int,
    current_tps: float,
    tuning_history: Dict[str, Any],
    target_tps: float,
    target_hit: bool,
):
    """
    Display summary at end of each round with TPS milestone chart.
    """
    ui.print()
    if RICH_AVAILABLE and ui.console:
        ui.console.rule(f"[bold cyan]Round {round_num} Summary[/]")
    else:
        ui.print("=" * 50)
        ui.print(f"Round {round_num} Summary")
        ui.print("=" * 50)

    # TPS comparison
    baseline_tps = tuning_history['baseline_tps']
    best_tps = tuning_history['best_tps']
    improvement_from_baseline = ((current_tps - baseline_tps) / baseline_tps * 100) if baseline_tps > 0 else 0

    ui.print()
    if ui.console:
        if target_hit:
            ui.console.print(f"[bold green]✓ Target Achieved![/] Current: {current_tps:,.0f} TPS | Target: {target_tps:,.0f} TPS")
        else:
            ui.console.print(f"[yellow]○ Target Not Hit[/] Current: {current_tps:,.0f} TPS | Target: {target_tps:,.0f} TPS")
        ui.console.print(f"  Baseline: {baseline_tps:,.0f} TPS | Best: {best_tps:,.0f} TPS | Improvement: {improvement_from_baseline:+.1f}%")
    else:
        status = "✓ Target Achieved!" if target_hit else "○ Target Not Hit"
        ui.print(f"{status} Current: {current_tps:.0f} TPS | Target: {target_tps:.0f} TPS")
        ui.print(f"  Baseline: {baseline_tps:.0f} TPS | Best: {best_tps:.0f} TPS | Improvement: {improvement_from_baseline:+.1f}%")

    # Display TPS milestone chart
    if RICH_AVAILABLE and tuning_history.get('tps_history'):
        timeline = ProgressTimeline(ui.console)
        timeline.display(
            tuning_history['tps_history'],
            target_tps,
            round_num,
            target_achieved=target_hit
        )


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

    # ==================== Load Config File ====================
    config = None
    config_source = None
    try:
        config = Config.load(args.config)
        if config._config_file:
            config_source = str(config._config_file)
    except FileNotFoundError as e:
        # Only error if explicit config was requested
        if args.config:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Warning: Failed to load config: {e}", file=sys.stderr)

    # Apply config values as defaults for missing CLI arguments
    if config:
        # Database settings
        if not args.host and config.database.host:
            args.host = config.database.host
        if args.port == 5432 and config.database.port != 5432:
            args.port = config.database.port
        if args.user == "postgres" and config.database.user != "postgres":
            args.user = config.database.user
        if not args.password and config.database.password:
            args.password = config.database.password
        if not args.database and config.database.name != "postgres":
            args.database = config.database.name

        # SSH settings
        if not args.ssh_host and config.ssh.host:
            args.ssh_host = f"{config.ssh.user}@{config.ssh.host}"
        if args.ssh_port == 22 and config.ssh.port != 22:
            args.ssh_port = config.ssh.port

        # Tuning settings
        if args.max_rounds == 5 and config.tuning.max_rounds != 5:
            args.max_rounds = config.tuning.max_rounds

        # AI API key - set environment variable if in config
        if config.ai.api_key and not os.environ.get('GEMINI_API_KEY'):
            os.environ['GEMINI_API_KEY'] = config.ai.api_key

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

    # Show config file status
    if config_source:
        auto_detected = " (auto-detected)" if not args.config else ""
        ui.print(f"[dim]Config: {config_source}{auto_detected}[/]" if RICH_AVAILABLE else f"Config: {config_source}{auto_detected}")

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
        ui.print("Provide via CLI: pg_diagnose -H hostname")
        ui.print("Or set [database].host in config.toml")
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
            continue_choice = ui.prompt("Continue? ", default="y").strip().lower()
            if continue_choice in ['n', 'no']:
                ui.print("Session remains paused.")
                return

        # Step 3: Discovery (skip if resuming - already done above)
        if not resumed_session:
            ui.print()
            ui.spinner_start("Analyzing system and schema...")
            context_packet = run_discovery(conn, ssh_config)
            ui.spinner_stop("Analysis complete")

        # Step 4: First Sight Analysis (skip only if resuming to tuning loop)
        # Note: skip_to_baseline still needs first_sight for get_round1_config
        first_sight = None
        if not skip_to_tuning_loop:
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

            # ===== SKIP BASELINE IF RESUMING WITH SAVED BASELINE =====
            if skip_to_tuning_loop or (skip_to_baseline and resume_true_baseline_tps and resume_true_baseline_tps > 0):
                # Use saved values from resume - already have baseline data
                true_baseline_tps = resume_true_baseline_tps
                target_tps = resume_target_tps
                ui.print()
                ui.print(f"[dim]Using saved baseline: {true_baseline_tps:,.0f} TPS | Target: {target_tps:,.0f} TPS[/]" if RICH_AVAILABLE else f"Using saved baseline: {true_baseline_tps:.0f} TPS | Target: {target_tps:.0f} TPS")
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
                    elif error_choice in ['/new', 'new', 'n']:
                        ui.print("Starting new session...")
                        if ws_session:
                            ws_session.fail("Baseline benchmark failed - user started new session")
                        # Reset and restart the main loop
                        ws_session = None
                        chosen_strategy_dict = None
                        context_packet = None
                        continue  # Go back to strategy selection
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
                    target_cmd = ui.prompt("> ", default="/accept").strip()

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

                    continue_choice = ui.prompt("Continue tuning? ", default="y").strip().lower()
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

                        continue_strategy = ui.prompt("Try a different strategy? ", default="n").strip().lower()
                        if continue_strategy not in ['y', 'yes']:
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

                # Step 12: Display benchmark results + Round Summary (merged - no LLM call yet)
                round1_tps = benchmark_result.metrics.tps if benchmark_result.metrics else 0
                round1_improvement = ((round1_tps - true_baseline_tps) / true_baseline_tps * 100) if true_baseline_tps > 0 else 0

                # Combined benchmark result + round summary display
                ui.display_benchmark_result(benchmark_result, chosen_strategy_dict.get('target_kpis', {}))

                # Initialize tuning history BEFORE checking target
                tuning_history = {
                    'iterations_completed': 1,  # Round 1 is complete
                    'baseline_tps': true_baseline_tps,  # TRUE baseline (before any config changes)
                    'tps_history': [true_baseline_tps, round1_tps],  # [baseline, round1]
                    'applied_changes': initial_applied_changes.copy(),  # Include round 0 changes
                    'best_tps': round1_tps,  # Best TPS after tuning
                    'best_config_snapshot': {},
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

                # Check if target hit after Round 1
                round1_target_hit = target_tps > 0 and round1_tps >= target_tps
                if round1_target_hit:
                    target_ever_achieved = True
                    last_successful_snapshot = capture_config_snapshot(conn, tuning_history['applied_changes'])
                    last_successful_round = 1
                    last_successful_changes = tuning_history['applied_changes'].copy()
                    ui.print()
                    ui.print(f"[bold green]🎉 Target TPS {target_tps:.0f} achieved in Round 1![/]" if ui.console else f"Target TPS {target_tps:.0f} achieved!")

                # ===== ROUND 1 SUMMARY (displayed immediately, no LLM call) =====
                display_round_summary(ui, 1, round1_tps, tuning_history, target_tps, round1_target_hit)

                # ===== ROUND 1 END PROMPT: Continue or Stop (LLM call deferred to /go) =====
                ui.print()
                if ui.console:
                    ui.console.print("[bold cyan]Round 1 Complete - What's Next?[/]")
                    ui.console.print()
                    ui.console.print("  [green]/go[/]      Get AI analysis & continue to Round 2")
                    ui.console.print("  [yellow]/done[/]    End session (keep current config)")
                    ui.console.print("  [dim]/custom[/]   Add instructions for AI")
                    ui.console.print("  [dim]/pause[/]    Pause and return later")
                else:
                    ui.print("Round 1 Complete - What's Next?")
                    ui.print("  /go    - Get AI analysis & continue")
                    ui.print("  /done  - End session")
                    ui.print("  /custom - Add AI instructions")
                    ui.print("  /pause - Pause session")

                r1_human_feedback = None
                r1_action = None
                while True:
                    r1_cmd = ui.prompt("> ", default="/go").strip().lower()

                    if r1_cmd in ['/go', '/g', '']:
                        r1_action = 'continue'
                        break

                    if r1_cmd in ['/done', '/d']:
                        ui.print("Ending session with Round 1 results...")
                        r1_action = 'done'
                        break

                    if r1_cmd in ['/stop', '/pause']:
                        ui.print()
                        ui.print("[yellow]Pausing session...[/]" if ui.console else "Pausing session...")
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused. Returning to session list...")
                        raise StopIteration("SESSION")

                    if r1_cmd in ['/quit', '/q']:
                        ui.print()
                        ui.print("[yellow]Pausing session...[/]" if ui.console else "Pausing session...")
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused. Returning to session list...")
                        raise StopIteration("SESSION")

                    if r1_cmd.startswith('/custom'):
                        inline_text = r1_cmd[7:].strip()
                        if inline_text:
                            r1_human_feedback = {'message': inline_text, 'round': 1}
                            ui.print(f"[green]Custom instructions noted for Round 2[/]" if ui.console else f"Instructions noted")
                            r1_action = 'continue'
                            break
                        if ui.console:
                            ui.console.print("[dim]Enter your custom instructions:[/]")
                        else:
                            print("Enter your custom instructions:")
                        custom_text = ui.prompt("Instructions: ", allow_commands=False).strip()
                        if custom_text:
                            r1_human_feedback = {'message': custom_text, 'round': 1}
                            ui.print(f"[green]Custom instructions noted for Round 2[/]" if ui.console else f"Instructions noted")
                        r1_action = 'continue'
                        break

                    ui.print(f"[yellow]Unknown command: {r1_cmd}. Type /go to continue or /done to end.[/]" if ui.console else f"Unknown command: {r1_cmd}")

                # Handle user choice
                if r1_action == 'done':
                    # End session with current results
                    target_hit_final = round1_target_hit
                    display_final_summary(ui, tuning_history, target_tps, target_hit_final)

                    if ws_session:
                        if round1_target_hit:
                            conclusion = f"Target achieved in Round 1! Best TPS: {round1_tps:.0f}"
                            ws_session.archive(conclusion)
                        else:
                            conclusion = f"Session ended by user after Round 1. Best TPS: {round1_tps:.0f}"
                            ws_session.archive(conclusion)
                        ui.show_status_line()

                    # Session save
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

                    # Session end options
                    ui.print()
                    if ui.console:
                        ui.console.print("[bold cyan]Session Complete[/]")
                        ui.console.print()
                        ui.console.print("  [green]/new[/]      Start new session with different strategy")
                        ui.console.print("  [yellow]/back[/]     Return to session list")
                        ui.console.print("  [dim]/quit[/]     Exit tool")
                    else:
                        ui.print("Session Complete")
                        ui.print("  /new  - New session")
                        ui.print("  /back - Session list")
                        ui.print("  /quit - Exit")

                    end_cmd = ui.prompt("> ", default="/back").strip().lower()

                    if end_cmd in ['/new', '/n', 'y', 'yes']:
                        # Get new strategies from AI
                        previous_session_summary = {
                            'baseline_tps': tuning_history['baseline_tps'],
                            'best_tps': tuning_history['best_tps'],
                            'improvement_pct': round1_improvement,
                            'rounds_completed': 1,
                            'target_tps': target_tps,
                            'target_achieved': round1_target_hit,
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
                        ui.spinner_start("AI generating new strategies...")

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

                        ws_session = None  # Reset for new session
                        continue  # Go to strategy selection

                    elif end_cmd in ['/back', '/b']:
                        raise StopIteration("SESSION")
                    else:
                        # Default: quit
                        ui.print()
                        if ui.console:
                            ui.console.print("[bold green]Thank you for using PostgreSQL Diagnostic Tool![/]")
                        else:
                            ui.print("Thank you for using PostgreSQL Diagnostic Tool!")
                        break

                # User chose to continue - NOW call LLM for analysis (deferred from earlier)
                ui.print()
                ui.spinner_start("AI analyzing results for Round 2...")

                telemetry_summary = format_telemetry_summary(telemetry)
                analysis = agent.analyze_results(
                    strategy=strategy,
                    result=benchmark_result,
                    telemetry_summary=telemetry_summary,
                    current_config=db_config,
                    target_kpis=chosen_strategy_dict.get('target_kpis', {}),
                    human_feedback=r1_human_feedback,
                )
                ui.spinner_stop("Analysis ready")

                # Display AI analysis
                ui.display_ai_analysis(analysis)

            # ===== TUNING LOOP (accessible from both resume and normal paths) =====
            # Initialize tuning loop tracking variables
            # Note: No auto-stop - user controls when to stop
            last_successful_snapshot = {}  # Config snapshot when target was last hit
            last_successful_round = 0  # Round number when target was last hit
            last_successful_changes = []  # Changes applied up to successful round
            target_ever_achieved = False  # Whether target was ever hit in this session

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
                    cmd = ui.prompt("> ", default="/go").strip().lower()

                    if cmd in ['/go', '/g', '']:
                        break  # Continue without custom input

                    if cmd in ['/stop', '/pause']:
                        ui.print()
                        if ui.console:
                            ui.console.print("[yellow]Pausing session...[/]")
                        else:
                            ui.print("Pausing session...")
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused at Round {round_num}. Use /resume to continue later.")
                        raise StopIteration("SESSION")  # Return to session list

                    if cmd in ['/quit', '/q']:
                        ui.print()
                        if ui.console:
                            ui.console.print("[yellow]Pausing session...[/]")
                        else:
                            ui.print("Pausing session...")
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused. Returning to session list...")
                        raise StopIteration("SESSION")  # Return to session list

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

                # Handle /stop and /done exits (these are now raised as StopIteration above)
                # This check is kept for safety but shouldn't be reached
                if cmd in ['/stop', '/pause', '/quit', '/q']:
                    raise StopIteration("SESSION")  # Exit tuning loop to session list

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
                    apply_cmd = ui.prompt("> ", default="/apply").strip().lower()

                    if apply_cmd in ['/apply', '/a', 'y', 'yes', '']:
                        apply_action = 'apply'
                        break

                    if apply_cmd in ['/stop', '/pause']:
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused at Round {round_num}. Returning to session list...")
                        raise StopIteration("SESSION")

                    if apply_cmd in ['/quit', '/q']:
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused. Returning to session list...")
                        raise StopIteration("SESSION")

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

                    if apply_cmd in ['n', 'no']:
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

                                # Verify applied settings after restart
                                # Extract parameter names from all ALTER SYSTEM commands
                                changed_params = []
                                for chunk in applied_this_round:
                                    for cmd in chunk.apply_commands:
                                        if 'ALTER SYSTEM SET' in cmd.upper():
                                            # Extract param: "ALTER SYSTEM SET param_name = ..."
                                            import re
                                            match = re.search(r'ALTER\s+SYSTEM\s+SET\s+(\w+)', cmd, re.IGNORECASE)
                                            if match:
                                                changed_params.append(match.group(1))

                                if changed_params:
                                    verify_applied_settings(conn, changed_params, ui)

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
                        ui.print("Returning to session list...")
                        raise StopIteration("SESSION")
                    elif error_action in ['/quit', '/q']:
                        if ws_session:
                            ws_session.set_error(
                                error_type="benchmark_failed",
                                message=f"Round {round_num} benchmark returned 0 TPS",
                                recoverable=True
                            )
                        ui.print("Returning to session list...")
                        raise StopIteration("SESSION")

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

                # Check if target hit
                target_hit = False
                if target_tps > 0 and current_tps >= target_tps:
                    target_hit = True
                    target_ever_achieved = True

                    # Save successful snapshot for potential rollback
                    last_successful_snapshot = capture_config_snapshot(conn, tuning_history['applied_changes'])
                    last_successful_round = round_num
                    last_successful_changes = tuning_history['applied_changes'].copy()

                    ui.print(f"[bold green]🎉 Target TPS {target_tps:.0f} achieved! (current: {current_tps:.0f})[/]" if ui.console else f"Target TPS {target_tps:.0f} achieved!")

                # ===== AUTO-ROLLBACK: If target was achieved before but now regressed =====
                if target_ever_achieved and not target_hit and last_successful_round > 0:
                    ui.print()
                    if ui.console:
                        ui.console.print(f"[bold red]⚠ REGRESSION DETECTED![/] TPS dropped from {tuning_history['best_tps']:.0f} to {current_tps:.0f}")
                        ui.console.print(f"[yellow]Auto-rolling back to Round {last_successful_round} configuration (when target was hit)...[/]")
                    else:
                        ui.print(f"REGRESSION DETECTED! TPS dropped from {tuning_history['best_tps']:.0f} to {current_tps:.0f}")
                        ui.print(f"Auto-rolling back to Round {last_successful_round}...")

                    # Perform auto-rollback
                    rollback_result = rollback_to_snapshot(
                        conn, last_successful_snapshot,
                        tuning_history['applied_changes'],
                        ui, ssh_config
                    )

                    if rollback_result['success']:
                        # Update tuning_history to reflect rollback
                        tuning_history['applied_changes'] = last_successful_changes.copy()

                        if rollback_result['restart_required']:
                            ui.print("[yellow]Restart required for rollback...[/]" if ui.console else "Restart required...")
                            restart_success = restart_postgresql_with_retry(ssh_config, ui)
                            if restart_success:
                                conn.close()
                                conn = reconnect_with_retry(
                                    args.host, args.port, args.user, password, database,
                                    ui, max_retries=5, timeout_sec=5
                                )
                                session_state.conn = conn

                        ui.print(f"[green]✓ Configuration rolled back to Round {last_successful_round}[/]" if ui.console else f"Rolled back to Round {last_successful_round}")
                    else:
                        ui.print("[red]Auto-rollback failed - manual intervention may be needed[/]" if ui.console else "Rollback failed")

                # ===== ROUND SUMMARY WITH TPS CHART =====
                display_round_summary(ui, round_num, current_tps, tuning_history, target_tps, target_hit)

                # ===== ROUND-END PROMPT: Continue, Stop, or Rollback =====
                ui.print()
                if ui.console:
                    ui.console.print("[bold cyan]Round Complete - What's Next?[/]")
                    ui.console.print()
                    ui.console.print("  [green]/go[/]      Continue to next round")
                    ui.console.print("  [yellow]/done[/]    End session (keep current config)")
                    if target_ever_achieved and not target_hit:
                        ui.console.print(f"  [red]/rollback[/] Revert to Round {last_successful_round} config (when target was hit)")
                    ui.console.print("  [dim]/custom[/]   Add instructions for AI")
                    ui.console.print("  [dim]/pause[/]    Pause and return later")
                else:
                    ui.print("Round Complete - What's Next?")
                    ui.print("  /go      - Continue to next round")
                    ui.print("  /done    - End session")
                    if target_ever_achieved and not target_hit:
                        ui.print(f"  /rollback - Revert to Round {last_successful_round} config")
                    ui.print("  /custom  - Add AI instructions")
                    ui.print("  /pause   - Pause session")

                human_feedback = None
                next_action = None
                while True:
                    next_cmd = ui.prompt("> ", default="/go").strip().lower()

                    if next_cmd in ['/go', '/g', '']:
                        next_action = 'continue'
                        break

                    if next_cmd in ['/done', '/d']:
                        ui.print("Ending session with current results...")
                        next_action = 'done'
                        break

                    if next_cmd in ['/rollback', '/rb'] and target_ever_achieved:
                        ui.print()
                        ui.print(f"[yellow]Rolling back to Round {last_successful_round} configuration...[/]" if ui.console else f"Rolling back to Round {last_successful_round}...")

                        # Perform rollback
                        rollback_result = rollback_to_snapshot(
                            conn, last_successful_snapshot,
                            tuning_history['applied_changes'],
                            ui, ssh_config
                        )

                        if rollback_result['success']:
                            # Update tuning_history to reflect rollback
                            tuning_history['applied_changes'] = last_successful_changes.copy()

                            if rollback_result['restart_required']:
                                ui.print("[yellow]Restart required for some parameters...[/]" if ui.console else "Restart required...")
                                restart_success = restart_postgresql_with_retry(ssh_config, ui)
                                if restart_success:
                                    # Reconnect
                                    conn.close()
                                    conn = reconnect_with_retry(
                                        args.host, args.port, args.user, password, database,
                                        ui, max_retries=5, timeout_sec=5
                                    )
                                    session_state.conn = conn

                            ui.print(f"[green]Configuration rolled back to Round {last_successful_round}[/]" if ui.console else f"Rolled back to Round {last_successful_round}")
                            ui.print("[dim]Run another benchmark to verify.[/]" if ui.console else "Run another benchmark to verify.")
                        else:
                            ui.print("[red]Rollback failed - continuing with current config[/]" if ui.console else "Rollback failed")

                        next_action = 'continue'
                        break

                    if next_cmd in ['/stop', '/pause']:
                        ui.print()
                        ui.print("[yellow]Pausing session...[/]" if ui.console else "Pausing session...")
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused. Returning to session list...")
                        raise StopIteration("SESSION")

                    if next_cmd in ['/quit', '/q']:
                        ui.print()
                        ui.print("[yellow]Pausing session...[/]" if ui.console else "Pausing session...")
                        if ws_session:
                            ws_session.pause()
                        ui.print(f"Session paused. Returning to session list...")
                        raise StopIteration("SESSION")

                    if next_cmd in ['/status', '/s']:
                        ui.print(f"Round: {round_num} | Best TPS: {tuning_history['best_tps']:.0f} | Target: {target_tps:.0f}")
                        if target_ever_achieved:
                            ui.print(f"  Last successful: Round {last_successful_round}")
                        continue

                    if next_cmd in ['/history', '/h']:
                        ui.print("Tuning History:")
                        for change in tuning_history.get('applied_changes', []):
                            ui.print(f"  R{change['round']}: {change['name']}")
                        continue

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

                    ui.print(f"[yellow]Unknown command: {next_cmd}. Type /go to continue or /done to end.[/]" if ui.console else f"Unknown command: {next_cmd}")

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

            # ===================== SESSION END - LIFECYCLE NAVIGATION =====================
            ui.print()
            if ui.console:
                ui.console.print("\n[cyan]Session Complete. What would you like to do?[/]")
                ui.console.print("  [green]/new[/]       - Start new session in this workspace")
                ui.console.print("  [yellow]/back[/]      - Return to session list")
                ui.console.print("  [blue]/workspace[/] - Return to workspace selection")
                ui.console.print("  [red]/quit[/]      - Exit the tool")
            else:
                ui.print("\nSession Complete. What would you like to do?")
                ui.print("  /new       - Start new session in this workspace")
                ui.print("  /back      - Return to session list")
                ui.print("  /workspace - Return to workspace selection")
                ui.print("  /quit      - Exit the tool")

            continue_choice = ui.prompt("\nChoice [/back]: ").strip().lower()

            # Handle slash commands
            if continue_choice in ['/quit', '/q', 'quit']:
                ui.print()
                if ui.console:
                    ui.console.print("[bold green]Thank you for using PostgreSQL Diagnostic Tool![/]")
                    ui.console.print("[dim]Goodbye![/dim]")
                else:
                    ui.print("Thank you for using PostgreSQL Diagnostic Tool!")
                    ui.print("Goodbye!")
                raise SystemExit(0)  # Exit app completely

            if continue_choice in ['/workspace', '/w', 'workspace']:
                ui.print("[dim]Returning to workspace selection...[/]" if ui.console else "Returning to workspace selection...")
                raise StopIteration("WORKSPACE")  # Signal to return to workspace selection

            if continue_choice in ['/back', '/b', 'back', '']:
                ui.print("[dim]Returning to session list...[/]" if ui.console else "Returning to session list...")
                raise StopIteration("SESSION")  # Signal to return to session selection

            if continue_choice in ['/export', '/e']:
                if ws_session:
                    ui.print(f"Session data saved in workspace: {workspace.name}")
                continue

            if continue_choice not in ['y', 'yes', '/new', '/n', 'new']:
                # Default to session list
                raise StopIteration("SESSION")

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

    except StopIteration as nav_signal:
        # ===================== LIFECYCLE NAVIGATION =====================
        # Handle navigation signals from session end
        nav_target = str(nav_signal) if nav_signal.args else "SESSION"

        if workspace_manager:
            workspace_manager.close()
        if conn:
            try:
                conn.close()
            except Exception:
                pass

        if nav_target == "WORKSPACE":
            # Restart main() to go back to workspace selection
            ui.print()
            return main()
        else:
            # SESSION - restart main() to show session list for current workspace
            # For now, restart main() - the workspace will be shown first
            ui.print()
            return main()

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
