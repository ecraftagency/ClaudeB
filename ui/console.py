"""
ConsoleUI - Rich-based console interface.

Provides interactive progress display and result formatting.
"""

from typing import Optional, Dict, Any, List
from dataclasses import asdict

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.tree import Tree
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..runner.state import State


class ConsoleUI:
    """
    Rich console interface for pg_diagnose.
    """

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.console = Console() if RICH_AVAILABLE else None
        self._progress = None
        self._live = None

    def print(self, *args, **kwargs):
        """Print to console."""
        if self.quiet:
            return
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args)

    def print_header(self, title: str):
        """Print a section header."""
        if self.quiet:
            return

        if self.console:
            self.console.print()
            self.console.rule(f"[bold blue]{title}[/]")
        else:
            print(f"\n{'='*60}")
            print(f" {title}")
            print('='*60)

    def print_banner(self):
        """Print application banner."""
        if self.quiet:
            return

        banner = """
[bold cyan]PostgreSQL Diagnostic Tool[/] [dim]v2.2[/]
[dim]AI-Powered Performance Analysis & Tuning[/]
        """
        if self.console:
            self.console.print(Panel(banner.strip(), border_style="cyan"))
        else:
            print("PostgreSQL Diagnostic Tool v2.2")
            print("AI-Powered Performance Analysis & Tuning")
            print()

    def print_state_change(self, from_state: State, to_state: State, metadata: Dict = None):
        """Display state transition."""
        if self.quiet:
            return

        status_colors = {
            State.INIT: "dim",
            State.DISCOVER: "blue",
            State.STRATEGIZE: "cyan",
            State.EXECUTE: "yellow",
            State.AWAIT_USER_INPUT: "bold cyan",  # v2.3: Human-in-the-loop
            State.ANALYZE: "magenta",
            State.TUNE: "green",
            State.VERIFY: "blue",
            State.COMPLETE: "bold green",
            State.EMERGENCY_ROLLBACK: "bold red",
            State.FAILED: "bold red",
        }

        color = status_colors.get(to_state, "white")

        if self.console:
            self.console.print(
                f"[dim]{from_state.name}[/] -> [{color}]{to_state.name}[/]"
            )
        else:
            print(f"{from_state.name} -> {to_state.name}")

    def print_context(self, context_packet):
        """Display discovered context."""
        if self.quiet:
            return

        if not context_packet:
            return

        self.print_header("System Context")

        if self.console:
            # System info
            sys_ctx = context_packet.system_context
            if sys_ctx:
                table = Table(title="Hardware", show_header=False, box=None)
                table.add_column("Key", style="dim")
                table.add_column("Value")

                table.add_row("CPU", f"{sys_ctx.cpu_cores} cores ({sys_ctx.cpu_architecture})")
                table.add_row("RAM", f"{sys_ctx.ram_total_gb:.1f} GB")
                table.add_row("Storage", sys_ctx.storage_topology or "Unknown")

                self.console.print(table)

            # Schema summary
            schema_ctx = context_packet.schema_context
            if schema_ctx:
                self.console.print()
                table = Table(title="Schema", show_header=False, box=None)
                table.add_column("Key", style="dim")
                table.add_column("Value")

                tables = schema_ctx.table_statistics or {}
                indexes = schema_ctx.index_statistics or {}
                total_size_mb = sum(t.size_mb for t in tables.values())
                table.add_row("Tables", str(len(tables)))
                table.add_row("Indexes", str(len(indexes)))
                table.add_row("Total Size", f"{total_size_mb / 1024:.2f} GB")

                self.console.print(table)
        else:
            ctx = asdict(context_packet)
            print(f"CPU: {ctx.get('system_context', {}).get('cpu_cores', '?')} cores")
            print(f"RAM: {ctx.get('system_context', {}).get('ram_total_gb', '?')} GB")

    def print_strategy(self, strategy):
        """Display generated strategy."""
        if self.quiet:
            return

        self.print_header("Strategy")

        if self.console:
            # Strategy overview
            self.console.print(f"[bold]{strategy.name}[/]")
            self.console.print(f"[dim]Hypothesis:[/] {strategy.hypothesis}")

            if strategy.execution_plan:
                plan = strategy.execution_plan
                table = Table(title="Benchmark Plan", show_header=False, box=None)
                table.add_column("Key", style="dim")
                table.add_column("Value")

                table.add_row("Type", plan.benchmark_type)
                table.add_row("Scale", str(plan.scale))
                table.add_row("Clients", str(plan.clients))
                table.add_row("Duration", f"{plan.duration_seconds}s")

                self.console.print(table)

            if strategy.success_criteria:
                criteria = strategy.success_criteria
                self.console.print()
                self.console.print("[bold]Success Criteria:[/]")
                if criteria.target_tps:
                    self.console.print(f"  Target TPS: {criteria.target_tps}")
                if criteria.max_latency_p99_ms:
                    self.console.print(f"  Max P99 Latency: {criteria.max_latency_p99_ms}ms")
        else:
            print(f"Strategy: {strategy.name}")
            print(f"Hypothesis: {strategy.hypothesis}")

    def print_benchmark_result(self, result):
        """Display benchmark results."""
        if self.quiet:
            return

        self.print_header("Benchmark Results")

        if self.console:
            metrics = result.metrics

            # Main metrics
            table = Table(title="Performance Metrics", box=None)
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")

            if metrics.tps:
                table.add_row("TPS", f"[bold green]{metrics.tps:.2f}[/]")
            if metrics.latency_avg_ms:
                table.add_row("Avg Latency", f"{metrics.latency_avg_ms:.2f} ms")
            if metrics.latency_max_ms:
                table.add_row("Max Latency", f"{metrics.latency_max_ms:.2f} ms")
            if metrics.transactions:
                table.add_row("Transactions", f"{metrics.transactions:,}")

            self.console.print(table)

            # Criteria check
            if result.criteria_met:
                self.console.print()
                for criterion, met in result.criteria_met.items():
                    icon = "[green]:heavy_check_mark:[/]" if met else "[red]:x:[/]"
                    self.console.print(f"  {icon} {criterion}")
        else:
            print(f"TPS: {result.metrics.tps}")
            print(f"Latency: {result.metrics.latency_avg_ms}ms avg")

    def print_proposal(self, proposal):
        """Display tuning proposal."""
        if self.quiet:
            return

        self.print_header("Tuning Proposal")

        if self.console:
            self.console.print(f"[bold]{proposal.analysis_summary}[/]")
            self.console.print(
                f"Bottleneck: [yellow]{proposal.bottleneck_type}[/] "
                f"(confidence: {proposal.confidence:.0%})"
            )

            if proposal.tuning_chunks:
                self.console.print()
                self.console.print(f"[bold]{len(proposal.tuning_chunks)} Tuning Changes:[/]")

                for i, chunk in enumerate(proposal.tuning_chunks, 1):
                    tree = Tree(f"[bold]{i}. {chunk.name}[/]")
                    tree.add(f"[dim]Category:[/] {chunk.category}")
                    tree.add(f"[dim]Rationale:[/] {chunk.rationale}")

                    if chunk.requires_restart:
                        tree.add("[yellow]Requires restart[/]")

                    cmds = tree.add("[dim]Commands:[/]")
                    for cmd in chunk.apply_commands[:3]:  # Show first 3
                        cmds.add(f"[cyan]{cmd}[/]")

                    self.console.print(tree)
                    self.console.print()

            if proposal.expected_improvement:
                imp = proposal.expected_improvement
                self.console.print(
                    f"[green]Expected: +{imp.tps_increase_pct}% TPS, "
                    f"-{imp.latency_reduction_pct}% latency[/]"
                )
        else:
            print(f"Analysis: {proposal.analysis_summary}")
            print(f"Bottleneck: {proposal.bottleneck_type}")
            for chunk in proposal.tuning_chunks:
                print(f"  - {chunk.name}")

    def print_tuning_results(self, results: List):
        """Display tuning application results."""
        if self.quiet:
            return

        self.print_header("Tuning Results")

        if self.console:
            for result in results:
                if hasattr(result, 'success') and result.success:
                    self.console.print(
                        f"[green]:heavy_check_mark:[/] {result.chunk_id}: Applied and verified"
                    )
                else:
                    self.console.print(
                        f"[red]:x:[/] {getattr(result, 'failed_chunk_id', 'Unknown')}: Failed"
                    )
        else:
            for result in results:
                status = "OK" if getattr(result, 'success', False) else "FAILED"
                print(f"  [{status}] {getattr(result, 'chunk_id', 'Unknown')}")

    def print_summary(self, summary: Dict[str, Any]):
        """Display session summary."""
        if self.quiet:
            return

        self.print_header("Session Summary")

        if self.console:
            sm = summary.get('state_machine', {})

            table = Table(show_header=False, box=None)
            table.add_column("Key", style="dim")
            table.add_column("Value")

            table.add_row("Session ID", summary.get('session_id', 'Unknown'))
            table.add_row("Final State", sm.get('current_state', 'Unknown'))
            table.add_row("Iterations", str(sm.get('iteration', 0)))
            table.add_row("Duration", f"{sm.get('total_duration_ms', 0) / 1000:.1f}s")
            table.add_row("Output Dir", summary.get('output_dir', 'Unknown'))

            self.console.print(table)
        else:
            print(f"Session: {summary.get('session_id')}")
            print(f"State: {summary.get('state_machine', {}).get('current_state')}")

    def print_error(self, message: str, exception: Optional[Exception] = None):
        """Display error message."""
        if self.console:
            self.console.print(f"[bold red]Error:[/] {message}")
            if exception:
                self.console.print(f"[dim]{type(exception).__name__}: {exception}[/]")
        else:
            print(f"Error: {message}")
            if exception:
                print(f"  {exception}")

    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask for confirmation."""
        if self.quiet:
            return default

        if self.console:
            from rich.prompt import Confirm
            return Confirm.ask(message, default=default)
        else:
            response = input(f"{message} [y/N]: ").lower()
            return response in ('y', 'yes')

    def review_tuning_chunks(self, proposal) -> List:
        """
        Interactive review of tuning chunks.

        Returns list of approved chunks.
        """
        if self.quiet or not proposal.tuning_chunks:
            return proposal.tuning_chunks  # Return all in quiet mode

        self.print_header("Review Tuning Recommendations")
        self.console.print("[dim]Review each recommendation and approve (y) or skip (n)[/]")
        self.console.print()

        approved_chunks = []

        for i, chunk in enumerate(proposal.tuning_chunks, 1):
            # Display chunk details
            self.console.print(f"[bold cyan]━━━ Recommendation {i}/{len(proposal.tuning_chunks)} ━━━[/]")
            self.console.print()

            tree = Tree(f"[bold]{chunk.name}[/]")
            tree.add(f"[dim]Category:[/] {chunk.category}")
            tree.add(f"[dim]Rationale:[/] {chunk.rationale}")

            if chunk.risk_level:
                risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(chunk.risk_level, "white")
                tree.add(f"[dim]Risk Level:[/] [{risk_color}]{chunk.risk_level}[/]")

            if chunk.requires_restart:
                tree.add("[yellow]⚠ Requires PostgreSQL restart[/]")

            cmds = tree.add("[dim]Commands to execute:[/]")
            for cmd in chunk.apply_commands:
                cmds.add(f"[cyan]{cmd}[/]")

            if chunk.rollback_commands:
                rollback = tree.add("[dim]Rollback commands:[/]")
                for cmd in chunk.rollback_commands:
                    rollback.add(f"[dim]{cmd}[/]")

            self.console.print(tree)
            self.console.print()

            # Ask for confirmation
            from rich.prompt import Confirm
            approved = Confirm.ask(
                f"[bold]Apply '{chunk.name}'?[/]",
                default=True
            )

            if approved:
                approved_chunks.append(chunk)
                self.console.print("[green]✓ Approved[/]")
            else:
                self.console.print("[yellow]✗ Skipped[/]")

            self.console.print()

        # Summary
        self.console.print(f"[bold]Review complete:[/] {len(approved_chunks)}/{len(proposal.tuning_chunks)} recommendations approved")

        return approved_chunks

    def start_progress(self, description: str):
        """Start progress indicator."""
        if self.quiet or not self.console:
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(description, total=100)

    def update_progress(self, description: str, completed: float):
        """Update progress."""
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                description=description,
                completed=completed * 100
            )

    def stop_progress(self):
        """Stop progress indicator."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None
