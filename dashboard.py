"""
Dashboard - Live TUI dashboard for pg_diagnose.

Provides a persistent terminal UI with:
- Live metrics panel
- Session status
- Recent changes
- AI thinking indicator
- Keyboard shortcuts
"""

import time
import threading
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class DashboardState:
    """State for the dashboard."""
    # Connection
    db_name: str = ""
    db_host: str = ""

    # Strategy
    strategy_name: str = ""
    current_round: int = 0
    target_tps: float = 0

    # Live metrics
    current_tps: float = 0
    latency_ms: float = 0
    cache_hit: float = 0
    io_wait: float = 0
    cpu_usage: float = 0
    memory_pct: float = 0

    # TPS history for sparkline
    tps_history: List[float] = field(default_factory=list)

    # Recent changes
    recent_changes: List[Dict] = field(default_factory=list)

    # AI status
    ai_thinking: bool = False
    ai_message: str = ""

    # Benchmark status
    benchmark_running: bool = False
    benchmark_progress: float = 0
    benchmark_duration: int = 0

    # Errors/warnings
    warnings: List[str] = field(default_factory=list)


class Dashboard:
    """
    Live TUI dashboard for pg_diagnose.

    Usage:
        dashboard = Dashboard(conn, ssh_config)
        dashboard.start()  # Start background refresh

        # Update state as needed
        dashboard.state.current_tps = 5892
        dashboard.state.ai_thinking = True

        dashboard.stop()  # Stop when done
    """

    def __init__(self, conn, ssh_config: Optional[Dict] = None, refresh_rate: float = 1.0):
        self.conn = conn
        self.ssh_config = ssh_config
        self.refresh_rate = refresh_rate
        self.console = Console() if RICH_AVAILABLE else None
        self.state = DashboardState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._live: Optional[Live] = None
        self._action_callback: Optional[Callable] = None

    def set_action_callback(self, callback: Callable):
        """Set callback for keyboard actions."""
        self._action_callback = callback

    def start(self):
        """Start the dashboard."""
        if not RICH_AVAILABLE:
            print("Rich library required for dashboard mode")
            return

        self._running = True
        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=1 / self.refresh_rate,
            screen=True,
        )
        self._live.start()

        # Start refresh thread
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the dashboard."""
        self._running = False
        if self._live:
            self._live.stop()
        if self._thread:
            self._thread.join(timeout=2)

    def update(self):
        """Force an update of the display."""
        if self._live:
            self._live.update(self._build_layout())

    def _refresh_loop(self):
        """Background refresh loop."""
        while self._running:
            try:
                self._fetch_live_metrics()
                self.update()
            except Exception:
                pass
            time.sleep(self.refresh_rate)

    def _fetch_live_metrics(self):
        """Fetch current metrics from database."""
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                # Cache hit ratio
                cur.execute("""
                    SELECT CASE WHEN blks_hit + blks_read = 0 THEN 100
                           ELSE (blks_hit::float / (blks_hit + blks_read) * 100) END
                    FROM pg_stat_database WHERE datname = current_database()
                """)
                self.state.cache_hit = cur.fetchone()[0] or 0

        except Exception:
            pass

        # Get system metrics via SSH
        if self.ssh_config:
            try:
                cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=2",
                       f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                       "cat /proc/loadavg; vmstat 1 2 | tail -1"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        # Parse vmstat
                        vmstat = lines[-1].split()
                        if len(vmstat) >= 16:
                            self.state.io_wait = float(vmstat[15])  # wa column
                            self.state.cpu_usage = 100 - float(vmstat[14])  # 100 - idle
            except Exception:
                pass

    def _build_layout(self) -> Layout:
        """Build the dashboard layout."""
        layout = Layout()

        # Main structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Body split
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        # Left side: metrics + changes
        layout["left"].split_column(
            Layout(name="metrics", ratio=2),
            Layout(name="changes", ratio=1),
        )

        # Right side: status + AI
        layout["right"].split_column(
            Layout(name="status", ratio=1),
            Layout(name="ai", ratio=1),
        )

        # Build each panel
        layout["header"].update(self._build_header())
        layout["metrics"].update(self._build_metrics_panel())
        layout["changes"].update(self._build_changes_panel())
        layout["status"].update(self._build_status_panel())
        layout["ai"].update(self._build_ai_panel())
        layout["footer"].update(self._build_footer())

        return layout

    def _build_header(self) -> Panel:
        """Build header panel."""
        title = Text()
        title.append("pg_diagnose v2.2", style="bold cyan")
        title.append("  ")
        title.append(f"[{self.state.db_name}]", style="green")
        title.append(" @ ", style="dim")
        title.append(self.state.db_host, style="yellow")

        return Panel(title, style="blue", height=3)

    def _build_metrics_panel(self) -> Panel:
        """Build live metrics panel."""
        content = []

        # TPS with trend
        tps = self.state.current_tps
        target = self.state.target_tps
        if target > 0:
            pct = min(tps / target * 100, 100)
            bar = self._make_bar(pct, 20)
            tps_line = f"TPS: {tps:,.0f} {bar} {pct:.0f}%"
        else:
            tps_line = f"TPS: {tps:,.0f}"

        # Determine trend
        if len(self.state.tps_history) >= 2:
            prev = self.state.tps_history[-2]
            if tps > prev * 1.05:
                tps_line += " [green]▲[/]"
            elif tps < prev * 0.95:
                tps_line += " [red]▼[/]"
            else:
                tps_line += " [dim]─[/]"

        content.append(tps_line)
        content.append(f"Latency: {self.state.latency_ms:.1f}ms")
        content.append(f"Cache Hit: {self.state.cache_hit:.1f}%")
        content.append(f"IO Wait: {self._make_bar(self.state.io_wait, 15)} {self.state.io_wait:.0f}%")
        content.append(f"CPU: {self._make_bar(self.state.cpu_usage, 15)} {self.state.cpu_usage:.0f}%")
        content.append(f"Memory: {self._make_bar(self.state.memory_pct, 15)} {self.state.memory_pct:.0f}%")

        # Sparkline for TPS history
        if self.state.tps_history:
            sparkline = self._make_sparkline(self.state.tps_history[-20:])
            content.append(f"\nTPS Trend: {sparkline}")

        return Panel(
            "\n".join(content),
            title="[bold]LIVE METRICS[/]",
            border_style="green",
        )

    def _build_changes_panel(self) -> Panel:
        """Build recent changes panel."""
        lines = []

        if not self.state.recent_changes:
            lines.append("[dim]No changes applied yet[/]")
        else:
            for change in self.state.recent_changes[-5:]:
                status = change.get('status', 'applied')
                icon = {'applied': '[green]✓[/]', 'pending': '[yellow]→[/]', 'failed': '[red]✗[/]'}.get(status, '•')
                name = change.get('name', 'Unknown')
                lines.append(f"{icon} {name}")

                # Show config if available
                for cmd in change.get('pg_configs', [])[:1]:
                    # Extract just the setting
                    if '=' in cmd:
                        setting = cmd.split('SET')[-1].strip() if 'SET' in cmd else cmd
                        lines.append(f"  [dim]{setting[:40]}[/]")

        return Panel(
            "\n".join(lines),
            title="[bold]RECENT CHANGES[/]",
            border_style="yellow",
        )

    def _build_status_panel(self) -> Panel:
        """Build session status panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Strategy", self.state.strategy_name or "[dim]Not selected[/]")
        table.add_row("Round", f"{self.state.current_round}/∞")
        table.add_row("Target", f"{self.state.target_tps:,.0f} TPS" if self.state.target_tps else "[dim]N/A[/]")

        # Progress to target
        if self.state.target_tps > 0:
            progress = min(self.state.current_tps / self.state.target_tps * 100, 100)
            bar = self._make_bar(progress, 15)
            table.add_row("Progress", f"{bar} {progress:.0f}%")

        # Benchmark progress
        if self.state.benchmark_running:
            bench_bar = self._make_bar(self.state.benchmark_progress, 15)
            table.add_row("Benchmark", f"{bench_bar} {self.state.benchmark_progress:.0f}%")

        return Panel(
            table,
            title="[bold]SESSION STATUS[/]",
            border_style="blue",
        )

    def _build_ai_panel(self) -> Panel:
        """Build AI thinking panel."""
        if self.state.ai_thinking:
            content = Text()
            content.append("● ", style="bold yellow")
            content.append(self.state.ai_message or "Analyzing...", style="italic")
        else:
            content = Text()
            content.append("○ ", style="dim")
            content.append("Idle", style="dim")

        # Add any warnings
        if self.state.warnings:
            content.append("\n\n")
            for w in self.state.warnings[-3:]:
                content.append(f"⚠ {w}\n", style="yellow")

        return Panel(
            content,
            title="[bold]AI STATUS[/]",
            border_style="magenta",
        )

    def _build_footer(self) -> Panel:
        """Build footer with shortcuts."""
        shortcuts = [
            "[A]pply next",
            "[S]kip",
            "[P]ause",
            "[R]ollback",
            "[H]elp",
            "[Q]uit",
        ]

        return Panel(
            "  ".join(shortcuts),
            style="dim",
            height=3,
        )

    def _make_bar(self, pct: float, width: int) -> str:
        """Create a progress bar string."""
        pct = max(0, min(100, pct))
        filled = int(pct / 100 * width)
        empty = width - filled

        # Color based on value
        if pct >= 90:
            color = "green"
        elif pct >= 70:
            color = "yellow"
        else:
            color = "red"

        return f"[{color}]{'█' * filled}[/][dim]{'░' * empty}[/]"

    def _make_sparkline(self, values: List[float]) -> str:
        """Create a sparkline from values."""
        if not values:
            return ""

        chars = "▁▂▃▄▅▆▇█"
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val or 1

        sparkline = ""
        for v in values:
            idx = int((v - min_val) / range_val * (len(chars) - 1))
            sparkline += chars[idx]

        return f"[cyan]{sparkline}[/]"


class ProgressTimeline:
    """
    Visual timeline showing tuning progress with milestone indicators.

    Example output:
    Session Timeline:
    ════════════════════════════════════════════════════════════════════
      ●────────●────────●────────✗────────○
      │        │        │        │        │
    Baseline  R1       R2       R3      Target
     3,500   4,200    5,100    4,800    6,000
             +20%     +21%     -6%
              ✓        ✓        ✗

    Legend: ● = completed  ✓ = target hit  ✗ = target miss  ○ = pending
    """

    def __init__(self, console=None):
        self.console = console or (Console() if RICH_AVAILABLE else None)

    def display(self, tps_history: List[float], target_tps: float, current_round: int,
                target_achieved: bool = False, round_results: List[Dict] = None):
        """
        Display the progress timeline.

        Args:
            tps_history: List of TPS values [baseline, r1, r2, ...]
            target_tps: Target TPS to achieve
            current_round: Current round number
            target_achieved: Whether target was achieved in latest round
            round_results: Optional list of round metadata [{'hit': bool, 'config_applied': bool}, ...]
        """
        if not tps_history:
            return

        if self.console and RICH_AVAILABLE:
            self._display_rich(tps_history, target_tps, current_round, target_achieved, round_results)
        else:
            self._display_plain(tps_history, target_tps, current_round, target_achieved, round_results)

    def _display_rich(self, tps_history: List[float], target_tps: float, current_round: int,
                       target_achieved: bool = False, round_results: List[Dict] = None):
        """Rich display of timeline with milestone indicators.

        Shows only completed rounds + target goal. No future rounds.
        """
        points = len(tps_history)  # This is baseline + completed rounds

        # Build round_results if not provided
        if round_results is None:
            round_results = []
            for i, tps in enumerate(tps_history):
                if i == 0:
                    # Baseline - no target check
                    round_results.append({'hit': False, 'is_baseline': True})
                else:
                    hit = tps >= target_tps if target_tps > 0 else False
                    round_results.append({'hit': hit, 'is_baseline': False})

        # Column width for proper spacing - must accommodate "Baseline" (8 chars)
        COL_WIDTH = 12

        # Build labels first
        labels = ["Baseline"] + [f"R{i}" for i in range(1, points)] + ["Target"]
        total_cols = len(labels)

        # Get all dots/symbols for each column
        dots = []
        for i in range(total_cols):
            if i == 0:
                dots.append("[cyan]●[/]")
            elif i < points:
                if round_results[i].get('hit', False):
                    dots.append("[bold green]●[/]")
                else:
                    dots.append("[bold red]✗[/]")
            else:
                if target_achieved:
                    dots.append("[bold green]★[/]")
                else:
                    dots.append("[dim]○[/]")

        # Build lines with proper centering
        # Each column is COL_WIDTH chars, dot/connector should be at center
        CENTER = COL_WIDTH // 2  # Position 6 for width 12

        # Line 1: dots connected by dashes
        # Pattern: [pad][dot][dashes...][dot][dashes...][dot]
        line1_parts = []
        for i, dot in enumerate(dots):
            if i == 0:
                # First: pad to center, dot, then dashes to fill column
                line1_parts.append(" " * CENTER + dot + "─" * (COL_WIDTH - CENTER - 1))
            elif i == total_cols - 1:
                # Last: dashes from prev, dot at center
                line1_parts.append("─" * CENTER + dot + " " * (COL_WIDTH - CENTER - 1))
            else:
                # Middle: dashes from prev, dot at center, dashes to next
                line1_parts.append("─" * CENTER + dot + "─" * (COL_WIDTH - CENTER - 1))
        line1 = "".join(line1_parts)

        # Line 2: connector bars at same position as dots
        line2_parts = []
        for i in range(total_cols):
            line2_parts.append(" " * CENTER + "│" + " " * (COL_WIDTH - CENTER - 1))
        line2 = "".join(line2_parts)

        # Labels line - center each label in its column
        line3 = ""
        for label in labels:
            line3 += f"{label:^{COL_WIDTH}}"

        # TPS Values - build without markup first, then add colors
        line4 = ""
        for i, tps in enumerate(tps_history):
            tps_str = f"{tps:,.0f}"
            padding = COL_WIDTH - len(tps_str)
            left_pad = padding // 2
            right_pad = padding - left_pad
            if i == 0:
                line4 += " " * left_pad + f"[cyan]{tps_str}[/]" + " " * right_pad
            elif round_results[i].get('hit', False):
                line4 += " " * left_pad + f"[bold green]{tps_str}[/]" + " " * right_pad
            else:
                line4 += " " * left_pad + f"[bold red]{tps_str}[/]" + " " * right_pad
        # Target
        target_str = f"{target_tps:,.0f}"
        padding = COL_WIDTH - len(target_str)
        left_pad = padding // 2
        right_pad = padding - left_pad
        line4 += " " * left_pad + f"[bold yellow]{target_str}[/]" + " " * right_pad

        # Improvements with color coding (change from previous)
        line5 = " " * COL_WIDTH  # Empty for baseline
        for i in range(1, len(tps_history)):
            if tps_history[i-1] > 0:
                pct = (tps_history[i] - tps_history[i-1]) / tps_history[i-1] * 100
                if pct >= 0:
                    pct_str = f"+{pct:.0f}%"
                    padding = COL_WIDTH - len(pct_str)
                    left_pad = padding // 2
                    right_pad = padding - left_pad
                    line5 += " " * left_pad + f"[green]{pct_str}[/]" + " " * right_pad
                else:
                    pct_str = f"{pct:.0f}%"
                    padding = COL_WIDTH - len(pct_str)
                    left_pad = padding // 2
                    right_pad = padding - left_pad
                    line5 += " " * left_pad + f"[red]{pct_str}[/]" + " " * right_pad
            else:
                line5 += f"{'N/A':^{COL_WIDTH}}"
        line5 += " " * COL_WIDTH  # Empty for target

        # Status indicators line (hit/miss)
        line6 = " " * COL_WIDTH  # Empty for baseline
        for i in range(1, len(tps_history)):
            if round_results[i].get('hit', False):
                status_str = "✓ hit"
                padding = COL_WIDTH - len(status_str)
                left_pad = padding // 2
                right_pad = padding - left_pad
                line6 += " " * left_pad + f"[green]{status_str}[/]" + " " * right_pad
            else:
                status_str = "✗ miss"
                padding = COL_WIDTH - len(status_str)
                left_pad = padding // 2
                right_pad = padding - left_pad
                line6 += " " * left_pad + f"[red]{status_str}[/]" + " " * right_pad
        line6 += " " * COL_WIDTH  # Empty for target

        # Print timeline
        self.console.print()
        self.console.print("[bold]Session Timeline:[/]")
        self.console.print("═" * (len(labels) * COL_WIDTH + 2))
        self.console.print(line1)
        self.console.print(line2)
        self.console.print(line3)
        self.console.print(line4)
        self.console.print(line5)
        self.console.print(line6)
        self.console.print()
        self.console.print("[dim]Legend: [cyan]●[/]=baseline  [green]●[/]=hit  [red]✗[/]=miss  [green]★[/]=goal  ○=pending[/]")
        self.console.print()

    def _display_plain(self, tps_history: List[float], target_tps: float, current_round: int,
                       target_achieved: bool = False, round_results: List[Dict] = None):
        """Plain text timeline."""
        print("\nSession Timeline:")
        print("=" * 60)

        # Build round_results if not provided
        if round_results is None:
            round_results = []
            for i, tps in enumerate(tps_history):
                if i == 0:
                    round_results.append({'hit': False, 'is_baseline': True})
                else:
                    hit = tps >= target_tps if target_tps > 0 else False
                    round_results.append({'hit': hit, 'is_baseline': False})

        # Simple text representation
        for i, tps in enumerate(tps_history):
            label = "Baseline" if i == 0 else f"Round {i}"
            improvement = ""
            status = ""

            if i > 0:
                pct = (tps - tps_history[i-1]) / tps_history[i-1] * 100 if tps_history[i-1] > 0 else 0
                improvement = f" ({pct:+.0f}%)"
                status = " [HIT]" if round_results[i].get('hit', False) else " [MISS]"

            print(f"  {label}: {tps:,.0f} TPS{improvement}{status}")

        goal_status = " ★ ACHIEVED" if target_achieved else ""
        print(f"  Target: {target_tps:,.0f} TPS{goal_status}")
        print()


class DiffView:
    """
    Display configuration differences.

    Shows before/after comparison with impact indicators.
    """

    def __init__(self, console=None):
        self.console = console or (Console() if RICH_AVAILABLE else None)

    def display(self, before: Dict[str, str], after: Dict[str, str],
                estimated_impact: str = ""):
        """Display configuration diff."""
        if self.console and RICH_AVAILABLE:
            self._display_rich(before, after, estimated_impact)
        else:
            self._display_plain(before, after, estimated_impact)

    def _display_rich(self, before: Dict[str, str], after: Dict[str, str],
                      estimated_impact: str):
        """Rich display of diff."""
        table = Table(title="Configuration Changes", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan")
        table.add_column("Before", style="red")
        table.add_column("", width=3)
        table.add_column("After", style="green")
        table.add_column("Impact", style="yellow")

        # Find all changed parameters
        all_params = set(before.keys()) | set(after.keys())

        # Impact hints based on parameter type
        impact_hints = {
            'shared_buffers': '+++ TPS',
            'effective_cache_size': '++ cache',
            'work_mem': '+ query',
            'max_wal_size': '+ ckpt',
            'checkpoint_timeout': '+ ckpt',
            'random_page_cost': '+ plan',
            'effective_io_concurrency': '+ IO',
            'max_parallel_workers': '+ parallel',
        }

        for param in sorted(all_params):
            old_val = before.get(param, '-')
            new_val = after.get(param, '-')

            if old_val != new_val:
                impact = impact_hints.get(param, '')
                table.add_row(param, str(old_val), "→", str(new_val), impact)

        self.console.print(table)

        if estimated_impact:
            self.console.print(f"\n[bold]Estimated Impact:[/] {estimated_impact}")

    def _display_plain(self, before: Dict[str, str], after: Dict[str, str],
                       estimated_impact: str):
        """Plain text diff."""
        print("\nConfiguration Changes:")
        print("-" * 60)

        all_params = set(before.keys()) | set(after.keys())
        for param in sorted(all_params):
            old_val = before.get(param, '-')
            new_val = after.get(param, '-')
            if old_val != new_val:
                print(f"  {param}: {old_val} → {new_val}")

        if estimated_impact:
            print(f"\nEstimated Impact: {estimated_impact}")


class SafetyDisplay:
    """
    Display proposed changes with risk levels and confidence.
    """

    def __init__(self, console=None):
        self.console = console or (Console() if RICH_AVAILABLE else None)

    def display_proposal(self, changes: List[Dict], allow_selection: bool = True) -> List[int]:
        """
        Display proposed changes with safety info.

        Args:
            changes: List of change dicts with name, risk, confidence, etc.
            allow_selection: Whether to allow user selection

        Returns:
            List of indices of selected changes (if allow_selection)
        """
        if self.console and RICH_AVAILABLE:
            return self._display_rich(changes, allow_selection)
        else:
            return self._display_plain(changes, allow_selection)

    def _display_rich(self, changes: List[Dict], allow_selection: bool) -> List[int]:
        """Rich display with safety indicators."""
        self.console.print(Panel("[bold]PROPOSED CHANGES[/]", border_style="blue"))

        for i, change in enumerate(changes, 1):
            risk = change.get('risk', 'MEDIUM').upper()
            confidence = change.get('confidence', 0.5) * 100
            requires_restart = change.get('requires_restart', False)

            # Risk styling
            risk_style = {
                'LOW': ('[SAFE]', 'green'),
                'MEDIUM': ('[CAUTION]', 'yellow'),
                'HIGH': ('[EXPERIMENTAL]', 'red'),
            }.get(risk, ('[UNKNOWN]', 'white'))

            # Build display
            self.console.print(f"\n  {i}. [{risk_style[1]}]{risk_style[0]}[/] {change.get('name', 'Unknown')}")
            self.console.print(f"     Confidence: {confidence:.0f}%  Risk: {risk}  Restart: {'Yes' if requires_restart else 'No'}")

            # Show warning if any
            if change.get('warning'):
                self.console.print(f"     [yellow]⚠ {change['warning']}[/]")

            # Show commands
            for cmd in change.get('commands', [])[:2]:
                self.console.print(f"     [dim]→ {cmd}[/]")

        if allow_selection:
            self.console.print("\n  [bold]Apply:[/] [A]ll  [S]afe only  [1-{0}] Select  [N]one".format(len(changes)))
            choice = input("\n  Your choice: ").strip().lower()

            if choice == 'a':
                return list(range(len(changes)))
            elif choice == 's':
                return [i for i, c in enumerate(changes) if c.get('risk', '').upper() == 'LOW']
            elif choice == 'n':
                return []
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(changes):
                    return [idx]
            # Parse comma-separated
            elif ',' in choice:
                indices = []
                for part in choice.split(','):
                    if part.strip().isdigit():
                        idx = int(part.strip()) - 1
                        if 0 <= idx < len(changes):
                            indices.append(idx)
                return indices

        return list(range(len(changes)))

    def _display_plain(self, changes: List[Dict], allow_selection: bool) -> List[int]:
        """Plain text safety display."""
        print("\nPROPOSED CHANGES")
        print("=" * 50)

        for i, change in enumerate(changes, 1):
            risk = change.get('risk', 'MEDIUM').upper()
            confidence = change.get('confidence', 0.5) * 100
            restart = "Yes" if change.get('requires_restart') else "No"

            print(f"\n  {i}. [{risk}] {change.get('name', 'Unknown')}")
            print(f"     Confidence: {confidence:.0f}%  Restart: {restart}")

        if allow_selection:
            print("\n  Apply: [A]ll  [S]afe only  [N]one")
            choice = input("  Choice: ").strip().lower()

            if choice == 'a':
                return list(range(len(changes)))
            elif choice == 's':
                return [i for i, c in enumerate(changes) if c.get('risk', '').upper() == 'LOW']

        return []
