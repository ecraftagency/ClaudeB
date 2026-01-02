"""
Progress & Status - Workflow progress tracking and live status display.

Provides:
- Step indicator (Step X of Y: Phase Name)
- Session timeline visualization
- Live TPS display during benchmarks
- AI thinking indicator with spinner
- Time estimates for operations
"""

import time
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class WorkflowPhase(Enum):
    """Main workflow phases."""
    CONNECT = ("Connect", "Connecting to database", 30)
    DISCOVER = ("Discover", "Analyzing system", 60)
    STRATEGY = ("Strategy", "AI generating strategy", 45)
    BASELINE = ("Baseline", "Running baseline benchmark", 120)
    TUNING = ("Tuning", "AI tuning loop", 300)
    COMPLETE = ("Complete", "Session complete", 0)

    def __init__(self, display_name: str, description: str, estimated_seconds: int):
        self.display_name = display_name
        self.description = description
        self.estimated_seconds = estimated_seconds


@dataclass
class WorkflowState:
    """Current workflow state."""
    current_phase: WorkflowPhase = WorkflowPhase.CONNECT
    phase_number: int = 1
    total_phases: int = 5
    current_round: int = 0
    max_rounds: int = 3
    baseline_tps: float = 0
    current_tps: float = 0
    target_tps: float = 0
    rounds_completed: List[Dict[str, Any]] = field(default_factory=list)
    phase_start_time: float = field(default_factory=time.time)


class WorkflowProgress:
    """
    Workflow progress indicator.

    Shows: Step 2 of 5: Discover - Analyzing system
    """

    PHASE_ORDER = [
        WorkflowPhase.CONNECT,
        WorkflowPhase.DISCOVER,
        WorkflowPhase.STRATEGY,
        WorkflowPhase.BASELINE,
        WorkflowPhase.TUNING,
        WorkflowPhase.COMPLETE,
    ]

    def __init__(self, console: Console = None):
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.state = WorkflowState()
        self._live = None

    def set_phase(self, phase: WorkflowPhase):
        """Set current workflow phase."""
        self.state.current_phase = phase
        self.state.phase_number = self.PHASE_ORDER.index(phase) + 1
        self.state.total_phases = len(self.PHASE_ORDER) - 1  # Exclude COMPLETE from count
        self.state.phase_start_time = time.time()

    def set_round(self, round_num: int, max_rounds: int = 3):
        """Set current tuning round."""
        self.state.current_round = round_num
        self.state.max_rounds = max_rounds

    def set_tps(self, current: float, target: float = None, baseline: float = None):
        """Update TPS values."""
        self.state.current_tps = current
        if target is not None:
            self.state.target_tps = target
        if baseline is not None:
            self.state.baseline_tps = baseline

    def add_completed_round(self, round_info: Dict[str, Any]):
        """Add a completed round to history."""
        self.state.rounds_completed.append(round_info)

    def render_step_indicator(self) -> str:
        """Render step indicator string."""
        phase = self.state.current_phase
        step = self.state.phase_number
        total = self.state.total_phases

        if phase == WorkflowPhase.COMPLETE:
            return "Complete"

        # Add round info for tuning phase
        round_info = ""
        if phase == WorkflowPhase.TUNING and self.state.current_round > 0:
            round_info = f" (Round {self.state.current_round}/{self.state.max_rounds})"

        return f"Step {step} of {total}: {phase.display_name}{round_info}"

    def render_progress_bar(self) -> str:
        """Render ASCII progress bar."""
        step = self.state.phase_number
        total = self.state.total_phases

        filled = step - 1
        current = 1
        remaining = total - step

        bar = "█" * filled + "▓" + "░" * remaining
        return f"[{bar}]"

    def display(self):
        """Display the step indicator."""
        step_text = self.render_step_indicator()
        progress_bar = self.render_progress_bar()
        description = self.state.current_phase.description

        if self.console and RICH_AVAILABLE:
            text = Text()
            text.append(f"  {progress_bar} ", style="dim")
            text.append(step_text, style="bold cyan")
            text.append(f" - {description}", style="dim")

            self.console.print()
            self.console.print(Panel(
                text,
                box=box.SIMPLE,
                padding=(0, 1),
                style="dim",
            ))
        else:
            print()
            print(f"  {progress_bar} {step_text} - {description}")
            print()


class SessionTimeline:
    """
    Visual timeline of session progress.

    Shows: [Baseline ✓] → [R1 ✓ +12%] → [R2 ●] → [R3 ○]
    """

    def __init__(self, console: Console = None):
        self.console = console or (Console() if RICH_AVAILABLE else None)

    def render(self, state: WorkflowState) -> str:
        """Render timeline string."""
        parts = []

        # Baseline
        if state.baseline_tps > 0:
            parts.append(f"[Baseline ✓ {state.baseline_tps:,.0f}]")
        else:
            parts.append("[Baseline ○]")

        # Rounds
        for i in range(1, state.max_rounds + 1):
            if i < state.current_round:
                # Completed round
                round_data = None
                if i <= len(state.rounds_completed):
                    round_data = state.rounds_completed[i - 1]

                if round_data:
                    improvement = round_data.get('improvement_pct', 0)
                    sign = "+" if improvement >= 0 else ""
                    parts.append(f"[R{i} ✓ {sign}{improvement:.0f}%]")
                else:
                    parts.append(f"[R{i} ✓]")
            elif i == state.current_round:
                # Current round
                parts.append(f"[R{i} ●]")
            else:
                # Future round
                parts.append(f"[R{i} ○]")

        return " → ".join(parts)

    def display(self, state: WorkflowState):
        """Display the timeline."""
        timeline = self.render(state)

        if self.console and RICH_AVAILABLE:
            self.console.print()
            self.console.print(f"  {timeline}", style="dim")
        else:
            print()
            print(f"  {timeline}")


class LiveTPSDisplay:
    """
    Live TPS display during benchmarks.

    Shows real-time TPS updates with sparkline history.
    """

    def __init__(self, console: Console = None):
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.tps_history: List[float] = []
        self.max_history = 30
        self._live = None
        self._running = False
        self._update_thread = None
        self._tps_callback: Optional[Callable[[], float]] = None

    def _sparkline(self, values: List[float], width: int = 20) -> str:
        """Generate ASCII sparkline from values."""
        if not values:
            return "─" * width

        # Normalize to available characters
        chars = " ▁▂▃▄▅▆▇█"
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1

        # Take last 'width' values
        recent = values[-width:]

        line = ""
        for v in recent:
            normalized = (v - min_val) / range_val
            idx = int(normalized * (len(chars) - 1))
            line += chars[idx]

        # Pad if needed
        if len(line) < width:
            line = "─" * (width - len(line)) + line

        return line

    def add_tps(self, tps: float):
        """Add a TPS reading."""
        self.tps_history.append(tps)
        if len(self.tps_history) > self.max_history:
            self.tps_history.pop(0)

    def render(self, current_tps: float, target_tps: float, elapsed_seconds: int, total_seconds: int) -> Any:
        """Render the live display."""
        sparkline = self._sparkline(self.tps_history)
        progress_pct = (elapsed_seconds / total_seconds * 100) if total_seconds > 0 else 0
        remaining = max(0, total_seconds - elapsed_seconds)

        # TPS status
        if target_tps > 0:
            pct_of_target = (current_tps / target_tps * 100) if target_tps > 0 else 0
            tps_color = "green" if pct_of_target >= 100 else "yellow" if pct_of_target >= 80 else "white"
        else:
            tps_color = "white"

        if self.console and RICH_AVAILABLE:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Label", style="dim", width=12)
            table.add_column("Value", width=25)
            table.add_column("Spark", width=22)

            # TPS row
            tps_text = Text(f"{current_tps:,.0f} TPS", style=tps_color)
            if target_tps > 0:
                tps_text.append(f" / {target_tps:,.0f}", style="dim")

            table.add_row("Current:", tps_text, Text(sparkline, style="cyan"))

            # Time row
            time_text = f"{elapsed_seconds}s / {total_seconds}s ({remaining}s left)"
            progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
            table.add_row("Elapsed:", time_text, Text(progress_bar, style="blue"))

            return Panel(table, title="[bold]Benchmark Progress[/]", box=box.ROUNDED)
        else:
            lines = [
                f"  TPS: {current_tps:,.0f}" + (f" / {target_tps:,.0f}" if target_tps > 0 else ""),
                f"  Elapsed: {elapsed_seconds}s / {total_seconds}s ({remaining}s left)",
                f"  {sparkline}",
            ]
            return "\n".join(lines)

    def start_live(self, total_seconds: int, tps_callback: Callable[[], float]):
        """Start live display with callback for TPS updates."""
        if not RICH_AVAILABLE:
            return

        self._tps_callback = tps_callback
        self._running = True
        self.tps_history = []
        start_time = time.time()

        def update_display():
            with Live(console=self.console, refresh_per_second=1) as live:
                self._live = live
                while self._running:
                    elapsed = int(time.time() - start_time)
                    if elapsed >= total_seconds:
                        break

                    try:
                        current_tps = self._tps_callback()
                        self.add_tps(current_tps)
                        display = self.render(current_tps, 0, elapsed, total_seconds)
                        live.update(display)
                    except Exception:
                        pass

                    time.sleep(1)

        self._update_thread = threading.Thread(target=update_display, daemon=True)
        self._update_thread.start()

    def stop_live(self):
        """Stop live display."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=2)
        self._live = None


class AIThinkingIndicator:
    """
    AI thinking indicator with spinner and estimated time.
    """

    def __init__(self, console: Console = None):
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self._progress = None
        self._task_id = None

    def start(self, message: str = "AI analyzing...", estimated_seconds: int = 30):
        """Start the thinking indicator."""
        if not RICH_AVAILABLE or not self.console:
            print(f"  {message} (est. {estimated_seconds}s)")
            return

        self._progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]{task.description}[/]"),
            TextColumn("[dim]•[/]"),
            TimeElapsedColumn(),
            TextColumn("[dim]/ ~{task.fields[estimate]}s[/]"),
            console=self.console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(message, total=None, estimate=estimated_seconds)

    def update(self, message: str):
        """Update the message."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=message)

    def stop(self, final_message: str = None):
        """Stop the indicator."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None

        if final_message and self.console:
            self.console.print(f"  [green]✓[/] {final_message}")
        elif final_message:
            print(f"  ✓ {final_message}")


class UnifiedStatusBar:
    """
    Unified always-visible status bar combining all status info.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Step 2/5: Baseline │ R1 ● R2 ○ R3 ○ │ TPS: 4,500 → 5,000 │ ~2m left │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, console: Console = None):
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.workflow = WorkflowProgress(self.console)
        self.timeline = SessionTimeline(self.console)

    def update(self, **kwargs):
        """Update status bar state."""
        if 'phase' in kwargs:
            self.workflow.set_phase(kwargs['phase'])
        if 'round' in kwargs:
            self.workflow.set_round(kwargs['round'], kwargs.get('max_rounds', 3))
        if 'current_tps' in kwargs:
            self.workflow.set_tps(
                kwargs['current_tps'],
                kwargs.get('target_tps'),
                kwargs.get('baseline_tps')
            )
        if 'completed_round' in kwargs:
            self.workflow.add_completed_round(kwargs['completed_round'])

    def render(self) -> str:
        """Render the unified status bar."""
        state = self.workflow.state
        parts = []

        # Step indicator
        step_text = self.workflow.render_step_indicator()
        parts.append(step_text)

        # Round status (if in tuning)
        if state.current_phase == WorkflowPhase.TUNING:
            round_marks = []
            for i in range(1, state.max_rounds + 1):
                if i < state.current_round:
                    round_marks.append(f"R{i}✓")
                elif i == state.current_round:
                    round_marks.append(f"R{i}●")
                else:
                    round_marks.append(f"R{i}○")
            parts.append(" ".join(round_marks))

        # TPS
        if state.current_tps > 0:
            tps_text = f"TPS: {state.current_tps:,.0f}"
            if state.target_tps > 0:
                arrow = "✓" if state.current_tps >= state.target_tps else "→"
                tps_text += f" {arrow} {state.target_tps:,.0f}"
            parts.append(tps_text)

        # Time estimate
        if state.current_phase.estimated_seconds > 0:
            elapsed = int(time.time() - state.phase_start_time)
            remaining = max(0, state.current_phase.estimated_seconds - elapsed)
            if remaining > 60:
                parts.append(f"~{remaining // 60}m left")
            elif remaining > 0:
                parts.append(f"~{remaining}s left")

        return " │ ".join(parts)

    def display(self):
        """Display the unified status bar."""
        status_text = self.render()

        if self.console and RICH_AVAILABLE:
            self.console.print()
            self.console.print(Panel(
                status_text,
                box=box.ROUNDED,
                style="dim",
                padding=(0, 1),
            ))
        else:
            print()
            print("-" * 70)
            print(f" {status_text}")
            print("-" * 70)

    def display_with_timeline(self):
        """Display status bar with full timeline."""
        self.display()
        if self.workflow.state.current_phase == WorkflowPhase.TUNING:
            self.timeline.display(self.workflow.state)
