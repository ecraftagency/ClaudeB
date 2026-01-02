"""
Status Line - Always-visible status bar showing current workspace/session.

Like vim/VS Code status bar at the bottom of the terminal.
"""

from typing import Dict, Any, Optional
from datetime import datetime


try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class StatusLine:
    """
    Status line component - shows current workspace/session state.

    Example:
    ┌──────────────────────────────────────────────────────────────────┐
    │ 📂 mydb@10.0.0.230 │ 📋 BalancedTPS (R2) │ 🎯 8928/9000 TPS     │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, console: Console = None):
        self.console = console
        self._workspace_name: str = ""
        self._session_name: str = ""
        self._session_round: int = 0
        self._current_tps: float = 0
        self._target_tps: float = 0
        self._session_state: str = ""

    def update(self, workspace_status: Dict[str, Any]):
        """Update status from workspace manager."""
        self._workspace_name = workspace_status.get('workspace', '')

        session = workspace_status.get('session')
        if session:
            self._session_name = session.get('name', '')
            self._session_round = session.get('round', 0)
            self._current_tps = session.get('tps', 0)
            self._target_tps = session.get('target', 0)
            self._session_state = session.get('state', '')
        else:
            self._session_name = ""
            self._session_round = 0
            self._current_tps = 0
            self._target_tps = 0
            self._session_state = ""

    def render(self) -> str:
        """Render status line as string."""
        if not self._workspace_name:
            return "[No workspace]"

        parts = []

        # Workspace
        parts.append(f"📂 {self._workspace_name}")

        # Session
        if self._session_name:
            session_part = f"📋 {self._session_name}"
            if self._session_round > 0:
                session_part += f" (R{self._session_round})"
            parts.append(session_part)

            # TPS
            if self._current_tps > 0:
                tps_part = f"🎯 {self._current_tps:,.0f}"
                if self._target_tps > 0:
                    tps_part += f"/{self._target_tps:,.0f}"
                    if self._current_tps >= self._target_tps:
                        tps_part += " ✓"
                tps_part += " TPS"
                parts.append(tps_part)

        return " │ ".join(parts)

    def display(self):
        """Display status line."""
        status_text = self.render()

        if self.console and RICH_AVAILABLE:
            # Rich panel at bottom
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

    def get_prompt_prefix(self) -> str:
        """Get short prefix for input prompts."""
        if not self._workspace_name:
            return ""

        if self._session_name:
            return f"[{self._session_name[:15]}] "
        return f"[{self._workspace_name[:15]}] "


class StatusLineManager:
    """
    Manages status line display and updates.

    Integrates with workspace manager to show current state.
    """

    def __init__(self, console: Console = None):
        self.console = console
        self.status_line = StatusLine(console)
        self._enabled = True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def update_from_workspace(self, workspace_manager):
        """Update status from workspace manager."""
        if workspace_manager:
            status = workspace_manager.get_status()
            self.status_line.update(status)

    def show(self):
        """Show status line if enabled."""
        if self._enabled:
            self.status_line.display()

    def get_prompt(self, base_prompt: str = "") -> str:
        """Get prompt with status prefix."""
        prefix = self.status_line.get_prompt_prefix()
        return f"{prefix}{base_prompt}"
