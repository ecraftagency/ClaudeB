"""
UI module - Rich console interface for diagnostics.

Provides:
- Progress display
- State machine visualization
- Results formatting
- Interactive proposal review
- Human-in-the-loop interaction (v2.3)
- Status line for workspace/session display
- Slash commands for IDE-like interaction
- Workflow progress indicators (v2.4)
- Live TPS display during benchmarks (v2.4)
- AI thinking indicator (v2.4)
"""

from .console import ConsoleUI
from .display import ResultDisplay
from .interaction import InteractionManager, UserAction  # v2.3
from .statusline import StatusLine, StatusLineManager
from .commands import SlashCommandHandler, CommandResult, handle_slash_command
from .progress import (  # v2.4
    WorkflowPhase,
    WorkflowProgress,
    SessionTimeline,
    LiveTPSDisplay,
    AIThinkingIndicator,
    UnifiedStatusBar,
)

__all__ = [
    "ConsoleUI",
    "ResultDisplay",
    "InteractionManager",  # v2.3
    "UserAction",  # v2.3
    "StatusLine",
    "StatusLineManager",
    "SlashCommandHandler",
    "CommandResult",
    "handle_slash_command",
    # v2.4 Progress & Status
    "WorkflowPhase",
    "WorkflowProgress",
    "SessionTimeline",
    "LiveTPSDisplay",
    "AIThinkingIndicator",
    "UnifiedStatusBar",
]
