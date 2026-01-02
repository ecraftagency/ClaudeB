"""
Slash Command Handler - IDE-like commands for workspace/session management.

Commands:
    /status     - Show current workspace and session status
    /sessions   - List all sessions in current workspace
    /export     - Export session recommendations (archived sessions only)
    /close      - Close/archive current session
    /pause      - Pause current session
    /resume     - Resume a paused session
    /switch     - Switch to a different session
    /quit       - Exit the tool
    /help       - Show available commands
"""

from typing import Optional, Tuple, List, Dict, Any, Callable
from enum import Enum


class CommandResult(Enum):
    """Result of command execution."""
    SUCCESS = "success"
    ERROR = "error"
    EXIT = "exit"
    CONTINUE = "continue"


class SlashCommandHandler:
    """
    Handles slash commands in the CLI.

    Provides IDE-like commands for workspace and session management.
    """

    def __init__(self, workspace_manager=None, console=None):
        self.workspace_manager = workspace_manager
        self.console = console

        # Command registry
        self._commands: Dict[str, Callable] = {
            'help': self._cmd_help,
            'h': self._cmd_help,
            '?': self._cmd_help,
            'status': self._cmd_status,
            's': self._cmd_status,
            'sessions': self._cmd_sessions,
            'ls': self._cmd_sessions,
            'export': self._cmd_export,
            'close': self._cmd_close,
            'pause': self._cmd_pause,
            'resume': self._cmd_resume,
            'retry': self._cmd_retry,
            'switch': self._cmd_switch,
            'quit': self._cmd_quit,
            'q': self._cmd_quit,
            'exit': self._cmd_quit,
        }

        # Command descriptions for help
        self._descriptions = {
            'help': 'Show available commands',
            'status': 'Show current workspace and session status',
            'sessions': 'List all sessions in current workspace',
            'export': 'Export session recommendations to file',
            'close': 'Close/archive current session',
            'pause': 'Pause current session',
            'resume': 'Resume a paused session',
            'retry': 'Retry a failed/error session',
            'switch': 'Switch to a different session',
            'quit': 'Exit the tool',
        }

    def is_command(self, input_text: str) -> bool:
        """Check if input is a slash command."""
        return input_text.strip().startswith('/')

    def parse(self, input_text: str) -> Tuple[str, List[str]]:
        """Parse command and arguments from input."""
        parts = input_text.strip().split()
        if not parts:
            return '', []

        command = parts[0].lstrip('/').lower()
        args = parts[1:] if len(parts) > 1 else []

        return command, args

    def execute(self, input_text: str) -> Tuple[CommandResult, str]:
        """
        Execute a slash command.

        Returns:
            Tuple of (result, message)
        """
        command, args = self.parse(input_text)

        if not command:
            return CommandResult.ERROR, "Empty command"

        if command not in self._commands:
            return CommandResult.ERROR, f"Unknown command: /{command}. Type /help for available commands."

        try:
            return self._commands[command](args)
        except Exception as e:
            return CommandResult.ERROR, f"Command error: {str(e)}"

    def _cmd_help(self, args: List[str]) -> Tuple[CommandResult, str]:
        """Show available commands."""
        lines = [
            "Available Commands:",
            "",
            "  /status, /s      - Show current workspace and session status",
            "  /sessions, /ls   - List all sessions in current workspace",
            "  /export          - Export session recommendations",
            "  /close           - Close/archive current session",
            "  /pause           - Pause current session",
            "  /resume [name]   - Resume a paused session",
            "  /retry [name]    - Retry a failed/error session",
            "  /switch [name]   - Switch to a different session",
            "  /quit, /q        - Exit the tool",
            "  /help, /h, /?    - Show this help",
            "",
            "Session States:",
            "  ACTIVE     - Currently running",
            "  PAUSED     - Interrupted, can /resume",
            "  ERROR      - Recoverable error, can /retry",
            "  FAILED     - Max rounds without target, can /retry",
            "  ARCHIVED   - Completed successfully",
            "  ABANDONED  - Stale (no activity 24h+), can /resume",
            "",
            "Tips:",
            "  - Sessions auto-save at strategy selection, baseline, and each round",
            "  - Use /export on archived sessions to get LLM-composed recommendations",
            "  - Use /retry to restart failed sessions from last checkpoint",
        ]
        return CommandResult.SUCCESS, "\n".join(lines)

    def _cmd_status(self, args: List[str]) -> Tuple[CommandResult, str]:
        """Show current workspace and session status."""
        if not self.workspace_manager:
            return CommandResult.ERROR, "No workspace manager available"

        status = self.workspace_manager.get_status()

        lines = []

        # Workspace info
        workspace_name = status.get('workspace', 'None')
        lines.append(f"Workspace: {workspace_name}")

        if status.get('workspace_path'):
            lines.append(f"Path: {status['workspace_path']}")

        # Session info
        session = status.get('session')
        if session:
            lines.append("")
            lines.append(f"Active Session: {session.get('name', 'Unknown')}")
            lines.append(f"Strategy: {session.get('strategy', 'Not selected')}")
            lines.append(f"Round: {session.get('round', 0)}")
            lines.append(f"State: {session.get('state', 'Unknown')}")

            if session.get('tps', 0) > 0:
                tps_str = f"Current TPS: {session['tps']:,.0f}"
                if session.get('target', 0) > 0:
                    tps_str += f" / Target: {session['target']:,.0f}"
                lines.append(tps_str)

            if session.get('baseline_tps', 0) > 0:
                lines.append(f"Baseline TPS: {session['baseline_tps']:,.0f}")
        else:
            lines.append("")
            lines.append("No active session")

        # Session counts
        if 'session_count' in status:
            lines.append("")
            lines.append(f"Total Sessions: {status['session_count']}")
            if status.get('archived_count', 0) > 0:
                lines.append(f"Archived: {status['archived_count']}")

        return CommandResult.SUCCESS, "\n".join(lines)

    def _cmd_sessions(self, args: List[str]) -> Tuple[CommandResult, str]:
        """List all sessions in current workspace."""
        if not self.workspace_manager:
            return CommandResult.ERROR, "No workspace manager available"

        if not self.workspace_manager.current_workspace:
            return CommandResult.ERROR, "No workspace open"

        sessions = self.workspace_manager.list_sessions()

        if not sessions:
            return CommandResult.SUCCESS, "No sessions in this workspace"

        lines = ["Sessions in current workspace:", ""]

        # Group by state
        active = [s for s in sessions if s.get('state') == 'active']
        paused = [s for s in sessions if s.get('state') == 'paused']
        archived = [s for s in sessions if s.get('state') == 'archived']
        failed = [s for s in sessions if s.get('state') == 'failed']

        if active:
            lines.append("Active:")
            for s in active:
                lines.append(f"  * {s['name']} - R{s.get('round', 0)} - {s.get('tps', 0):,.0f} TPS")

        if paused:
            lines.append("Paused:")
            for s in paused:
                lines.append(f"  ⏸ {s['name']} - R{s.get('round', 0)}")

        if archived:
            lines.append("Archived:")
            for s in archived:
                result = "✓" if s.get('target_achieved') else "○"
                lines.append(f"  {result} {s['name']} - {s.get('tps', 0):,.0f} TPS")

        if failed:
            lines.append("Failed:")
            for s in failed:
                lines.append(f"  ✗ {s['name']}")

        return CommandResult.SUCCESS, "\n".join(lines)

    def _cmd_export(self, args: List[str]) -> Tuple[CommandResult, str]:
        """Export session recommendations."""
        if not self.workspace_manager:
            return CommandResult.ERROR, "No workspace manager available"

        if not self.workspace_manager.current_workspace:
            return CommandResult.ERROR, "No workspace open"

        # Check for specific session name in args
        session_name = args[0] if args else None

        # Export returns path or raises error
        try:
            result = self.workspace_manager.export_session(session_name)
            return CommandResult.SUCCESS, f"Exported to: {result}"
        except ValueError as e:
            return CommandResult.ERROR, str(e)

    def _cmd_close(self, args: List[str]) -> Tuple[CommandResult, str]:
        """Close/archive current session."""
        if not self.workspace_manager:
            return CommandResult.ERROR, "No workspace manager available"

        session = self.workspace_manager.current_session
        if not session:
            return CommandResult.ERROR, "No active session to close"

        # Archive the session
        session_name = session.name
        self.workspace_manager.archive_session()

        return CommandResult.SUCCESS, f"Session '{session_name}' archived. Use /export to generate recommendations."

    def _cmd_pause(self, args: List[str]) -> Tuple[CommandResult, str]:
        """Pause current session."""
        if not self.workspace_manager:
            return CommandResult.ERROR, "No workspace manager available"

        session = self.workspace_manager.current_session
        if not session:
            return CommandResult.ERROR, "No active session to pause"

        session_name = session.name
        self.workspace_manager.pause_session()

        return CommandResult.SUCCESS, f"Session '{session_name}' paused. Use /resume to continue later."

    def _cmd_resume(self, args: List[str]) -> Tuple[CommandResult, str]:
        """Resume a paused session."""
        if not self.workspace_manager:
            return CommandResult.ERROR, "No workspace manager available"

        if not self.workspace_manager.current_workspace:
            return CommandResult.ERROR, "No workspace open"

        if not args:
            # Try to resume most recent paused session
            sessions = self.workspace_manager.list_sessions()
            paused = [s for s in sessions if s.get('state') == 'paused']

            if not paused:
                return CommandResult.ERROR, "No paused sessions to resume. Use /sessions to list."

            session_name = paused[0]['name']
        else:
            session_name = args[0]

        try:
            self.workspace_manager.resume_session(session_name)
            return CommandResult.SUCCESS, f"Resumed session '{session_name}'"
        except ValueError as e:
            return CommandResult.ERROR, str(e)

    def _cmd_retry(self, args: List[str]) -> Tuple[CommandResult, str]:
        """Retry a failed/error session."""
        if not self.workspace_manager:
            return CommandResult.ERROR, "No workspace manager available"

        if not self.workspace_manager.current_workspace:
            return CommandResult.ERROR, "No workspace open"

        # Get list of retryable sessions
        retryable = self.workspace_manager.get_recoverable_sessions()
        retryable = [s for s in retryable if s['state'] in ('error', 'failed', 'abandoned')]

        if not retryable:
            return CommandResult.ERROR, "No failed/error sessions to retry. Use /sessions to list."

        if not args:
            # Show retryable sessions and prompt
            lines = ["Sessions that can be retried:", ""]
            for i, s in enumerate(retryable, 1):
                state = s['state'].upper()
                lines.append(f"  [{i}] {s['strategy'][:30]} - {state} - {s['best_tps']:.0f} TPS")
            lines.append("")
            lines.append("Usage: /retry <session_name> or /retry <number>")
            return CommandResult.SUCCESS, "\n".join(lines)

        # Parse session name or number
        session_name = args[0]
        try:
            idx = int(session_name) - 1
            if 0 <= idx < len(retryable):
                session_name = retryable[idx]['name']
        except ValueError:
            pass  # Use as-is

        try:
            session = self.workspace_manager.retry_session(session_name)
            return CommandResult.SUCCESS, f"Retrying session '{session.name}' from last checkpoint"
        except ValueError as e:
            return CommandResult.ERROR, str(e)

    def _cmd_switch(self, args: List[str]) -> Tuple[CommandResult, str]:
        """Switch to a different session."""
        if not self.workspace_manager:
            return CommandResult.ERROR, "No workspace manager available"

        if not args:
            return CommandResult.ERROR, "Usage: /switch <session_name>"

        session_name = args[0]

        try:
            self.workspace_manager.switch_session(session_name)
            return CommandResult.SUCCESS, f"Switched to session '{session_name}'"
        except ValueError as e:
            return CommandResult.ERROR, str(e)

    def _cmd_quit(self, args: List[str]) -> Tuple[CommandResult, str]:
        """Exit the tool."""
        # Auto-save current session if any
        if self.workspace_manager and self.workspace_manager.current_session:
            session = self.workspace_manager.current_session
            self.workspace_manager.save_session()
            return CommandResult.EXIT, f"Session '{session.name}' saved. Goodbye!"

        return CommandResult.EXIT, "Goodbye!"


def handle_slash_command(input_text: str, workspace_manager=None, console=None) -> Tuple[CommandResult, str]:
    """
    Convenience function to handle a slash command.

    Usage:
        if input_text.startswith('/'):
            result, message = handle_slash_command(input_text, workspace_manager)
            print(message)
            if result == CommandResult.EXIT:
                break
    """
    handler = SlashCommandHandler(workspace_manager, console)
    return handler.execute(input_text)
