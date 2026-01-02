"""
Test Mode Handler - Enables automated testing of pg_diagnose CLI.

Provides:
- Structured JSON output for test verification
- Input from file for reproducible tests
- State tracking and dumping
- Integration with mock components
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, TextIO
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class TestCommand:
    """A command executed during testing."""
    timestamp: str
    command: str
    result: Dict[str, Any]
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None


@dataclass
class TestSession:
    """Test session tracking."""
    test_id: str
    started_at: str
    scenario: str
    commands: List[TestCommand] = field(default_factory=list)
    completed_at: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class TestModeHandler:
    """
    Handles test mode operations for pg_diagnose.

    Provides structured I/O for automated testing.

    Usage:
        handler = TestModeHandler(
            input_file="tests/scenario.txt",
            output_json=True,
            state_dump=True,
        )

        # Get next command
        cmd = handler.get_input("prompt> ")

        # Record result
        handler.record_result(cmd, {"success": True, "data": {...}})

        # Output result
        handler.output({"message": "Done"})
    """

    def __init__(
        self,
        input_file: Optional[str] = None,
        output_json: bool = False,
        state_dump: bool = False,
        scenario: str = "balanced_tps",
        output_stream: TextIO = None,
    ):
        """
        Initialize test mode handler.

        Args:
            input_file: Path to file with commands (one per line)
            output_json: Output all responses as JSON
            state_dump: Include state in output after each command
            scenario: Test scenario name
            output_stream: Output stream (default: stdout)
        """
        self.input_file = input_file
        self.output_json = output_json
        self.state_dump = state_dump
        self.scenario = scenario
        self.output_stream = output_stream or sys.stdout

        # Input handling
        self._input_lines: List[str] = []
        self._input_index = 0
        self._load_input_file()

        # Session tracking
        self.session = TestSession(
            test_id=f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            started_at=datetime.now().isoformat(),
            scenario=scenario,
        )

        # State tracking
        self._current_state: Dict[str, Any] = {
            "workspace": None,
            "session": None,
            "phase": None,
        }

    def _load_input_file(self):
        """Load commands from input file."""
        if self.input_file:
            try:
                with open(self.input_file, 'r') as f:
                    for line in f:
                        # Strip whitespace, skip empty lines and comments
                        line = line.strip()
                        if line and not line.startswith('#'):
                            self._input_lines.append(line)
            except FileNotFoundError:
                self.output_error(f"Input file not found: {self.input_file}")
                sys.exit(1)

    def get_input(self, prompt: str = "") -> str:
        """
        Get next input (from file or stdin).

        Args:
            prompt: Prompt to display (ignored in file mode)

        Returns:
            Next command/input string
        """
        if self._input_lines:
            # File input mode
            if self._input_index < len(self._input_lines):
                cmd = self._input_lines[self._input_index]
                self._input_index += 1

                # Log the command being executed
                if self.output_json:
                    self.output({
                        "type": "input",
                        "prompt": prompt,
                        "value": cmd,
                        "source": "file",
                        "line": self._input_index,
                    })
                else:
                    print(f"{prompt}{cmd}")

                return cmd
            else:
                # End of file - return quit
                return "/quit"
        else:
            # Interactive mode (shouldn't happen in test mode)
            return input(prompt)

    def has_more_input(self) -> bool:
        """Check if there's more input available."""
        if self._input_lines:
            return self._input_index < len(self._input_lines)
        return True  # Interactive mode always has more

    def output(self, data: Any, message_type: str = "output"):
        """
        Output data (as JSON or plain text).

        Args:
            data: Data to output
            message_type: Type of message (output, result, state, error)
        """
        if self.output_json:
            output = {
                "timestamp": datetime.now().isoformat(),
                "type": message_type,
            }
            if isinstance(data, dict):
                output["data"] = data
            else:
                output["data"] = {"message": str(data)}

            print(json.dumps(output), file=self.output_stream)
            self.output_stream.flush()
        else:
            if isinstance(data, dict):
                print(json.dumps(data, indent=2), file=self.output_stream)
            else:
                print(data, file=self.output_stream)

    def output_error(self, error: str):
        """Output an error."""
        self.session.success = False
        self.session.error = error
        self.output({"error": error}, message_type="error")

    def record_command(
        self,
        command: str,
        result: Dict[str, Any],
        state_before: Dict[str, Any] = None,
        state_after: Dict[str, Any] = None,
    ):
        """
        Record a command execution for the test log.

        Args:
            command: Command that was executed
            result: Result of the command
            state_before: State before command
            state_after: State after command
        """
        cmd_record = TestCommand(
            timestamp=datetime.now().isoformat(),
            command=command,
            result=result,
            state_before=state_before if self.state_dump else None,
            state_after=state_after if self.state_dump else None,
        )
        self.session.commands.append(cmd_record)

        # Output if state dump enabled
        if self.state_dump and state_after:
            self.output(state_after, message_type="state")

    def update_state(
        self,
        workspace: str = None,
        session: str = None,
        session_state: str = None,
        phase: str = None,
    ):
        """Update tracked state."""
        if workspace is not None:
            self._current_state["workspace"] = workspace
        if session is not None:
            self._current_state["session"] = session
        if session_state is not None:
            self._current_state["session_state"] = session_state
        if phase is not None:
            self._current_state["phase"] = phase

    def get_state(self) -> Dict[str, Any]:
        """Get current tracked state."""
        return self._current_state.copy()

    def finish(self) -> Dict[str, Any]:
        """
        Finish test session and return results.

        Returns:
            Complete test session data
        """
        self.session.completed_at = datetime.now().isoformat()

        result = {
            "test_id": self.session.test_id,
            "started_at": self.session.started_at,
            "completed_at": self.session.completed_at,
            "scenario": self.session.scenario,
            "success": self.session.success,
            "error": self.session.error,
            "commands_executed": len(self.session.commands),
            "final_state": self._current_state,
        }

        if self.output_json:
            self.output(result, message_type="test_complete")

        return result

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of test session."""
        return {
            "test_id": self.session.test_id,
            "scenario": self.session.scenario,
            "commands_executed": len(self.session.commands),
            "success": self.session.success,
            "current_state": self._current_state,
        }


class TestInputAdapter:
    """
    Adapter that wraps UI prompt methods for test mode.

    Replaces interactive prompts with file-based input.
    """

    def __init__(self, handler: TestModeHandler):
        self.handler = handler

    def prompt(self, message: str, allow_commands: bool = True) -> str:
        """Get input from test handler."""
        return self.handler.get_input(message)

    def confirm(self, message: str, default: bool = False) -> bool:
        """Get yes/no confirmation from test handler."""
        response = self.handler.get_input(message)
        return response.lower() in ['y', 'yes', '/run', '/apply', '/accept']

    def select(self, message: str, options: List[str]) -> int:
        """Get selection from test handler."""
        response = self.handler.get_input(message)
        try:
            return int(response) - 1  # 1-indexed to 0-indexed
        except ValueError:
            return 0  # Default to first option


class TestOutputAdapter:
    """
    Adapter that captures UI output for test verification.
    """

    def __init__(self, handler: TestModeHandler):
        self.handler = handler
        self.captured: List[Dict[str, Any]] = []

    def print(self, message: str):
        """Capture print output."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": "print",
            "message": message,
        }
        self.captured.append(record)
        self.handler.output(record, message_type="print")

    def print_error(self, message: str):
        """Capture error output."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "message": message,
        }
        self.captured.append(record)
        self.handler.output(record, message_type="error")

    def get_captured(self) -> List[Dict[str, Any]]:
        """Get all captured output."""
        return self.captured


def create_test_context(args) -> Dict[str, Any]:
    """
    Create test context from CLI arguments.

    Args:
        args: Parsed CLI arguments

    Returns:
        Dict with test mode configuration
    """
    return {
        "enabled": args.test_mode,
        "mock_benchmark": args.mock_benchmark,
        "mock_ai": args.mock_ai,
        "scenario": args.test_scenario,
        "input_file": args.input_file,
        "output_json": args.output_json,
        "state_dump": args.state_dump,
        "workspace_path": args.test_workspace,
    }


def setup_test_mode(args) -> Optional[TestModeHandler]:
    """
    Setup test mode if enabled.

    Args:
        args: Parsed CLI arguments

    Returns:
        TestModeHandler if test mode enabled, None otherwise
    """
    if not args.test_mode:
        return None

    handler = TestModeHandler(
        input_file=args.input_file,
        output_json=args.output_json,
        state_dump=args.state_dump,
        scenario=args.test_scenario,
    )

    # Output test start
    handler.output({
        "test_id": handler.session.test_id,
        "scenario": args.test_scenario,
        "mock_benchmark": args.mock_benchmark,
        "mock_ai": args.mock_ai,
        "input_file": args.input_file,
    }, message_type="test_start")

    return handler
