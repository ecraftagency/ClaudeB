"""
InteractionManager - Human-in-the-Loop UI for v2.3.

Handles:
- Post-benchmark user interaction (AWAIT_USER_INPUT state)
- DBA feedback collection
- Session flow control
"""

import sys
from typing import Optional, Tuple
from enum import Enum
from datetime import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..protocol.result import HumanFeedback


class UserAction(str, Enum):
    """Possible user actions after benchmark."""
    CONTINUE = "CONTINUE"       # Send results to AI for analysis
    ADD_FEEDBACK = "FEEDBACK"   # Add observation/correction before sending
    QUIT = "QUIT"               # Exit the session
    SKIP = "SKIP"               # Skip AI analysis, proceed to next iteration


class InteractionManager:
    """
    Manages human-in-the-loop interaction for pg_diagnose v2.3.

    Displayed after benchmark execution to allow DBA to:
    1. Review results
    2. Add feedback/observations
    3. Control session flow
    """

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.console = Console() if RICH_AVAILABLE else None
        self._collected_feedback: Optional[HumanFeedback] = None

    def await_user_input(self, benchmark_summary: str = "") -> Tuple[UserAction, Optional[HumanFeedback]]:
        """
        Display post-benchmark prompt and await user decision.

        Args:
            benchmark_summary: Brief summary of benchmark results to display

        Returns:
            Tuple of (action, feedback) where feedback is populated if user added any
        """
        if self.quiet:
            return UserAction.CONTINUE, None

        self._collected_feedback = None

        if self.console:
            return self._rich_prompt(benchmark_summary)
        else:
            return self._simple_prompt(benchmark_summary)

    def _rich_prompt(self, summary: str) -> Tuple[UserAction, Optional[HumanFeedback]]:
        """Rich-based interactive prompt."""
        self.console.print()
        self.console.rule("[bold cyan]Benchmark Complete[/]")

        if summary:
            self.console.print(Panel(summary, title="Results Summary", border_style="dim"))

        # Show options
        self.console.print()
        options = Text()
        options.append("[ENTER]", style="bold green")
        options.append(" Send to AI for analysis  ")
        options.append("[F]", style="bold yellow")
        options.append(" Add feedback  ")
        options.append("[S]", style="bold blue")
        options.append(" Skip analysis  ")
        options.append("[Q]", style="bold red")
        options.append(" Quit session")

        self.console.print(Panel(options, title="Options", border_style="cyan"))

        while True:
            choice = Prompt.ask(
                "[bold]Your choice[/]",
                default="",
                show_default=False
            ).strip().upper()

            if choice == "" or choice == "C":
                return UserAction.CONTINUE, self._collected_feedback

            elif choice == "F":
                feedback = self._collect_feedback_rich()
                if feedback:
                    self._collected_feedback = feedback
                    self.console.print("[green]Feedback recorded.[/] Press ENTER to send to AI, or add more feedback.")
                continue

            elif choice == "S":
                return UserAction.SKIP, self._collected_feedback

            elif choice == "Q":
                if self._confirm_quit():
                    return UserAction.QUIT, self._collected_feedback
                continue

            else:
                self.console.print("[yellow]Invalid choice. Try again.[/]")

    def _simple_prompt(self, summary: str) -> Tuple[UserAction, Optional[HumanFeedback]]:
        """Simple stdin-based prompt (fallback without Rich)."""
        print()
        print("=" * 60)
        print(" Benchmark Complete")
        print("=" * 60)

        if summary:
            print(summary)
            print()

        print("Options:")
        print("  [ENTER] Send results to AI for analysis")
        print("  [F]     Add feedback/observation")
        print("  [S]     Skip AI analysis")
        print("  [Q]     Quit session")
        print()

        while True:
            choice = input("Your choice: ").strip().upper()

            if choice == "" or choice == "C":
                return UserAction.CONTINUE, self._collected_feedback

            elif choice == "F":
                feedback = self._collect_feedback_simple()
                if feedback:
                    self._collected_feedback = feedback
                    print("Feedback recorded. Press ENTER to send to AI.")
                continue

            elif choice == "S":
                return UserAction.SKIP, self._collected_feedback

            elif choice == "Q":
                confirm = input("Really quit? [y/N]: ").strip().lower()
                if confirm == 'y':
                    return UserAction.QUIT, self._collected_feedback
                continue

            else:
                print("Invalid choice. Try again.")

    def _collect_feedback_rich(self) -> Optional[HumanFeedback]:
        """Collect feedback using Rich prompts."""
        self.console.print()
        self.console.print("[bold]Add DBA Feedback[/]")
        self.console.print("[dim]This will be sent to AI along with benchmark results.[/]")
        self.console.print()

        # Get intent
        intent_options = {
            "1": ("CLARIFICATION", "Provide additional context or observations"),
            "2": ("CORRECTION", "Correct an AI assumption or prior analysis"),
            "3": ("NEW_DIRECTION", "Suggest a different tuning approach"),
        }

        self.console.print("[bold]Feedback type:[/]")
        for key, (intent, desc) in intent_options.items():
            self.console.print(f"  [{key}] {intent}: {desc}")

        intent_choice = Prompt.ask(
            "Select type",
            choices=["1", "2", "3"],
            default="1"
        )
        intent = intent_options[intent_choice][0]

        # Get feedback text
        self.console.print()
        self.console.print("[dim]Enter your feedback (press ENTER twice to finish):[/]")

        lines = []
        empty_count = 0
        while empty_count < 1:
            try:
                line = input()
                if line == "":
                    empty_count += 1
                else:
                    empty_count = 0
                    lines.append(line)
            except EOFError:
                break

        feedback_text = "\n".join(lines).strip()

        if not feedback_text:
            self.console.print("[yellow]No feedback entered.[/]")
            return None

        return HumanFeedback(
            feedback_text=feedback_text,
            intent=intent,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    def _collect_feedback_simple(self) -> Optional[HumanFeedback]:
        """Collect feedback using simple stdin."""
        print()
        print("Add DBA Feedback")
        print("-" * 40)
        print("Feedback types:")
        print("  [1] CLARIFICATION - Additional context")
        print("  [2] CORRECTION - Correct AI assumption")
        print("  [3] NEW_DIRECTION - Suggest different approach")

        intent_choice = input("Select type [1]: ").strip() or "1"
        intent_map = {"1": "CLARIFICATION", "2": "CORRECTION", "3": "NEW_DIRECTION"}
        intent = intent_map.get(intent_choice, "CLARIFICATION")

        print()
        print("Enter feedback (empty line to finish):")

        lines = []
        while True:
            try:
                line = input()
                if line == "":
                    break
                lines.append(line)
            except EOFError:
                break

        feedback_text = "\n".join(lines).strip()

        if not feedback_text:
            print("No feedback entered.")
            return None

        return HumanFeedback(
            feedback_text=feedback_text,
            intent=intent,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    def _confirm_quit(self) -> bool:
        """Confirm session quit."""
        if self.console:
            from rich.prompt import Confirm
            return Confirm.ask("Really quit the session?", default=False)
        else:
            response = input("Really quit? [y/N]: ").strip().lower()
            return response == 'y'

    def display_feedback_summary(self, feedback: HumanFeedback):
        """Display recorded feedback for confirmation."""
        if self.quiet:
            return

        if self.console:
            self.console.print()
            self.console.print(Panel(
                f"[bold]Intent:[/] {feedback.intent}\n\n{feedback.feedback_text}",
                title="Recorded Feedback",
                border_style="green"
            ))
        else:
            print()
            print(f"Recorded Feedback ({feedback.intent}):")
            print("-" * 40)
            print(feedback.feedback_text)
            print("-" * 40)

    def notify_sending_to_ai(self, has_feedback: bool = False):
        """Notify user that results are being sent to AI."""
        if self.quiet:
            return

        msg = "Sending results to AI for analysis"
        if has_feedback:
            msg += " (with your feedback)"
        msg += "..."

        if self.console:
            self.console.print(f"\n[bold cyan]{msg}[/]")
        else:
            print(f"\n{msg}")
