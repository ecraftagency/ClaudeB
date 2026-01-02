"""
StateMachine - Manages diagnostic workflow states.

v2.3 State Machine (with Human-in-the-Loop):
INIT → DISCOVER → STRATEGIZE → EXECUTE → AWAIT_USER_INPUT → ANALYZE → TUNE → VERIFY → COMPLETE
                                                                        ↓
                                                              EMERGENCY_ROLLBACK

AWAIT_USER_INPUT allows DBA to:
- Review benchmark results before AI analysis
- Add feedback/observations for AI context
- Control session flow (continue, skip, quit)
"""

from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


class State(Enum):
    """Diagnostic workflow states."""
    INIT = auto()
    DISCOVER = auto()
    STRATEGIZE = auto()
    EXECUTE = auto()
    AWAIT_USER_INPUT = auto()  # v2.3: Human-in-the-loop after benchmark
    ANALYZE = auto()
    TUNE = auto()
    VERIFY = auto()
    COMPLETE = auto()
    EMERGENCY_ROLLBACK = auto()
    FAILED = auto()


# Valid state transitions
TRANSITIONS: Dict[State, List[State]] = {
    State.INIT: [State.DISCOVER],
    State.DISCOVER: [State.STRATEGIZE, State.FAILED],
    State.STRATEGIZE: [State.EXECUTE, State.FAILED],
    State.EXECUTE: [State.AWAIT_USER_INPUT, State.ANALYZE, State.FAILED],  # v2.3: can go to user input
    State.AWAIT_USER_INPUT: [State.ANALYZE, State.COMPLETE, State.FAILED],  # v2.3: user decides next step
    State.ANALYZE: [State.TUNE, State.COMPLETE, State.FAILED],
    State.TUNE: [State.VERIFY, State.EMERGENCY_ROLLBACK],
    State.VERIFY: [State.COMPLETE, State.EXECUTE, State.FAILED],  # v2.3: VERIFY → EXECUTE for next iteration
    State.EMERGENCY_ROLLBACK: [State.ANALYZE, State.FAILED],
    State.COMPLETE: [],
    State.FAILED: [],
}


@dataclass
class StateEvent:
    """Record of a state transition."""
    from_state: State
    to_state: State
    timestamp: datetime
    duration_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateMachine:
    """
    Manages state transitions for the diagnostic workflow.

    Ensures valid transitions and tracks history.
    """

    def __init__(self, initial_state: State = State.INIT):
        self._state = initial_state
        self._history: List[StateEvent] = []
        self._state_entered_at = datetime.utcnow()
        self._callbacks: Dict[State, List[Callable]] = {}
        self._iteration = 0

    @property
    def state(self) -> State:
        """Get current state."""
        return self._state

    @property
    def iteration(self) -> int:
        """Get current iteration number."""
        return self._iteration

    @property
    def history(self) -> List[StateEvent]:
        """Get state transition history."""
        return self._history.copy()

    def can_transition(self, to_state: State) -> bool:
        """Check if transition to target state is valid."""
        return to_state in TRANSITIONS.get(self._state, [])

    def transition(self, to_state: State, metadata: Optional[Dict[str, Any]] = None):
        """
        Transition to a new state.

        Args:
            to_state: Target state
            metadata: Optional data about the transition

        Raises:
            ValueError: If transition is not valid
        """
        if not self.can_transition(to_state):
            raise ValueError(
                f"Invalid transition: {self._state.name} → {to_state.name}. "
                f"Valid transitions: {[s.name for s in TRANSITIONS.get(self._state, [])]}"
            )

        now = datetime.utcnow()
        duration_ms = int((now - self._state_entered_at).total_seconds() * 1000)

        # Record event
        event = StateEvent(
            from_state=self._state,
            to_state=to_state,
            timestamp=now,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._history.append(event)

        # Update state
        old_state = self._state
        self._state = to_state
        self._state_entered_at = now

        # Track iterations (ANALYZE → TUNE → VERIFY → ANALYZE loop)
        if to_state == State.ANALYZE and old_state in (State.VERIFY, State.EMERGENCY_ROLLBACK):
            self._iteration += 1

        # Trigger callbacks
        if to_state in self._callbacks:
            for callback in self._callbacks[to_state]:
                try:
                    callback(event)
                except Exception:
                    pass

    def on_enter(self, state: State, callback: Callable[[StateEvent], None]):
        """Register callback for state entry."""
        if state not in self._callbacks:
            self._callbacks[state] = []
        self._callbacks[state].append(callback)

    def get_duration_in_state(self) -> int:
        """Get milliseconds spent in current state."""
        now = datetime.utcnow()
        return int((now - self._state_entered_at).total_seconds() * 1000)

    def get_total_duration(self) -> int:
        """Get total duration of all completed states."""
        return sum(event.duration_ms for event in self._history)

    def is_terminal(self) -> bool:
        """Check if in terminal state (COMPLETE or FAILED)."""
        return self._state in (State.COMPLETE, State.FAILED)

    def reset(self):
        """Reset state machine to initial state."""
        self._state = State.INIT
        self._history = []
        self._state_entered_at = datetime.utcnow()
        self._iteration = 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of state machine execution."""
        state_durations = {}
        for event in self._history:
            state_name = event.from_state.name
            if state_name not in state_durations:
                state_durations[state_name] = 0
            state_durations[state_name] += event.duration_ms

        return {
            'current_state': self._state.name,
            'iteration': self._iteration,
            'total_transitions': len(self._history),
            'total_duration_ms': self.get_total_duration(),
            'state_durations': state_durations,
            'is_terminal': self.is_terminal(),
        }

    def format_history(self) -> str:
        """Format history as human-readable string."""
        lines = []
        for event in self._history:
            lines.append(
                f"{event.from_state.name} → {event.to_state.name} "
                f"({event.duration_ms}ms)"
            )
        return '\n'.join(lines)
