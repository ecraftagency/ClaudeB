# Session Lifecycle Analysis

## Current State Machine

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                        WORKSPACE                             │
                                    │  (No active session - can list, create, resume sessions)     │
                                    └─────────────────────────────────────────────────────────────┘
                                                              │
                        ┌─────────────────────────────────────┼─────────────────────────────────────┐
                        │                                     │                                     │
                        ▼                                     ▼                                     ▼
               ┌─────────────────┐                   ┌─────────────────┐                   ┌─────────────────┐
               │  Create New     │                   │  Resume Paused  │                   │  View Archived  │
               │  Session        │                   │  Session        │                   │  Sessions       │
               └────────┬────────┘                   └────────┬────────┘                   └─────────────────┘
                        │                                     │
                        ▼                                     │
        ┌───────────────────────────────┐                     │
        │                               │                     │
        │     SESSION: ACTIVE           │◄────────────────────┘
        │                               │
        │  States within ACTIVE:        │
        │  ├── Strategy Selection       │
        │  ├── Discovery Running        │
        │  ├── Baseline Benchmark       │
        │  ├── Tuning Loop              │
        │  │   ├── AI Analysis          │
        │  │   ├── DBA Review           │
        │  │   ├── Apply Changes        │
        │  │   ├── Run Benchmark        │
        │  │   └── Evaluate Results     │
        │  └── Target Check             │
        │                               │
        └───────────────┬───────────────┘
                        │
        ┌───────────────┼───────────────┬───────────────────────┐
        │               │               │                       │
        ▼               ▼               ▼                       ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐   ┌───────────────┐
│   PAUSED      │ │   ARCHIVED    │ │    FAILED     │   │  (Abandoned)  │
│               │ │               │ │               │   │   Implicit    │
│ User chose    │ │ Target hit    │ │ Max rounds    │   │   No state    │
│ /stop or      │ │ OR user       │ │ without hit   │   │               │
│ /pause        │ │ satisfied     │ │ OR error      │   │               │
└───────┬───────┘ └───────────────┘ └───────────────┘   └───────────────┘
        │
        │ /resume
        ▼
┌───────────────────────────────────┐
│     SESSION: ACTIVE (resumed)     │
│                                   │
│  Context restored:                │
│  - Strategy, target TPS           │
│  - Rounds completed               │
│  - Best TPS achieved              │
│  - Applied changes                │
│                                   │
│  Continues from last checkpoint   │
└───────────────────────────────────┘
```

## Session State Transitions

```
                    ┌──────────────┐
                    │    (none)    │
                    └──────┬───────┘
                           │ create_session()
                           ▼
                    ┌──────────────┐
         ┌──────────│    ACTIVE    │──────────┐
         │          └──────────────┘          │
         │                 │                  │
         │ pause()         │ archive()        │ fail()
         │                 │                  │
         ▼                 ▼                  ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │    PAUSED    │  │   ARCHIVED   │  │    FAILED    │
  └──────────────┘  └──────────────┘  └──────────────┘
         │
         │ resume()
         ▼
  ┌──────────────┐
  │    ACTIVE    │
  └──────────────┘
```

## Issues Identified

### 1. Missing State Transitions

| From State | To State | Use Case | Currently Supported |
|------------|----------|----------|---------------------|
| ACTIVE → PAUSED | User interrupt | ✅ Yes |
| ACTIVE → ARCHIVED | Success | ✅ Yes |
| ACTIVE → FAILED | Max rounds | ✅ Yes (but inconsistent) |
| PAUSED → ACTIVE | Resume | ✅ Yes |
| **FAILED → ACTIVE** | **Retry with different approach** | ❌ No |
| **ARCHIVED → ACTIVE** | **Re-run to improve further** | ❌ No |
| **PAUSED → ARCHIVED** | **Accept current results** | ❌ No |
| **PAUSED → FAILED** | **Abandon** | ❌ No |

### 2. Missing "Abandoned" State

Sessions that are:
- Started but never completed (connection lost)
- Left in ACTIVE state but workspace closed
- Started but user quit without explicit pause

These remain as "ACTIVE" indefinitely, which is confusing.

### 3. No Error Recovery States

What happens when:
- Benchmark returns 0 TPS? → Currently continues (wrong)
- Connection to DB drops? → Crashes
- AI agent fails? → Crashes
- SSH to remote server fails? → Crashes

### 4. Inconsistent State Management in CLI

Looking at `cli.py`, state transitions are scattered:

```python
# These are in different locations with no central state machine:
ws_session.pause()      # Line ~2713
ws_session.archive()    # Line ~2828
ws_session.save()       # Line ~2958
ws_session.fail()       # Never called!
```

### 5. Happy Path Focus

Current flow assumes:
1. Discovery succeeds
2. Strategy is valid
3. Baseline runs successfully
4. Each tuning round works
5. Target is achievable

No handling for:
- Discovery fails (no schema found)
- AI suggests invalid strategy
- Baseline returns 0 TPS (our current bug!)
- Benchmark consistently fails
- Target is unrealistic

## Recommended State Machine

```
                         ┌─────────────────────────────────────┐
                         │            WORKSPACE                 │
                         │  Commands: /new /resume /sessions    │
                         └──────────────────┬──────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
              ▼                             ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
    │      NEW         │          │     PAUSED       │          │    ARCHIVED      │
    │                  │          │                  │          │                  │
    │ create_session() │          │   /resume        │          │   (read-only)    │
    └────────┬─────────┘          └────────┬─────────┘          │   /export        │
             │                             │                    │   /reopen        │◄─────┐
             │                             │                    └──────────────────┘      │
             ▼                             ▼                                              │
    ┌────────────────────────────────────────────────────────────────┐                   │
    │                        ACTIVE                                   │                   │
    │                                                                 │                   │
    │  Sub-states (internal):                                         │                   │
    │  ┌─────────────────────────────────────────────────────────┐   │                   │
    │  │  INITIALIZING → DISCOVERING → STRATEGY_SELECT →         │   │                   │
    │  │  BASELINE → TUNING → EVALUATING                         │   │                   │
    │  └─────────────────────────────────────────────────────────┘   │                   │
    │                                                                 │                   │
    │  Commands: /stop /status /history /skip                         │                   │
    └───────────────────────────┬─────────────────────────────────────┘                   │
                                │                                                          │
          ┌─────────────────────┼─────────────────────┬──────────────────────┐            │
          │                     │                     │                      │            │
          ▼                     ▼                     ▼                      ▼            │
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       ┌──────────────┐     │
   │   PAUSED     │     │  COMPLETED   │     │    FAILED    │       │   ARCHIVED   │─────┘
   │              │     │              │     │              │       │              │
   │ User /stop   │     │ Target hit   │     │ Max rounds   │       │ User accepts │
   │ Recoverable  │     │ Confirmed    │     │ No progress  │       │ results      │
   └──────────────┘     └──────────────┘     │ Errors       │       └──────────────┘
          │                    │             └──────┬───────┘
          │                    │                    │
          │                    │                    ▼
          │                    │            ┌──────────────┐
          │                    │            │    RETRY     │
          │                    │            │              │
          │                    │            │ /retry cmd   │
          │                    │            │ Reset round  │
          │                    └────────────│ Try different│
          │                                 │ strategy     │
          └────────────────────────────────►└──────────────┘
```

## Detailed Sub-State Machine (within ACTIVE)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              ACTIVE SESSION                                     │
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ INITIALIZING│───►│ DISCOVERING │───►│  STRATEGY   │───►│  BASELINE   │      │
│  │             │    │             │    │  SELECTION  │    │             │      │
│  │ Load config │    │ Schema scan │    │             │    │ Run bench   │      │
│  │ Check conn  │    │ System info │    │ AI suggest  │    │ Capture TPS │      │
│  └─────────────┘    └──────┬──────┘    │ User choose │    └──────┬──────┘      │
│        │                   │           └──────┬──────┘           │             │
│        │                   │                  │                  │             │
│   [conn fail]         [no schema]        [no good]          [0 TPS]           │
│        │                   │             [strategy]             │             │
│        ▼                   ▼                  │                  ▼             │
│  ┌─────────────┐    ┌─────────────┐          │           ┌─────────────┐      │
│  │   ERROR     │    │   ERROR     │          │           │   ERROR     │      │
│  │             │    │             │          │           │             │      │
│  │ Show error  │    │ Check DB    │          │           │ Debug info  │      │
│  │ Retry/Exit  │    │ Manual hint │          │           │ Retry/Exit  │      │
│  └─────────────┘    └─────────────┘          │           └─────────────┘      │
│                                              │                                 │
│                                              ▼                                 │
│                     ┌────────────────────────────────────────────┐             │
│                     │               TUNING LOOP                   │             │
│                     │                                             │             │
│                     │  ┌──────────┐   ┌──────────┐   ┌──────────┐│             │
│                     │  │   AI     │──►│  DBA     │──►│  APPLY   ││             │
│                     │  │ ANALYSIS │   │  REVIEW  │   │ CHANGES  ││             │
│                     │  └──────────┘   └──────────┘   └────┬─────┘│             │
│                     │       ▲                             │      │             │
│                     │       │                             ▼      │             │
│                     │  ┌──────────┐   ┌──────────┐   ┌──────────┐│             │
│                     │  │ EVALUATE │◄──│ COLLECT  │◄──│BENCHMARK ││             │
│                     │  │ RESULTS  │   │ METRICS  │   │          ││             │
│                     │  └────┬─────┘   └──────────┘   └──────────┘│             │
│                     │       │                                     │             │
│                     │       ├── [target hit] ───► SUCCESS        │             │
│                     │       ├── [max rounds] ───► FAILED         │             │
│                     │       ├── [user /stop] ───► PAUSED         │             │
│                     │       └── [continue] ──────► (loop)        │             │
│                     │                                             │             │
│                     └─────────────────────────────────────────────┘             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Recommendations

### 1. Add Explicit Error State & Recovery

```python
class SessionState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    FAILED = "failed"
    ERROR = "error"      # NEW: Recoverable error
    ABANDONED = "abandoned"  # NEW: Stale session
```

### 2. Add Sub-State Tracking

```python
class SessionPhase(str, Enum):
    INITIALIZING = "initializing"
    DISCOVERING = "discovering"
    STRATEGY_SELECTION = "strategy_selection"
    BASELINE = "baseline"
    TUNING = "tuning"
    EVALUATING = "evaluating"
    COMPLETED = "completed"

@dataclass
class SessionData:
    state: str = "active"
    phase: str = "initializing"  # NEW: Sub-state tracking
    error_info: Optional[Dict] = None  # NEW: Error details
```

### 3. Centralize State Transitions

```python
class SessionStateMachine:
    """Central state management for session lifecycle."""

    VALID_TRANSITIONS = {
        SessionState.ACTIVE: [SessionState.PAUSED, SessionState.ARCHIVED,
                              SessionState.FAILED, SessionState.ERROR],
        SessionState.PAUSED: [SessionState.ACTIVE, SessionState.ARCHIVED,
                              SessionState.FAILED],
        SessionState.ERROR: [SessionState.ACTIVE, SessionState.FAILED],
        SessionState.FAILED: [SessionState.ACTIVE],  # Retry
        SessionState.ARCHIVED: [SessionState.ACTIVE],  # Reopen
    }

    def transition(self, from_state: SessionState, to_state: SessionState,
                   reason: str = "") -> bool:
        if to_state not in self.VALID_TRANSITIONS.get(from_state, []):
            raise InvalidStateTransition(f"Cannot go from {from_state} to {to_state}")
        # Log transition, update state, save
```

### 4. Add Recovery Points

```python
@dataclass
class SessionCheckpoint:
    """Recovery checkpoint for crash recovery."""
    phase: SessionPhase
    round_num: int
    timestamp: str
    data_snapshot: Dict[str, Any]

class Session:
    def save_checkpoint(self):
        """Save checkpoint for crash recovery."""
        checkpoint = SessionCheckpoint(
            phase=self.phase,
            round_num=self.data.current_round,
            timestamp=datetime.now().isoformat(),
            data_snapshot={...}
        )
        # Save to session directory
```

### 5. Handle Edge Cases

| Scenario | Current Behavior | Recommended |
|----------|------------------|-------------|
| Baseline returns 0 TPS | Continues with 0 | Show debug, offer retry/skip |
| Connection drops | Crash | Auto-pause, save checkpoint |
| AI returns invalid | Varies | Show error, offer retry/manual |
| User quits during round | ACTIVE forever | Auto-pause + checkpoint |
| Max rounds, no progress | Just stops | Offer /retry with different strategy |

### 6. Add Session Cleanup

```python
def cleanup_stale_sessions(workspace: Workspace, max_age_hours: int = 24):
    """Mark old ACTIVE sessions as ABANDONED."""
    for session in workspace.list_sessions():
        if session['state'] == 'active':
            updated = datetime.fromisoformat(session['updated_at'])
            if (datetime.now() - updated).total_seconds() > max_age_hours * 3600:
                sess = workspace.open_session(session['name'])
                sess.data.state = SessionState.ABANDONED.value
                sess.save()
```

## Priority Implementation Order

1. **[P0] Fix 0 TPS handling** - Show debug output, offer retry ✅ DONE
2. **[P0] Add checkpoint on crash** - Auto-save before risky operations ✅ DONE
3. **[P1] Add ERROR state** - For recoverable errors ✅ DONE
4. **[P1] Add phase tracking** - Know where we are in the flow ✅ DONE
5. **[P2] Add ABANDONED state** - Cleanup stale sessions ✅ DONE
6. **[P2] Add FAILED → ACTIVE transition** - Retry with different approach ✅ DONE
7. **[P3] Centralize state machine** - Single source of truth ✅ DONE

## Implementation Status

All priority items have been implemented:

- **SessionState enum**: Added ERROR, ABANDONED states
- **SessionPhase enum**: Tracks sub-states within ACTIVE (INITIALIZING through COMPLETED)
- **SessionStateMachine class**: Centralized state transitions with validation
- **Checkpoint system**: save_checkpoint() called before risky operations
- **Error handling**: set_error() for recoverable errors, proper FAILED transitions
- **Stale session cleanup**: cleanup_stale_sessions() runs at startup
- **Exception handling**: KeyboardInterrupt and Exception save session state
- **Connection loss handling**: Sets ERROR state on reconnect failure

## Visual Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SESSION LIFECYCLE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   HAPPY PATH:                                                                │
│   ───────────                                                                │
│   [Workspace] → [Create] → [Active:Discover] → [Active:Strategy] →          │
│   [Active:Baseline] → [Active:Tuning×N] → [Archived] ✓                      │
│                                                                              │
│   PAUSE PATH:                                                                │
│   ──────────                                                                 │
│   [Active:*] → /stop → [Paused] → /resume → [Active:*] (continue)           │
│                                                                              │
│   FAILURE PATH:                                                              │
│   ────────────                                                               │
│   [Active:Tuning] → 3 misses → [Failed] → /retry? → [Active:Strategy]       │
│                                                                              │
│   ERROR PATH (NEW):                                                          │
│   ─────────────────                                                          │
│   [Active:*] → error → [Error] → /retry → [Active:*] (restart phase)        │
│                        └──────→ /abort → [Failed]                            │
│                                                                              │
│   ABANDON PATH (NEW):                                                        │
│   ───────────────────                                                        │
│   [Active:*] → (no activity 24h) → [Abandoned] → /resume → [Active:*]       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
