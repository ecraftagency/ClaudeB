"""
Workspace Management - IDE-like workspace for database optimization.

Concepts:
- Workspace = One database optimization project (like VS Code workspace)
- Session = One tuning strategy run (like a project within workspace)

Workspace Identity:
    A workspace is uniquely identified by: db_name + db_host + db_port
    This means:
    - postgres@192.168.1.1:5432 and postgres@192.168.1.1:6432 are DIFFERENT workspaces
    - Same database via direct connection vs PgBouncer = separate workspaces
    - Different users on same connection = SAME workspace (user is auth, not identity)

Directory Structure:
    ~/.pg_diagnose/workspaces/{db_name}_{db_host}_{db_port}/
    ├── workspace.json           # System info, proxy config, metadata
    ├── sessions/
    │   ├── BalancedTPS_01022026-050830/
    │   │   └── session.json
    │   └── WriteOptimize_01022026-060000/
    │       └── session.json
    └── exports/
        ├── report.md
        ├── postgresql.conf
        └── ansible.yml

Session States:
- ACTIVE: Currently running
- PAUSED: Interrupted, can resume
- ARCHIVED: Completed successfully (contributes to workspace export)
- FAILED: Did not achieve goal (excluded from export)
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum


WORKSPACES_DIR = Path.home() / ".pg_diagnose" / "workspaces"


class SessionState(str, Enum):
    """Session lifecycle states."""
    ACTIVE = "active"        # Currently running
    PAUSED = "paused"        # User interrupted, can resume
    ARCHIVED = "archived"    # Completed successfully
    FAILED = "failed"        # Max rounds without target, or unrecoverable error
    ERROR = "error"          # Recoverable error, can retry
    ABANDONED = "abandoned"  # Stale session (no activity for 24h+)


class SessionPhase(str, Enum):
    """Sub-states within ACTIVE session - tracks where we are in the workflow."""
    INITIALIZING = "initializing"      # Loading config, checking connection
    DISCOVERING = "discovering"        # Schema scan, system info collection
    STRATEGY_SELECTION = "strategy_selection"  # AI suggesting, user choosing
    BASELINE = "baseline"              # Running baseline benchmark
    TUNING = "tuning"                  # In tuning loop (rounds)
    EVALUATING = "evaluating"          # Checking if target hit
    COMPLETED = "completed"            # Finished (before archive/fail)


class WorkspaceState(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    ARCHIVED = "archived"


class InvalidStateTransition(Exception):
    """Raised when attempting an invalid state transition."""
    pass


class SessionStateMachine:
    """
    Centralized state management for session lifecycle.

    Valid transitions:
        ACTIVE → PAUSED, ARCHIVED, FAILED, ERROR
        PAUSED → ACTIVE, ARCHIVED, FAILED, ABANDONED
        ERROR → ACTIVE (retry), FAILED (give up)
        FAILED → ACTIVE (retry with different strategy)
        ARCHIVED → ACTIVE (re-run to improve)
        ABANDONED → ACTIVE (resume), FAILED (give up)
    """

    VALID_TRANSITIONS = {
        SessionState.ACTIVE: [
            SessionState.PAUSED,
            SessionState.ARCHIVED,
            SessionState.FAILED,
            SessionState.ERROR,
        ],
        SessionState.PAUSED: [
            SessionState.ACTIVE,
            SessionState.ARCHIVED,
            SessionState.FAILED,
            SessionState.ABANDONED,
        ],
        SessionState.ERROR: [
            SessionState.ACTIVE,  # Retry
            SessionState.FAILED,  # Give up
        ],
        SessionState.FAILED: [
            SessionState.ACTIVE,  # Retry with different approach
        ],
        SessionState.ARCHIVED: [
            SessionState.ACTIVE,  # Re-run to improve
        ],
        SessionState.ABANDONED: [
            SessionState.ACTIVE,  # Resume
            SessionState.FAILED,  # Give up
        ],
    }

    @classmethod
    def can_transition(cls, from_state: SessionState, to_state: SessionState) -> bool:
        """Check if transition is valid."""
        valid_targets = cls.VALID_TRANSITIONS.get(from_state, [])
        return to_state in valid_targets

    @classmethod
    def transition(cls, session: 'Session', to_state: SessionState,
                   reason: str = "", phase: SessionPhase = None) -> bool:
        """
        Perform state transition with validation.

        Args:
            session: Session to transition
            to_state: Target state
            reason: Reason for transition (stored in conclusion)
            phase: New phase (only for ACTIVE state)

        Returns:
            True if transition succeeded

        Raises:
            InvalidStateTransition: If transition is not valid
        """
        from_state = SessionState(session.data.state)

        if not cls.can_transition(from_state, to_state):
            raise InvalidStateTransition(
                f"Cannot transition from {from_state.value} to {to_state.value}"
            )

        # Update state
        session.data.state = to_state.value

        # Update phase if transitioning to ACTIVE
        if to_state == SessionState.ACTIVE and phase:
            session.data.phase = phase.value
        elif to_state != SessionState.ACTIVE:
            # Clear phase for non-active states
            session.data.phase = ""

        # Store reason
        if reason:
            if to_state in (SessionState.FAILED, SessionState.ERROR):
                session.data.conclusion = reason
            elif to_state == SessionState.ARCHIVED:
                session.data.conclusion = reason

        # Record transition in history
        if not hasattr(session.data, 'state_history'):
            session.data.state_history = []
        session.data.state_history.append({
            'from': from_state.value,
            'to': to_state.value,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        })

        session.save()
        return True

    @classmethod
    def get_available_transitions(cls, state: SessionState) -> List[SessionState]:
        """Get list of valid transitions from current state."""
        return cls.VALID_TRANSITIONS.get(state, [])


@dataclass
class ProxyConfig:
    """Proxy (pgcat/pgbouncer) configuration."""
    type: str = ""  # pgcat, pgbouncer, none
    host: str = ""
    port: int = 6432
    pool_mode: str = "transaction"
    pool_size: int = 100
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """System configuration snapshot."""
    # Host info
    hostname: str = ""
    os_version: str = ""
    kernel: str = ""

    # CPU
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0

    # Memory
    total_memory_gb: float = 0
    swap_gb: float = 0

    # Storage
    storage_type: str = ""  # nvme, ssd, hdd
    raid_config: str = ""   # raid0, raid10, single
    mount_points: List[Dict] = field(default_factory=list)  # [{mount, device, fs, size}]

    # Network
    network_config: Dict[str, Any] = field(default_factory=dict)

    # PostgreSQL
    pg_version: str = ""
    pg_data_dir: str = ""
    pg_wal_dir: str = ""
    pg_config: Dict[str, str] = field(default_factory=dict)


@dataclass
class SessionData:
    """One tuning session within a workspace."""
    # Identity
    name: str = ""
    created_at: str = ""
    updated_at: str = ""
    state: str = "active"  # active, paused, archived, failed, error, abandoned
    phase: str = "initializing"  # Sub-state within ACTIVE (see SessionPhase)

    # Strategy
    strategy_id: str = ""
    strategy_name: str = ""
    strategy_rationale: str = ""
    target_tps: float = 0

    # First sight
    first_sight: str = ""
    bottleneck: str = ""
    confidence: float = 0

    # Benchmark data
    baseline: Optional[Dict] = None
    rounds: List[Dict] = field(default_factory=list)

    # Progress
    current_round: int = 0
    best_tps: float = 0
    best_round: int = -1

    # Result
    conclusion: str = ""
    sweet_spot_changes: List[Dict] = field(default_factory=list)

    # Metrics (sampled - last 10 per round to save space)
    metrics_samples: Dict[str, List[Dict]] = field(default_factory=dict)

    # Error tracking (for ERROR state)
    error_info: Optional[Dict] = None  # {type, message, phase, recoverable, timestamp}

    # State transition history
    state_history: List[Dict] = field(default_factory=list)  # [{from, to, reason, timestamp}]

    # Checkpoint for crash recovery
    last_checkpoint: Optional[Dict] = None  # {phase, round, timestamp, data}


@dataclass
class WorkspaceData:
    """Workspace metadata and configuration."""
    # Identity
    name: str = ""
    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""

    # SSH (for remote operations)
    ssh_host: str = ""
    ssh_user: str = "ubuntu"
    ssh_port: int = 22

    # State
    state: str = "new"
    created_at: str = ""
    updated_at: str = ""
    last_session: str = ""  # Last active session name

    # System snapshot
    system: Dict[str, Any] = field(default_factory=dict)

    # Proxy config
    proxy: Dict[str, Any] = field(default_factory=dict)

    # Schema info
    schema_overview: str = ""
    tables: List[Dict] = field(default_factory=list)


class Session:
    """
    One tuning session - manages a single strategy run.
    """

    def __init__(self, workspace_path: Path, name: str, data: SessionData = None):
        self.workspace_path = workspace_path
        self.name = name
        self.path = workspace_path / "sessions" / name
        self._data = data

    @property
    def data(self) -> SessionData:
        if self._data is None:
            self.load()
        return self._data

    @property
    def state(self) -> SessionState:
        return SessionState(self.data.state)

    @property
    def is_active(self) -> bool:
        return self.state == SessionState.ACTIVE

    @property
    def is_archived(self) -> bool:
        return self.state == SessionState.ARCHIVED

    def load(self):
        """Load session from disk."""
        session_file = self.path / "session.json"
        if session_file.exists():
            with open(session_file) as f:
                self._data = SessionData(**json.load(f))
        else:
            self._data = SessionData(name=self.name, created_at=datetime.now().isoformat())

    def save(self):
        """Save session to disk."""
        self.path.mkdir(parents=True, exist_ok=True)
        self.data.updated_at = datetime.now().isoformat()

        with open(self.path / "session.json", 'w') as f:
            json.dump(asdict(self.data), f, indent=2, default=str)

    def set_strategy(self, strategy_id: str, name: str, rationale: str = "", target_tps: float = 0):
        """Set strategy info."""
        self.data.strategy_id = strategy_id
        self.data.strategy_name = name
        self.data.strategy_rationale = rationale
        self.data.target_tps = target_tps
        self.save()

    def set_first_sight(self, analysis: str, bottleneck: str, confidence: float):
        """Set first sight analysis."""
        self.data.first_sight = analysis
        self.data.bottleneck = bottleneck
        self.data.confidence = confidence
        self.save()

    def save_baseline(self, tps: float, latency_avg: float = 0, latency_p99: float = 0,
                     metrics: Dict = None, raw_output: str = "", ai_analysis: str = ""):
        """Save baseline benchmark result."""
        self.data.baseline = {
            'round_num': 0,
            'tps': tps,
            'latency_avg_ms': latency_avg,
            'latency_p99_ms': latency_p99,
            'raw_output': raw_output,
            'ai_analysis': ai_analysis,
            'timestamp': datetime.now().isoformat(),
        }

        # Store sampled metrics (last 10 samples)
        if metrics:
            self._store_metrics_sample('baseline', metrics)

        self.data.best_tps = tps
        self.data.best_round = 0
        self.save()

    def save_round(self, round_num: int, tps: float, changes: List[Dict],
                  latency_avg: float = 0, latency_p99: float = 0,
                  metrics: Dict = None, raw_output: str = "",
                  ai_analysis: str = "", ai_recommendations: List[str] = None):
        """Save round benchmark result."""
        round_data = {
            'round_num': round_num,
            'tps': tps,
            'latency_avg_ms': latency_avg,
            'latency_p99_ms': latency_p99,
            'changes': changes,
            'raw_output': raw_output,
            'ai_analysis': ai_analysis,
            'ai_recommendations': ai_recommendations or [],
            'timestamp': datetime.now().isoformat(),
        }

        # Update or append
        while len(self.data.rounds) < round_num:
            self.data.rounds.append({})
        if round_num == len(self.data.rounds) + 1:
            self.data.rounds.append(round_data)
        else:
            self.data.rounds[round_num - 1] = round_data

        # Store sampled metrics
        if metrics:
            self._store_metrics_sample(f'round_{round_num}', metrics)

        self.data.current_round = round_num

        # Track best
        if tps > self.data.best_tps:
            self.data.best_tps = tps
            self.data.best_round = round_num

        self.save()

    def _store_metrics_sample(self, key: str, metrics: Dict, max_samples: int = 10):
        """Store last N metrics samples to save space."""
        sampled = {}
        for metric_name, samples in metrics.items():
            if isinstance(samples, list) and len(samples) > max_samples:
                sampled[metric_name] = samples[-max_samples:]
            else:
                sampled[metric_name] = samples
        self.data.metrics_samples[key] = sampled

    # === Phase Management ===

    @property
    def phase(self) -> SessionPhase:
        """Get current phase."""
        try:
            return SessionPhase(self.data.phase)
        except ValueError:
            return SessionPhase.INITIALIZING

    def set_phase(self, phase: SessionPhase):
        """Set current phase within ACTIVE state."""
        self.data.phase = phase.value
        self.save()

    # === Checkpoint Management ===

    def save_checkpoint(self, description: str = ""):
        """
        Save checkpoint for crash recovery.

        Call this before risky operations (benchmarks, AI calls, etc.)
        """
        self.data.last_checkpoint = {
            'phase': self.data.phase,
            'round': self.data.current_round,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'best_tps': self.data.best_tps,
            'rounds_count': len(self.data.rounds),
        }
        self.save()

    def get_checkpoint(self) -> Optional[Dict]:
        """Get last checkpoint for recovery."""
        return self.data.last_checkpoint

    # === Error Management ===

    def set_error(self, error_type: str, message: str, recoverable: bool = True):
        """
        Record error and transition to ERROR state.

        Args:
            error_type: Category of error (benchmark_failed, connection_lost, ai_failed)
            message: Human-readable error message
            recoverable: Whether user can retry
        """
        self.data.error_info = {
            'type': error_type,
            'message': message,
            'phase': self.data.phase,
            'round': self.data.current_round,
            'recoverable': recoverable,
            'timestamp': datetime.now().isoformat(),
        }

        # Use state machine for transition
        try:
            SessionStateMachine.transition(
                self, SessionState.ERROR,
                reason=f"{error_type}: {message}"
            )
        except InvalidStateTransition:
            # Fallback: just set state directly
            self.data.state = SessionState.ERROR.value
            self.save()

    def clear_error(self):
        """Clear error info when retrying."""
        self.data.error_info = None
        self.save()

    # === State Transitions (using StateMachine) ===

    def archive(self, conclusion: str = ""):
        """Mark session as archived (successful)."""
        # Collect sweet spot changes
        if self.data.best_round == 0:
            self.data.sweet_spot_changes = []
        else:
            all_changes = []
            for r in self.data.rounds[:self.data.best_round]:
                all_changes.extend(r.get('changes', []))
            self.data.sweet_spot_changes = all_changes

        try:
            SessionStateMachine.transition(
                self, SessionState.ARCHIVED,
                reason=conclusion,
                phase=SessionPhase.COMPLETED
            )
        except InvalidStateTransition:
            # Fallback for backwards compatibility
            self.data.state = SessionState.ARCHIVED.value
            self.data.conclusion = conclusion
            self.save()

    def fail(self, reason: str = ""):
        """Mark session as failed."""
        try:
            SessionStateMachine.transition(
                self, SessionState.FAILED,
                reason=reason
            )
        except InvalidStateTransition:
            # Fallback
            self.data.state = SessionState.FAILED.value
            self.data.conclusion = reason
            self.save()

    def pause(self):
        """Pause session for later resumption."""
        try:
            SessionStateMachine.transition(
                self, SessionState.PAUSED,
                reason="User paused session"
            )
        except InvalidStateTransition:
            self.data.state = SessionState.PAUSED.value
            self.save()

    def resume(self, to_phase: SessionPhase = None):
        """Resume paused/error/failed session."""
        target_phase = to_phase or SessionPhase(self.data.phase) if self.data.phase else SessionPhase.TUNING

        try:
            SessionStateMachine.transition(
                self, SessionState.ACTIVE,
                reason="User resumed session",
                phase=target_phase
            )
        except InvalidStateTransition:
            self.data.state = SessionState.ACTIVE.value
            if to_phase:
                self.data.phase = to_phase.value
            self.save()

        # Clear error if resuming from ERROR
        if self.data.error_info:
            self.clear_error()

    def retry(self, from_phase: SessionPhase = None):
        """
        Retry from ERROR or FAILED state.

        Args:
            from_phase: Phase to restart from (defaults to last checkpoint phase)
        """
        if self.state not in (SessionState.ERROR, SessionState.FAILED):
            raise InvalidStateTransition(f"Cannot retry from {self.state.value} state")

        # Determine phase to restart from
        if from_phase:
            restart_phase = from_phase
        elif self.data.last_checkpoint:
            restart_phase = SessionPhase(self.data.last_checkpoint['phase'])
        else:
            restart_phase = SessionPhase.TUNING

        SessionStateMachine.transition(
            self, SessionState.ACTIVE,
            reason="User retried session",
            phase=restart_phase
        )

        # Clear error info
        self.clear_error()

    def abandon(self, reason: str = "Session abandoned"):
        """Mark session as abandoned (stale)."""
        try:
            SessionStateMachine.transition(
                self, SessionState.ABANDONED,
                reason=reason
            )
        except InvalidStateTransition:
            self.data.state = SessionState.ABANDONED.value
            self.data.conclusion = reason
            self.save()

    def can_retry(self) -> bool:
        """Check if session can be retried."""
        return self.state in (SessionState.ERROR, SessionState.FAILED, SessionState.ABANDONED)

    def can_resume(self) -> bool:
        """Check if session can be resumed."""
        return self.state in (SessionState.PAUSED, SessionState.ERROR, SessionState.ABANDONED)

    def export_markdown(self) -> str:
        """Export session to markdown."""
        s = self.data
        baseline_tps = s.baseline.get('tps', 0) if s.baseline else 0
        improvement = ((s.best_tps - baseline_tps) / baseline_tps * 100) if baseline_tps > 0 else 0

        md = f"""# Session: {s.strategy_name}

**Name:** {s.name}
**State:** {s.state.upper()}
**Created:** {s.created_at}

## Summary

| Metric | Value |
|--------|-------|
| Strategy | {s.strategy_name} |
| Target TPS | {s.target_tps:,.0f} |
| Baseline TPS | {baseline_tps:,.0f} |
| Best TPS | {s.best_tps:,.0f} |
| Improvement | {improvement:+.1f}% |
| Rounds | {len(s.rounds)} |
| Best Round | {s.best_round} |

## Analysis

**Bottleneck:** {s.bottleneck} ({s.confidence:.0%} confidence)

{s.first_sight}

## TPS History

| Round | TPS | Change |
|-------|-----|--------|
| Baseline | {baseline_tps:,.0f} | - |
"""
        for r in s.rounds:
            prev_tps = baseline_tps if r['round_num'] == 1 else s.rounds[r['round_num']-2].get('tps', baseline_tps)
            change = ((r['tps'] - prev_tps) / prev_tps * 100) if prev_tps > 0 else 0
            md += f"| Round {r['round_num']} | {r['tps']:,.0f} | {change:+.1f}% |\n"

        if s.sweet_spot_changes:
            md += "\n## Sweet Spot Configuration\n\n"
            for c in s.sweet_spot_changes:
                md += f"### {c.get('name', 'Change')}\n\n"
                for cmd in c.get('pg_configs', []):
                    md += f"```sql\n{cmd}\n```\n"
                if c.get('os_command'):
                    md += f"```bash\n{c['os_command']}\n```\n"

        if s.conclusion:
            md += f"\n## Conclusion\n\n{s.conclusion}\n"

        return md

    def get_status(self) -> Dict[str, Any]:
        """Get session status for status line."""
        return {
            'name': self.data.strategy_name or self.name,
            'state': self.data.state,
            'round': self.data.current_round,
            'tps': self.data.best_tps,
            'target': self.data.target_tps,
        }


class Workspace:
    """
    Workspace - manages a database optimization project.

    Like a VS Code workspace, contains multiple sessions (projects).
    """

    def __init__(self, path: Path, data: WorkspaceData = None):
        self.path = path
        self._data = data
        self._current_session: Optional[Session] = None

    @classmethod
    def create(cls, db_host: str, db_port: int, db_name: str, db_user: str,
               ssh_host: str = "", ssh_user: str = "ubuntu") -> 'Workspace':
        """Create a new workspace for a database connection.

        Workspace identity is: db_name + db_host + db_port
        This ensures different ports (e.g., 5432 direct vs 6432 PgBouncer)
        are treated as separate workspaces.
        """
        # Generate workspace name - include port for unique identity
        safe_host = db_host.replace('.', '-').replace(':', '-')
        name = f"{db_name}_{safe_host}_{db_port}"

        path = WORKSPACES_DIR / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "sessions").mkdir(exist_ok=True)
        (path / "exports").mkdir(exist_ok=True)

        data = WorkspaceData(
            name=name,
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            state=WorkspaceState.NEW.value,
            created_at=datetime.now().isoformat(),
        )

        ws = cls(path, data)
        ws.save()
        return ws

    @classmethod
    def open(cls, path: Path) -> Optional['Workspace']:
        """Open existing workspace."""
        ws_file = path / "workspace.json"
        if not ws_file.exists():
            return None

        ws = cls(path)
        ws.load()
        return ws

    @classmethod
    def find_for_database(cls, db_host: str, db_port: int, db_name: str) -> Optional['Workspace']:
        """Find existing workspace for a database connection.

        Workspace identity is: db_name + db_host + db_port
        """
        safe_host = db_host.replace('.', '-').replace(':', '-')
        expected_name = f"{db_name}_{safe_host}_{db_port}"

        path = WORKSPACES_DIR / expected_name
        if path.exists():
            return cls.open(path)
        return None

    @classmethod
    def list_all(cls) -> List[Dict[str, Any]]:
        """List all workspaces with session statistics."""
        workspaces = []

        if not WORKSPACES_DIR.exists():
            return workspaces

        for path in WORKSPACES_DIR.iterdir():
            if not path.is_dir():
                continue
            ws_file = path / "workspace.json"
            if not ws_file.exists():
                continue

            try:
                with open(ws_file) as f:
                    data = json.load(f)

                # Count sessions by state
                sessions_dir = path / "sessions"
                session_count = 0
                active_count = 0
                paused_count = 0
                archived_count = 0

                if sessions_dir.exists():
                    for sess_path in sessions_dir.iterdir():
                        if not sess_path.is_dir():
                            continue
                        sess_file = sess_path / "session.json"
                        if sess_file.exists():
                            session_count += 1
                            try:
                                with open(sess_file) as sf:
                                    sess_data = json.load(sf)
                                    state = sess_data.get('state', '')
                                    if state == 'active':
                                        active_count += 1
                                    elif state == 'paused':
                                        paused_count += 1
                                    elif state == 'archived':
                                        archived_count += 1
                            except:
                                pass

                workspaces.append({
                    'name': data.get('name', path.name),
                    'path': str(path),
                    'db_host': data.get('db_host', ''),
                    'db_port': data.get('db_port', 5432),
                    'db_name': data.get('db_name', ''),
                    'state': data.get('state', 'unknown'),
                    'sessions': session_count,
                    'session_count': session_count,  # Alias for compatibility
                    'active_sessions': active_count,
                    'paused_sessions': paused_count,
                    'archived_sessions': archived_count,
                    'updated': data.get('updated_at', ''),
                    'updated_at': data.get('updated_at', ''),  # Alias for compatibility
                    'last_session': data.get('last_session', ''),
                })
            except Exception:
                continue

        workspaces.sort(key=lambda x: x.get('updated', ''), reverse=True)
        return workspaces

    @property
    def data(self) -> WorkspaceData:
        if self._data is None:
            self.load()
        return self._data

    @property
    def current_session(self) -> Optional[Session]:
        return self._current_session

    @property
    def name(self) -> str:
        return self.data.name

    def load(self):
        """Load workspace from disk."""
        with open(self.path / "workspace.json") as f:
            self._data = WorkspaceData(**json.load(f))

    def save(self):
        """Save workspace to disk."""
        self.data.updated_at = datetime.now().isoformat()
        self.data.state = WorkspaceState.ACTIVE.value

        with open(self.path / "workspace.json", 'w') as f:
            json.dump(asdict(self.data), f, indent=2, default=str)

    def set_system_snapshot(self, snapshot: Dict[str, Any]):
        """Set system configuration snapshot."""
        self.data.system = snapshot
        self.save()

    def set_proxy_config(self, config: Dict[str, Any]):
        """Set proxy configuration."""
        self.data.proxy = config
        self.save()

    def set_schema_info(self, overview: str, tables: List[Dict]):
        """Set schema information."""
        self.data.schema_overview = overview
        self.data.tables = tables
        self.save()

    # === Session Management ===

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions in workspace."""
        sessions = []
        sessions_dir = self.path / "sessions"

        if not sessions_dir.exists():
            return sessions

        for path in sessions_dir.iterdir():
            if not path.is_dir():
                continue
            session_file = path / "session.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file) as f:
                    data = json.load(f)

                baseline_tps = data.get('baseline', {}).get('tps', 0) if data.get('baseline') else 0
                best_tps = data.get('best_tps', 0)

                sessions.append({
                    'name': path.name,
                    'strategy': data.get('strategy_name', ''),
                    'state': data.get('state', 'unknown'),
                    'baseline_tps': baseline_tps,
                    'best_tps': best_tps,
                    'target_tps': data.get('target_tps', 0),
                    'rounds': len(data.get('rounds', [])),
                    'updated_at': data.get('updated_at', ''),
                })
            except Exception:
                continue

        sessions.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return sessions

    def create_session(self, strategy_name: str, strategy_id: str = "",
                      target_tps: float = 0, rationale: str = "") -> Session:
        """Create a new session for a strategy."""
        # Generate session name
        ts = datetime.now().strftime("%m%d%Y-%H%M%S")
        safe_name = "".join(c for c in strategy_name if c.isalnum() or c in "-_")
        name = f"{safe_name}_{ts}"

        session = Session(self.path, name)
        session.data.name = name
        session.data.created_at = datetime.now().isoformat()
        session.data.state = SessionState.ACTIVE.value
        session.set_strategy(strategy_id or safe_name, strategy_name, rationale, target_tps)

        self._current_session = session
        self.data.last_session = name
        self.save()

        return session

    def open_session(self, name: str) -> Optional[Session]:
        """Open an existing session."""
        session_path = self.path / "sessions" / name
        if not (session_path / "session.json").exists():
            return None

        session = Session(self.path, name)
        session.load()

        self._current_session = session
        self.data.last_session = name
        self.save()

        return session

    def close_session(self):
        """Close current session, return to workspace level."""
        if self._current_session:
            self._current_session.save()
        self._current_session = None

    def get_archived_sessions(self) -> List[Session]:
        """Get all archived sessions for export."""
        archived = []
        for info in self.list_sessions():
            if info['state'] == SessionState.ARCHIVED.value:
                session = Session(self.path, info['name'])
                session.load()
                archived.append(session)
        return archived

    # === Export ===

    def export_session_markdown(self, session_name: str = None) -> str:
        """Export a session to markdown."""
        if session_name:
            session = Session(self.path, session_name)
            session.load()
        elif self._current_session:
            session = self._current_session
        else:
            raise ValueError("No session specified or active")

        return session.export_markdown()

    def get_export_context(self) -> Dict[str, Any]:
        """Get context for LLM workspace export composition."""
        archived = self.get_archived_sessions()

        context = {
            'workspace': {
                'name': self.data.name,
                'db_host': self.data.db_host,
                'db_name': self.data.db_name,
            },
            'system': self.data.system,
            'proxy': self.data.proxy,
            'sessions': [],
        }

        for session in archived:
            session_ctx = {
                'name': session.data.strategy_name,
                'best_tps': session.data.best_tps,
                'baseline_tps': session.data.baseline.get('tps', 0) if session.data.baseline else 0,
                'improvement_pct': 0,
                'sweet_spot_changes': session.data.sweet_spot_changes,
            }

            if session_ctx['baseline_tps'] > 0:
                session_ctx['improvement_pct'] = (
                    (session_ctx['best_tps'] - session_ctx['baseline_tps'])
                    / session_ctx['baseline_tps'] * 100
                )

            context['sessions'].append(session_ctx)

        return context

    def get_status(self) -> Dict[str, Any]:
        """Get workspace status for status line."""
        session_status = None
        if self._current_session:
            session_status = self._current_session.get_status()

        return {
            'workspace': f"{self.data.db_name}@{self.data.db_host}:{self.data.db_port}",
            'state': self.data.state,
            'session': session_status,
        }

    def delete(self):
        """Delete workspace and all sessions."""
        if self.path.exists():
            shutil.rmtree(self.path)


class WorkspaceManager:
    """
    Manages all workspaces - entry point for the tool.
    """

    def __init__(self):
        WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
        self._current: Optional[Workspace] = None

    @property
    def current(self) -> Optional[Workspace]:
        return self._current

    @property
    def current_workspace(self) -> Optional[Workspace]:
        """Alias for current."""
        return self._current

    @property
    def current_session(self) -> Optional[Session]:
        """Get current session from workspace."""
        if self._current:
            return self._current.current_session
        return None

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """List all available workspaces."""
        return Workspace.list_all()

    def find_or_create(self, db_host: str, db_port: int, db_name: str, db_user: str,
                       ssh_host: str = "", ssh_user: str = "ubuntu") -> Workspace:
        """Find existing workspace for database connection or create new one.

        Workspace identity is: db_name + db_host + db_port
        Same database on different ports = different workspaces.
        """
        ws = Workspace.find_for_database(db_host, db_port, db_name)
        if ws:
            self._current = ws
            return ws

        ws = Workspace.create(db_host, db_port, db_name, db_user, ssh_host, ssh_user)
        self._current = ws
        return ws

    def open(self, name: str) -> Optional[Workspace]:
        """Open workspace by name."""
        path = WORKSPACES_DIR / name
        ws = Workspace.open(path)
        if ws:
            self._current = ws
        return ws

    def close(self):
        """Close current workspace."""
        if self._current:
            if self._current.current_session:
                self._current.close_session()
            self._current.save()
        self._current = None

    def get_status(self) -> Dict[str, Any]:
        """Get current status for status line."""
        if not self._current:
            return {'workspace': None, 'session': None}

        status = self._current.get_status()

        # Add session counts
        sessions = self._current.list_sessions()
        status['session_count'] = len(sessions)
        status['archived_count'] = sum(1 for s in sessions if s['state'] == 'archived')
        status['workspace_path'] = str(self._current.path)

        return status

    # === Session Management Shortcuts ===

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List sessions in current workspace."""
        if not self._current:
            return []
        return self._current.list_sessions()

    def create_session(self, strategy_name: str, **kwargs) -> Optional[Session]:
        """Create new session in current workspace."""
        if not self._current:
            raise ValueError("No workspace open")
        return self._current.create_session(strategy_name, **kwargs)

    def save_session(self):
        """Save current session."""
        if self._current and self._current.current_session:
            self._current.current_session.save()

    def archive_session(self, conclusion: str = ""):
        """Archive current session."""
        if not self._current or not self._current.current_session:
            raise ValueError("No active session")
        self._current.current_session.archive(conclusion)
        self._current.close_session()

    def pause_session(self):
        """Pause current session."""
        if not self._current or not self._current.current_session:
            raise ValueError("No active session")
        self._current.current_session.pause()
        self._current.close_session()

    def resume_session(self, session_name: str):
        """Resume a paused session."""
        if not self._current:
            raise ValueError("No workspace open")

        session = self._current.open_session(session_name)
        if not session:
            raise ValueError(f"Session not found: {session_name}")
        if session.state != SessionState.PAUSED:
            raise ValueError(f"Session is not paused: {session.state.value}")

        session.resume()

    def switch_session(self, session_name: str):
        """Switch to a different session."""
        if not self._current:
            raise ValueError("No workspace open")

        # Close current if any
        if self._current.current_session:
            self._current.current_session.save()
            self._current.close_session()

        # Open target session
        session = self._current.open_session(session_name)
        if not session:
            raise ValueError(f"Session not found: {session_name}")

    def export_session(self, session_name: str = None) -> str:
        """
        Export session recommendations.

        If session_name is None, exports current session.
        Returns path to export file.
        """
        if not self._current:
            raise ValueError("No workspace open")

        # Determine which session to export
        if session_name:
            session = Session(self._current.path, session_name)
            session.load()
        elif self._current.current_session:
            session = self._current.current_session
        else:
            raise ValueError("No session specified and no active session")

        # Only allow export of archived sessions (or force for active)
        if session.state not in (SessionState.ARCHIVED, SessionState.ACTIVE):
            raise ValueError(f"Cannot export {session.state.value} session. Archive first.")

        # Generate markdown export
        md_content = session.export_markdown()

        # Save to exports directory
        exports_dir = self._current.path / "exports"
        exports_dir.mkdir(exist_ok=True)

        export_path = exports_dir / f"{session.name}.md"
        with open(export_path, 'w') as f:
            f.write(md_content)

        return str(export_path)

    # === Cleanup and Maintenance ===

    def cleanup_stale_sessions(self, max_age_hours: int = 24) -> List[str]:
        """
        Mark stale ACTIVE sessions as ABANDONED.

        A session is considered stale if:
        - State is ACTIVE
        - No updates for more than max_age_hours

        Args:
            max_age_hours: Hours of inactivity before marking as abandoned

        Returns:
            List of session names that were marked as abandoned
        """
        if not self._current:
            return []

        abandoned = []
        now = datetime.now()

        for session_info in self._current.list_sessions():
            if session_info['state'] != 'active':
                continue

            # Check last update time
            updated_at = session_info.get('updated_at', '')
            if not updated_at:
                continue

            try:
                last_update = datetime.fromisoformat(updated_at)
                age_hours = (now - last_update).total_seconds() / 3600

                if age_hours > max_age_hours:
                    session = self._current.open_session(session_info['name'])
                    if session:
                        session.abandon(f"No activity for {age_hours:.1f} hours")
                        abandoned.append(session_info['name'])
                        self._current.close_session()
            except (ValueError, TypeError):
                continue

        return abandoned

    def get_recoverable_sessions(self) -> List[Dict[str, Any]]:
        """
        Get sessions that can be retried or resumed.

        Returns:
            List of session info dicts for ERROR, FAILED, PAUSED, ABANDONED sessions
        """
        if not self._current:
            return []

        recoverable_states = {'error', 'failed', 'paused', 'abandoned'}
        sessions = self._current.list_sessions()

        return [s for s in sessions if s['state'] in recoverable_states]

    def retry_session(self, session_name: str, from_phase: SessionPhase = None) -> Optional[Session]:
        """
        Retry a failed/error session.

        Args:
            session_name: Name of session to retry
            from_phase: Phase to restart from (optional)

        Returns:
            Session if retry successful, None otherwise
        """
        if not self._current:
            raise ValueError("No workspace open")

        session = self._current.open_session(session_name)
        if not session:
            raise ValueError(f"Session not found: {session_name}")

        if not session.can_retry():
            raise ValueError(f"Session cannot be retried (state: {session.state.value})")

        session.retry(from_phase)
        return session


# === Display Functions ===

def display_workspaces(workspaces: List[Dict], console=None):
    """Display workspaces list."""
    if not workspaces:
        print("No workspaces found.")
        return

    try:
        from rich.table import Table
        from rich import box

        if console:
            table = Table(title="Workspaces", box=box.ROUNDED)
            table.add_column("#", style="dim", width=3)
            table.add_column("Database", style="cyan")
            table.add_column("Host:Port", style="green")
            table.add_column("Sessions", style="magenta")
            table.add_column("Last Updated", style="dim")

            for i, ws in enumerate(workspaces, 1):
                updated = ws.get('updated_at', '')[:16] if ws.get('updated_at') else ''
                host_port = f"{ws['db_host']}:{ws.get('db_port', 5432)}"
                table.add_row(
                    str(i),
                    ws['db_name'],
                    host_port,
                    str(ws['session_count']),
                    updated,
                )

            console.print(table)
            return
    except ImportError:
        pass

    print("\nWorkspaces:")
    for i, ws in enumerate(workspaces, 1):
        port = ws.get('db_port', 5432)
        print(f"  [{i}] {ws['db_name']}@{ws['db_host']}:{port} ({ws['session_count']} sessions)")


def display_sessions(sessions: List[Dict], console=None):
    """Display sessions list."""
    if not sessions:
        print("No sessions found.")
        return

    try:
        from rich.table import Table
        from rich import box

        if console:
            table = Table(title="Sessions", box=box.ROUNDED)
            table.add_column("#", style="dim", width=3)
            table.add_column("Strategy", style="cyan")
            table.add_column("State", style="yellow")
            table.add_column("TPS", style="magenta")
            table.add_column("Rounds", style="dim")

            for i, s in enumerate(sessions, 1):
                state_map = {
                    'active': '[yellow]ACTIVE[/]',
                    'paused': '[blue]PAUSED[/]',
                    'archived': '[green]ARCHIVED[/]',
                    'failed': '[red]FAILED[/]',
                    'error': '[red]ERROR[/]',
                    'abandoned': '[dim]ABANDONED[/]',
                }
                state = state_map.get(s['state'], s['state'])

                tps_str = f"{s['best_tps']:.0f}"
                if s['target_tps'] > 0:
                    if s['best_tps'] >= s['target_tps']:
                        tps_str += " [green]✓[/]"

                table.add_row(
                    str(i),
                    s['strategy'][:25],
                    state,
                    tps_str,
                    str(s['rounds']),
                )

            console.print(table)
            return
    except ImportError:
        pass

    print("\nSessions:")
    for i, s in enumerate(sessions, 1):
        state = s['state'].upper()
        print(f"  [{i}] {s['strategy'][:25]:25} [{state:8}] {s['best_tps']:.0f} TPS")


def format_status_line(status: Dict[str, Any]) -> str:
    """Format status line for display."""
    if not status.get('workspace'):
        return "[dim]No workspace[/]"

    parts = [f"[cyan]{status['workspace']}[/]"]

    if status.get('session'):
        s = status['session']
        parts.append(f"[yellow]{s['name']}[/]")

        if s.get('round', 0) > 0:
            parts.append(f"R{s['round']}")

        if s.get('tps', 0) > 0:
            tps_part = f"{s['tps']:.0f}"
            if s.get('target', 0) > 0:
                tps_part += f"/{s['target']:.0f}"
            parts.append(f"[magenta]{tps_part} TPS[/]")

    return " | ".join(parts)


def display_workspaces(workspaces: List[Dict[str, Any]], console=None):
    """
    Display list of workspaces in a formatted table.

    Args:
        workspaces: List of workspace info dicts from WorkspaceManager.list_workspaces()
        console: Rich console for formatting (optional)
    """
    if not workspaces:
        if console:
            console.print("[dim]No workspaces found[/]")
        else:
            print("No workspaces found")
        return

    # Try Rich table if available
    try:
        from rich.table import Table
        from rich import box

        if console:
            table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Database", style="green")
            table.add_column("Host:Port", style="dim")
            table.add_column("Sessions", justify="center")
            table.add_column("Last Active", style="dim")

            for i, ws in enumerate(workspaces, 1):
                # Format last updated
                updated = ws.get('updated', '')
                if updated:
                    try:
                        dt = datetime.fromisoformat(updated)
                        updated = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        updated = updated[:16]

                # Count sessions by state
                session_count = ws.get('sessions', 0)
                active_count = ws.get('active_sessions', 0)
                paused_count = ws.get('paused_sessions', 0)

                if active_count > 0:
                    sessions_str = f"[yellow]{active_count} active[/]"
                elif paused_count > 0:
                    sessions_str = f"[blue]{paused_count} paused[/]"
                else:
                    sessions_str = str(session_count)

                host_port = f"{ws.get('db_host', '?')}:{ws.get('db_port', 5432)}"
                table.add_row(
                    str(i),
                    ws.get('db_name', 'unknown'),
                    host_port,
                    sessions_str,
                    updated,
                )

            console.print(table)
            return
    except ImportError:
        pass

    # Fallback: plain text
    print()
    print(f"{'#':>3}  {'Database':<15} {'Host:Port':<20} {'Sessions':>8}  {'Last Active':<16}")
    print("-" * 65)

    for i, ws in enumerate(workspaces, 1):
        updated = ws.get('updated', '')[:16]
        host_port = f"{ws.get('db_host', '?')}:{ws.get('db_port', 5432)}"
        print(f"{i:>3}  {ws.get('db_name', '?'):<15} {host_port:<20} {ws.get('sessions', 0):>8}  {updated:<16}")
