"""
Mock snapshot components for testing.

Provides MockSnapshotCapture and MockSnapshotRestore that return
realistic data without requiring database connections.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

from pg_diagnose.snapshot.models import (
    SnapshotData,
    RestoreResult,
    RestorePreview,
    SettingChange,
    SnapshotDiff,
    requires_restart,
)
from .golden_data import SNAPSHOT_TEST_DATA


class MockSnapshotCapture:
    """Mock snapshot capture for testing.

    Returns realistic PostgreSQL configuration from SNAPSHOT_TEST_DATA.
    Alternates between initial and tuned states based on capture count.
    """

    def __init__(self, scenario: str = "balanced_tps"):
        self.scenario = scenario
        self.capture_count = 0
        self._forced_state: Optional[str] = None  # "initial" or "tuned"

    def force_state(self, state: str) -> None:
        """Force capture to return a specific state.

        Args:
            state: "initial" or "tuned"
        """
        self._forced_state = state

    def capture(
        self,
        name: str,
        trigger: str = "manual",
        session_name: str = "",
        round_num: int = 0,
        last_tps: Optional[float] = None,
        last_latency: Optional[float] = None,
    ) -> SnapshotData:
        """Return mock snapshot data.

        First capture returns initial state, subsequent captures return tuned state.
        Use force_state() to override this behavior.
        """
        self.capture_count += 1

        # Determine which state to return
        if self._forced_state == "initial":
            is_initial = True
        elif self._forced_state == "tuned":
            is_initial = False
        else:
            # Default: first capture is initial, rest are tuned
            is_initial = (self.capture_count == 1)

        if is_initial:
            return SnapshotData(
                id=str(uuid.uuid4()),
                name=name,
                created_at=datetime.now().isoformat(),
                pg_settings=SNAPSHOT_TEST_DATA["initial_pg_settings"].copy(),
                pg_auto_conf=SNAPSHOT_TEST_DATA["initial_auto_conf"],
                pg_conf_hash=SNAPSHOT_TEST_DATA["initial_conf_hash"],
                last_tps=last_tps,
                last_latency=last_latency,
                session_name=session_name,
                round_num=round_num,
                trigger=trigger,
            )
        else:
            return SnapshotData(
                id=str(uuid.uuid4()),
                name=name,
                created_at=datetime.now().isoformat(),
                pg_settings=SNAPSHOT_TEST_DATA["tuned_pg_settings"].copy(),
                pg_auto_conf=SNAPSHOT_TEST_DATA["tuned_auto_conf"],
                pg_conf_hash=SNAPSHOT_TEST_DATA["tuned_conf_hash"],
                last_tps=last_tps or 6000.0,
                last_latency=last_latency or 5.3,
                session_name=session_name,
                round_num=round_num,
                trigger=trigger,
            )


class MockSnapshotRestore:
    """Mock snapshot restore for testing.

    Simulates restore operations without modifying any actual database.
    Tracks restore calls for verification in tests.
    """

    def __init__(self):
        self.restore_count = 0
        self.last_restored: Optional[str] = None
        self.restore_history: List[str] = []
        self._should_fail: bool = False
        self._fail_message: str = ""

    def force_failure(self, message: str = "Mock restore failure") -> None:
        """Force the next restore to fail."""
        self._should_fail = True
        self._fail_message = message

    def restore(
        self,
        snapshot: SnapshotData,
        dry_run: bool = False,
    ) -> RestoreResult:
        """Mock restore operation.

        Args:
            snapshot: The snapshot to restore to
            dry_run: If True, only preview changes

        Returns:
            RestoreResult with mock data
        """
        # Check for forced failure
        if self._should_fail:
            self._should_fail = False
            return RestoreResult.failure(self._fail_message)

        # Generate preview
        preview = self.preview(snapshot)

        if dry_run:
            return RestoreResult.from_preview(preview, dry_run=True)

        # Track restore
        self.restore_count += 1
        self.last_restored = snapshot.name
        self.restore_history.append(snapshot.name)

        # Generate mock commands
        commands = [
            f"ALTER SYSTEM SET {c.param} = '{c.snapshot_value}'"
            for c in preview.changes
        ]
        commands.append("SELECT pg_reload_conf()")

        return RestoreResult(
            success=True,
            changes_applied=preview.total_changes,
            restart_required=preview.restart_required,
            commands_executed=commands,
            preview=preview,
        )

    def preview(self, snapshot: SnapshotData) -> RestorePreview:
        """Mock preview - shows what would change.

        Compares snapshot settings against "tuned" state as current.
        """
        # Assume current state is "tuned" for mock purposes
        current = SNAPSHOT_TEST_DATA["tuned_pg_settings"]
        changes = []

        for param, snapshot_value in snapshot.pg_settings.items():
            current_value = current.get(param, "")
            if current_value != snapshot_value:
                changes.append(SettingChange(
                    param=param,
                    current_value=current_value,
                    snapshot_value=snapshot_value,
                    requires_restart=requires_restart(param),
                ))

        return RestorePreview(changes=changes)

    def diff_settings(self, snapshot: SnapshotData) -> Dict[str, tuple]:
        """Compare current (tuned) settings with snapshot."""
        current = SNAPSHOT_TEST_DATA["tuned_pg_settings"]
        differences = {}

        for param, snapshot_value in snapshot.pg_settings.items():
            current_value = current.get(param, "")
            if current_value != snapshot_value:
                differences[param] = (current_value, snapshot_value)

        return differences


class MockSnapshotManager:
    """Mock snapshot manager for testing.

    Simulates high-level snapshot operations without file I/O.
    """

    def __init__(self, scenario: str = "balanced_tps"):
        self.scenario = scenario
        self._snapshots: Dict[str, SnapshotData] = {}
        self._capture = MockSnapshotCapture(scenario)
        self._restore = MockSnapshotRestore()

    @property
    def capture(self) -> MockSnapshotCapture:
        return self._capture

    @property
    def restore(self) -> MockSnapshotRestore:
        return self._restore

    def has_initial(self) -> bool:
        """Check if initial snapshot exists."""
        return "initial" in self._snapshots

    def get_initial(self) -> Optional[SnapshotData]:
        """Get initial snapshot if it exists."""
        return self._snapshots.get("initial")

    def get(self, name: str) -> Optional[SnapshotData]:
        """Get a snapshot by name."""
        return self._snapshots.get(name)

    def create_initial(self, session_name: str = "", force: bool = False) -> SnapshotData:
        """Create initial snapshot."""
        if not force and self.has_initial():
            return self.get_initial()

        self._capture.force_state("initial")
        snapshot = self._capture.capture(
            name="initial",
            trigger="automatic",
            session_name=session_name,
            round_num=0,
        )
        self._capture.force_state(None)

        self._snapshots["initial"] = snapshot
        return snapshot

    def create_checkpoint(
        self,
        name: str,
        session_name: str = "",
        round_num: int = 0,
        last_tps: Optional[float] = None,
        last_latency: Optional[float] = None,
    ) -> SnapshotData:
        """Create a checkpoint snapshot."""
        self._capture.force_state("tuned")
        snapshot = self._capture.capture(
            name=name,
            trigger="checkpoint",
            session_name=session_name,
            round_num=round_num,
            last_tps=last_tps,
            last_latency=last_latency,
        )
        self._capture.force_state(None)

        self._snapshots[name] = snapshot
        return snapshot

    def restore_to_initial(self, dry_run: bool = False) -> RestoreResult:
        """Restore to initial state."""
        initial = self.get_initial()
        if initial is None:
            return RestoreResult.failure("No initial snapshot found")
        return self._restore.restore(initial, dry_run=dry_run)

    def restore_to(self, name: str, dry_run: bool = False) -> RestoreResult:
        """Restore to a named snapshot."""
        snapshot = self.get(name)
        if snapshot is None:
            return RestoreResult.failure(f"Snapshot '{name}' not found")
        return self._restore.restore(snapshot, dry_run=dry_run)

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all snapshots."""
        return [
            {
                "name": s.name,
                "created_at": s.created_at,
                "trigger": s.trigger,
                "session_name": s.session_name,
                "round_num": s.round_num,
                "last_tps": s.last_tps,
                "param_count": len(s.pg_settings),
            }
            for s in self._snapshots.values()
        ]

    def compare(self, name_a: str, name_b: str) -> Optional[SnapshotDiff]:
        """Compare two snapshots."""
        snap_a = self.get(name_a)
        snap_b = self.get(name_b)

        if snap_a is None or snap_b is None:
            return None

        changes = []
        all_params = set(snap_a.pg_settings.keys()) | set(snap_b.pg_settings.keys())

        for param in all_params:
            val_a = snap_a.pg_settings.get(param, "")
            val_b = snap_b.pg_settings.get(param, "")

            if val_a != val_b:
                changes.append(SettingChange(
                    param=param,
                    current_value=val_a,
                    snapshot_value=val_b,
                    requires_restart=requires_restart(param),
                ))

        return SnapshotDiff(
            snapshot_a=name_a,
            snapshot_b=name_b,
            changes=changes,
        )

    def ensure_initial_exists(self, session_name: str = "") -> SnapshotData:
        """Create initial if it doesn't exist."""
        if self.has_initial():
            return self.get_initial()
        return self.create_initial(session_name=session_name)


# =============================================================================
# Factory
# =============================================================================

class MockSnapshotFactory:
    """Factory for creating mock snapshot components."""

    @staticmethod
    def create_capture(scenario: str = "balanced_tps") -> MockSnapshotCapture:
        """Create a mock capture component."""
        return MockSnapshotCapture(scenario)

    @staticmethod
    def create_restore() -> MockSnapshotRestore:
        """Create a mock restore component."""
        return MockSnapshotRestore()

    @staticmethod
    def create_manager(scenario: str = "balanced_tps") -> MockSnapshotManager:
        """Create a mock manager component."""
        return MockSnapshotManager(scenario)
