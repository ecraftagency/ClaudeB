"""
Snapshot manager - high-level snapshot operations.
"""

from pathlib import Path
from typing import List, Optional, Any, Dict
from datetime import datetime

from .models import (
    SnapshotData,
    SnapshotInfo,
    SnapshotDiff,
    SettingChange,
    RestoreResult,
    RestorePreview,
    requires_restart,
)
from .capture import SnapshotCapture
from .restore import SnapshotRestore


class SnapshotManager:
    """High-level snapshot operations for a workspace."""

    INITIAL_SNAPSHOT_NAME = "initial"

    def __init__(
        self,
        workspace_path: Path,
        conn=None,
        ssh_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize snapshot manager.

        Args:
            workspace_path: Path to workspace directory
            conn: psycopg2 database connection (required for capture/restore)
            ssh_config: Optional SSH config for remote operations
        """
        self.workspace_path = workspace_path
        self.conn = conn
        self.ssh_config = ssh_config

        # Snapshot storage directories
        self.snapshots_dir = workspace_path / "snapshots"
        self.backups_dir = self.snapshots_dir / "backups"

        # Ensure directories exist
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-initialized components
        self._capture: Optional[SnapshotCapture] = None
        self._restore: Optional[SnapshotRestore] = None

    @property
    def capture(self) -> SnapshotCapture:
        """Get or create snapshot capture component."""
        if self._capture is None:
            if self.conn is None:
                raise RuntimeError("Database connection required for capture")
            self._capture = SnapshotCapture(self.conn, self.ssh_config)
        return self._capture

    @property
    def restore(self) -> SnapshotRestore:
        """Get or create snapshot restore component."""
        if self._restore is None:
            if self.conn is None:
                raise RuntimeError("Database connection required for restore")
            self._restore = SnapshotRestore(self.conn, self.ssh_config)
        return self._restore

    def _snapshot_path(self, name: str) -> Path:
        """Get path to snapshot file."""
        return self.snapshots_dir / f"{name}.json"

    # =========================================================================
    # Core Operations
    # =========================================================================

    def create_initial(
        self,
        session_name: str = "",
        force: bool = False,
    ) -> SnapshotData:
        """
        Create 'initial' snapshot before any tuning.

        This is the baseline state to restore to.

        Args:
            session_name: Name of the session creating this
            force: If True, overwrite existing initial snapshot

        Returns:
            The created (or existing) snapshot
        """
        if not force and self.has_initial():
            return self.get_initial()

        snapshot = self.capture.capture(
            name=self.INITIAL_SNAPSHOT_NAME,
            trigger="automatic",
            session_name=session_name,
            round_num=0,
        )

        snapshot.save(self._snapshot_path(self.INITIAL_SNAPSHOT_NAME))

        # Also backup postgresql.auto.conf
        if snapshot.pg_auto_conf:
            backup_path = self.backups_dir / "postgresql.auto.conf.initial"
            backup_path.write_text(snapshot.pg_auto_conf)

        return snapshot

    def create_checkpoint(
        self,
        name: str,
        session_name: str = "",
        round_num: int = 0,
        last_tps: Optional[float] = None,
        last_latency: Optional[float] = None,
    ) -> SnapshotData:
        """
        Create a named checkpoint snapshot.

        Args:
            name: Checkpoint name (e.g., "after-memory-tuning")
            session_name: Name of the session
            round_num: Current tuning round
            last_tps: Latest TPS result
            last_latency: Latest latency result

        Returns:
            The created snapshot
        """
        # Sanitize name
        safe_name = name.replace(" ", "-").replace("/", "-")

        snapshot = self.capture.capture(
            name=safe_name,
            trigger="checkpoint",
            session_name=session_name,
            round_num=round_num,
            last_tps=last_tps,
            last_latency=last_latency,
        )

        snapshot.save(self._snapshot_path(safe_name))
        return snapshot

    def restore_to_initial(self, dry_run: bool = False) -> RestoreResult:
        """
        Restore to initial state.

        Args:
            dry_run: If True, only preview changes

        Returns:
            RestoreResult with details
        """
        initial = self.get_initial()
        if initial is None:
            return RestoreResult.failure("No initial snapshot found")

        return self.restore.restore(initial, dry_run=dry_run)

    def restore_to(self, name: str, dry_run: bool = False) -> RestoreResult:
        """
        Restore to a named snapshot.

        Args:
            name: Snapshot name
            dry_run: If True, only preview changes

        Returns:
            RestoreResult with details
        """
        snapshot = self.get(name)
        if snapshot is None:
            return RestoreResult.failure(f"Snapshot '{name}' not found")

        return self.restore.restore(snapshot, dry_run=dry_run)

    # =========================================================================
    # Query Operations
    # =========================================================================

    def has_initial(self) -> bool:
        """Check if initial snapshot exists."""
        return self._snapshot_path(self.INITIAL_SNAPSHOT_NAME).exists()

    def get_initial(self) -> Optional[SnapshotData]:
        """Get initial snapshot if it exists."""
        return self.get(self.INITIAL_SNAPSHOT_NAME)

    def get(self, name: str) -> Optional[SnapshotData]:
        """Get a snapshot by name."""
        path = self._snapshot_path(name)
        if not path.exists():
            return None
        return SnapshotData.load(path)

    def list_snapshots(self) -> List[SnapshotInfo]:
        """
        List all snapshots with summary info.

        Returns:
            List of SnapshotInfo sorted by creation time (newest first)
        """
        snapshots = []

        for path in self.snapshots_dir.glob("*.json"):
            try:
                snapshot = SnapshotData.load(path)
                snapshots.append(SnapshotInfo(
                    name=snapshot.name,
                    created_at=snapshot.created_at,
                    trigger=snapshot.trigger,
                    session_name=snapshot.session_name,
                    round_num=snapshot.round_num,
                    last_tps=snapshot.last_tps,
                    param_count=len(snapshot.pg_settings),
                ))
            except Exception:
                continue

        # Sort by creation time (newest first)
        snapshots.sort(key=lambda s: s.created_at, reverse=True)
        return snapshots

    def compare(self, name_a: str, name_b: str) -> Optional[SnapshotDiff]:
        """
        Compare two snapshots.

        Args:
            name_a: First snapshot name
            name_b: Second snapshot name

        Returns:
            SnapshotDiff or None if snapshots don't exist
        """
        snap_a = self.get(name_a)
        snap_b = self.get(name_b)

        if snap_a is None or snap_b is None:
            return None

        changes = []
        only_in_a = []
        only_in_b = []

        # Find differences
        all_params = set(snap_a.pg_settings.keys()) | set(snap_b.pg_settings.keys())

        for param in all_params:
            val_a = snap_a.pg_settings.get(param)
            val_b = snap_b.pg_settings.get(param)

            if val_a is None:
                only_in_b.append(param)
            elif val_b is None:
                only_in_a.append(param)
            elif val_a != val_b:
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
            only_in_a=only_in_a,
            only_in_b=only_in_b,
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def ensure_initial_exists(self, session_name: str = "") -> SnapshotData:
        """
        Create initial snapshot if it doesn't exist.

        Args:
            session_name: Name of the session

        Returns:
            The initial snapshot (created or existing)
        """
        if self.has_initial():
            return self.get_initial()
        return self.create_initial(session_name=session_name)

    def cleanup_old(self, keep_count: int = 10) -> int:
        """
        Remove old checkpoint snapshots, keeping the most recent ones.

        Always keeps the 'initial' snapshot.

        Args:
            keep_count: Number of checkpoints to keep (excluding initial)

        Returns:
            Number of snapshots removed
        """
        snapshots = self.list_snapshots()

        # Filter out initial
        checkpoints = [s for s in snapshots if s.name != self.INITIAL_SNAPSHOT_NAME]

        # Keep the most recent ones
        to_remove = checkpoints[keep_count:]

        removed = 0
        for snapshot in to_remove:
            path = self._snapshot_path(snapshot.name)
            if path.exists():
                path.unlink()
                removed += 1

        return removed

    def delete(self, name: str) -> bool:
        """
        Delete a snapshot.

        Cannot delete the initial snapshot.

        Args:
            name: Snapshot name

        Returns:
            True if deleted, False if not found or protected
        """
        if name == self.INITIAL_SNAPSHOT_NAME:
            return False

        path = self._snapshot_path(name)
        if path.exists():
            path.unlink()
            return True
        return False
