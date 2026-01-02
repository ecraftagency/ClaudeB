"""
Snapshot restore - applies configuration from a snapshot.
"""

import subprocess
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .models import (
    SnapshotData,
    RestoreResult,
    RestorePreview,
    SettingChange,
    requires_restart,
)


class SnapshotRestore:
    """Restores PostgreSQL to snapshot state."""

    def __init__(
        self,
        conn,
        ssh_config: Optional[Dict[str, Any]] = None,
        data_directory: Optional[str] = None,
    ):
        """
        Initialize snapshot restore.

        Args:
            conn: psycopg2 database connection
            ssh_config: Optional SSH config for remote restore
            data_directory: PostgreSQL data directory
        """
        self.conn = conn
        self.ssh_config = ssh_config
        self._data_directory = data_directory

    @property
    def data_directory(self) -> str:
        """Get PostgreSQL data directory."""
        if self._data_directory:
            return self._data_directory

        with self.conn.cursor() as cur:
            cur.execute("SHOW data_directory")
            self._data_directory = cur.fetchone()[0]
        return self._data_directory

    @property
    def is_remote(self) -> bool:
        """Check if we're restoring to a remote server."""
        return self.ssh_config is not None and self.ssh_config.get('host')

    def restore(
        self,
        snapshot: SnapshotData,
        dry_run: bool = False,
    ) -> RestoreResult:
        """
        Restore PostgreSQL to snapshot state.

        Strategy:
        1. Compare current settings with snapshot
        2. For each difference, generate ALTER SYSTEM SET command
        3. Execute commands (unless dry_run)
        4. Call pg_reload_conf() to apply changes

        Args:
            snapshot: The snapshot to restore to
            dry_run: If True, only preview changes without applying

        Returns:
            RestoreResult with success status and details
        """
        # Get preview of changes
        preview = self.preview(snapshot)

        if dry_run:
            return RestoreResult.from_preview(preview, dry_run=True)

        if preview.total_changes == 0:
            return RestoreResult(
                success=True,
                changes_applied=0,
                restart_required=False,
                commands_executed=[],
            )

        # Execute changes
        try:
            commands = self._generate_restore_commands(snapshot, preview)
            self._execute_commands(commands)

            return RestoreResult(
                success=True,
                changes_applied=preview.total_changes,
                restart_required=preview.restart_required,
                commands_executed=commands,
                preview=preview,
            )
        except Exception as e:
            return RestoreResult.failure(str(e))

    def preview(self, snapshot: SnapshotData) -> RestorePreview:
        """
        Show what would change without applying.

        Args:
            snapshot: The snapshot to compare against

        Returns:
            RestorePreview with list of changes
        """
        current_settings = self._get_current_settings()
        changes = []

        for param, snapshot_value in snapshot.pg_settings.items():
            current_value = current_settings.get(param)

            # Skip if current value matches snapshot
            if current_value == snapshot_value:
                continue

            # Skip if param doesn't exist in current settings
            # (might be a different PG version)
            if current_value is None:
                continue

            changes.append(SettingChange(
                param=param,
                current_value=current_value,
                snapshot_value=snapshot_value,
                requires_restart=requires_restart(param),
            ))

        return RestorePreview(changes=changes)

    def _get_current_settings(self) -> Dict[str, str]:
        """Get current pg_settings values."""
        settings = {}
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT name, setting
                FROM pg_settings
                WHERE context != 'internal'
            """)
            for name, setting in cur.fetchall():
                settings[name] = setting
        return settings

    def _generate_restore_commands(
        self,
        snapshot: SnapshotData,
        preview: RestorePreview,
    ) -> List[str]:
        """Generate ALTER SYSTEM commands to restore settings."""
        commands = []

        for change in preview.changes:
            # Use the snapshot value
            value = change.snapshot_value

            # Handle special cases
            if value == '' or value is None:
                # Reset to default
                cmd = f"ALTER SYSTEM RESET {change.param}"
            else:
                # Set to snapshot value (quote if needed)
                if self._needs_quoting(change.param, value):
                    cmd = f"ALTER SYSTEM SET {change.param} = '{value}'"
                else:
                    cmd = f"ALTER SYSTEM SET {change.param} = {value}"

            commands.append(cmd)

        # Always reload config after changes
        commands.append("SELECT pg_reload_conf()")

        return commands

    def _needs_quoting(self, param: str, value: str) -> bool:
        """Check if a value needs quoting in ALTER SYSTEM."""
        # Numeric values don't need quotes
        try:
            float(value)
            return False
        except ValueError:
            pass

        # Values with units (like '8GB') need quotes
        # Boolean values don't need quotes
        if value.lower() in ('on', 'off', 'true', 'false'):
            return False

        return True

    def _execute_commands(self, commands: List[str]) -> None:
        """Execute ALTER SYSTEM commands."""
        # Use autocommit for ALTER SYSTEM
        old_autocommit = self.conn.autocommit
        try:
            self.conn.autocommit = True
            with self.conn.cursor() as cur:
                for cmd in commands:
                    cur.execute(cmd)
        finally:
            self.conn.autocommit = old_autocommit

    def diff_settings(
        self,
        snapshot: SnapshotData,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Compare current settings with snapshot.

        Returns:
            Dict of {param: (current_value, snapshot_value)} for differences
        """
        current_settings = self._get_current_settings()
        differences = {}

        for param, snapshot_value in snapshot.pg_settings.items():
            current_value = current_settings.get(param, '')
            if current_value != snapshot_value:
                differences[param] = (current_value, snapshot_value)

        return differences
