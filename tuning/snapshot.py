"""
TuningSnapshotManager - Captures state before tuning for rollback.

v2.2 Resilience feature: Enables emergency rollback even when DB is dead.
"""

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2

from ..protocol.tuning import TuningChunk, TuningSnapshot


class TuningSnapshotManager:
    """
    Manages pre-tuning state snapshots for emergency rollback.

    Captures:
    - Database parameter values (SHOW)
    - OS sysctl values
    - Config file backups (for FILE_RESTORE strategy)
    """

    BACKUP_DIR = Path("/tmp/pg_diagnose_backups")

    def __init__(self, connection=None, ssh_config=None):
        """
        Initialize snapshot manager.

        Args:
            connection: psycopg2 connection for DB param capture
            ssh_config: SSH config for remote operations
        """
        self.conn = connection
        self.ssh_config = ssh_config
        self.snapshots: Dict[str, TuningSnapshot] = {}

    def capture(self, chunk: TuningChunk) -> TuningSnapshot:
        """
        Capture state before applying a TuningChunk.

        Args:
            chunk: The tuning chunk about to be applied

        Returns:
            TuningSnapshot containing original values
        """
        self.BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        snapshot = TuningSnapshot(
            chunk_id=chunk.id,
            timestamp=datetime.utcnow(),
            original_db_values={},
            original_os_values={},
            config_file_backup_path=None,
        )

        # Capture DB parameter values
        for cmd in chunk.apply_commands:
            param = self._extract_db_param(cmd)
            if param and self.conn:
                try:
                    value = self._get_db_param(param)
                    if value is not None:
                        snapshot.original_db_values[param] = value
                except Exception:
                    pass

        # Capture OS sysctl values
        for cmd in chunk.apply_commands:
            param = self._extract_sysctl_param(cmd)
            if param:
                try:
                    value = self._get_sysctl_value(param)
                    if value:
                        snapshot.original_os_values[param] = value
                except Exception:
                    pass

        # Backup config file if using FILE_RESTORE strategy
        if chunk.recovery_strategy == "FILE_RESTORE" and chunk.target_config_file:
            backup_path = self._backup_config_file(chunk.id, chunk.target_config_file)
            if backup_path:
                snapshot.config_file_backup_path = str(backup_path)

        # Store snapshot for later retrieval
        self.snapshots[chunk.id] = snapshot

        return snapshot

    def get_snapshot(self, chunk_id: str) -> Optional[TuningSnapshot]:
        """Retrieve a previously captured snapshot."""
        return self.snapshots.get(chunk_id)

    def _extract_db_param(self, cmd: str) -> Optional[str]:
        """Extract parameter name from ALTER SYSTEM command."""
        # Match: ALTER SYSTEM SET param_name = ...
        match = re.search(
            r"ALTER\s+SYSTEM\s+SET\s+(\w+)\s*=",
            cmd,
            re.IGNORECASE
        )
        if match:
            return match.group(1).lower()

        # Match: ALTER SYSTEM RESET param_name
        match = re.search(
            r"ALTER\s+SYSTEM\s+RESET\s+(\w+)",
            cmd,
            re.IGNORECASE
        )
        if match:
            return match.group(1).lower()

        return None

    def _extract_sysctl_param(self, cmd: str) -> Optional[str]:
        """Extract parameter name from sysctl command."""
        # Match: sysctl -w param.name=value or sysctl param.name=value
        match = re.search(r"sysctl\s+(?:-w\s+)?([a-z_.]+)\s*=", cmd, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _get_db_param(self, param: str) -> Optional[str]:
        """Get current value of a database parameter."""
        if not self.conn:
            return None

        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SHOW {param}")
                return cur.fetchone()[0]
        except Exception:
            return None

    def _get_sysctl_value(self, param: str) -> Optional[str]:
        """Get current value of a sysctl parameter."""
        try:
            if self.ssh_config:
                # Remote execution
                result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no",
                     f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                     f"sysctl -n {param}"],
                    capture_output=True, text=True
                )
            else:
                result = subprocess.run(
                    ["sysctl", "-n", param],
                    capture_output=True, text=True
                )

            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def _backup_config_file(self, chunk_id: str, config_path: str) -> Optional[Path]:
        """Backup a config file before modification."""
        try:
            source = Path(config_path)
            if not source.exists():
                return None

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{chunk_id}_{timestamp}.conf.bak"
            backup_path = self.BACKUP_DIR / backup_name

            if self.ssh_config:
                # Remote: use scp to copy file locally first, then back up
                # For simplicity, we'll create a remote backup
                remote_backup = f"/tmp/pg_diagnose_backups/{backup_name}"
                subprocess.run([
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                    f"mkdir -p /tmp/pg_diagnose_backups && cp {config_path} {remote_backup}"
                ], check=True)
                return Path(remote_backup)
            else:
                shutil.copy2(source, backup_path)
                return backup_path

        except Exception:
            return None

    def restore_from_snapshot(self, snapshot: TuningSnapshot, chunk: TuningChunk) -> bool:
        """
        Restore system to snapshot state.

        Used during emergency rollback.

        Args:
            snapshot: The snapshot to restore to
            chunk: The chunk that was being applied

        Returns:
            True if restore succeeded
        """
        success = True

        # Restore config file if FILE_RESTORE strategy
        if chunk.recovery_strategy == "FILE_RESTORE" and snapshot.config_file_backup_path:
            try:
                if self.ssh_config:
                    subprocess.run([
                        "ssh", "-o", "StrictHostKeyChecking=no",
                        f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                        f"cp {snapshot.config_file_backup_path} {chunk.target_config_file}"
                    ], check=True)
                else:
                    shutil.copy2(snapshot.config_file_backup_path, chunk.target_config_file)
            except Exception:
                success = False

        # Restore OS parameters
        if chunk.recovery_strategy == "OS_REVERT":
            for param, value in snapshot.original_os_values.items():
                try:
                    cmd = f"sysctl -w {param}={value}"
                    if self.ssh_config:
                        subprocess.run([
                            "ssh", "-o", "StrictHostKeyChecking=no",
                            f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                            f"sudo {cmd}"
                        ], check=True)
                    else:
                        subprocess.run(["sudo", "sysctl", "-w", f"{param}={value}"], check=True)
                except Exception:
                    success = False

        # Restore DB parameters (only if DB is accessible)
        if chunk.recovery_strategy == "SQL_REVERT" and self.conn:
            for param, value in snapshot.original_db_values.items():
                try:
                    with self.conn.cursor() as cur:
                        if value == '':
                            cur.execute(f"ALTER SYSTEM RESET {param}")
                        else:
                            cur.execute(f"ALTER SYSTEM SET {param} = '{value}'")
                    self.conn.commit()
                except Exception:
                    success = False

        return success

    def cleanup(self, chunk_id: Optional[str] = None):
        """
        Clean up snapshot files.

        Args:
            chunk_id: Specific chunk to clean up, or None for all
        """
        if chunk_id:
            snapshot = self.snapshots.pop(chunk_id, None)
            if snapshot and snapshot.config_file_backup_path:
                try:
                    Path(snapshot.config_file_backup_path).unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            # Clean all
            for snapshot in self.snapshots.values():
                if snapshot.config_file_backup_path:
                    try:
                        Path(snapshot.config_file_backup_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            self.snapshots.clear()
