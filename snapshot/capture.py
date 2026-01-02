"""
Snapshot capture - collects PostgreSQL configuration state.
"""

import hashlib
import subprocess
from typing import Dict, Optional, Any
from pathlib import Path

from .models import SnapshotData


class SnapshotCapture:
    """Captures PostgreSQL configuration state."""

    def __init__(
        self,
        conn,
        ssh_config: Optional[Dict[str, Any]] = None,
        data_directory: Optional[str] = None,
    ):
        """
        Initialize snapshot capture.

        Args:
            conn: psycopg2 database connection
            ssh_config: Optional SSH config for remote capture
                       {"host": str, "user": str, "port": int, "key": str}
            data_directory: PostgreSQL data directory (auto-detected if not provided)
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
        """Check if we're capturing from a remote server."""
        return self.ssh_config is not None and self.ssh_config.get('host')

    def capture(
        self,
        name: str,
        trigger: str = "manual",
        session_name: str = "",
        round_num: int = 0,
        last_tps: Optional[float] = None,
        last_latency: Optional[float] = None,
    ) -> SnapshotData:
        """
        Capture PostgreSQL configuration state.

        Args:
            name: Snapshot name (e.g., "initial", "checkpoint-1")
            trigger: How snapshot was triggered ("automatic", "manual", "checkpoint")
            session_name: Name of the session creating this snapshot
            round_num: Current tuning round (0 = before any tuning)
            last_tps: TPS at snapshot time (optional)
            last_latency: Latency at snapshot time (optional)

        Returns:
            SnapshotData with captured configuration
        """
        return SnapshotData.create(
            name=name,
            pg_settings=self._capture_pg_settings(),
            pg_auto_conf=self._capture_auto_conf(),
            pg_conf_hash=self._capture_conf_hash(),
            trigger=trigger,
            session_name=session_name,
            round_num=round_num,
            last_tps=last_tps,
            last_latency=last_latency,
        )

    def _capture_pg_settings(self) -> Dict[str, str]:
        """
        Query pg_settings for all tunable parameters.

        Returns dict of {param_name: current_setting}
        """
        settings = {}
        with self.conn.cursor() as cur:
            # Get all non-internal settings that can be changed
            cur.execute("""
                SELECT name, setting
                FROM pg_settings
                WHERE context != 'internal'
                  AND vartype != 'string'  -- Skip paths, etc.
                ORDER BY name
            """)
            for name, setting in cur.fetchall():
                settings[name] = setting

            # Also capture string settings we care about
            cur.execute("""
                SELECT name, setting
                FROM pg_settings
                WHERE name IN (
                    'shared_preload_libraries',
                    'listen_addresses',
                    'log_destination',
                    'log_directory',
                    'archive_command'
                )
            """)
            for name, setting in cur.fetchall():
                settings[name] = setting

        return settings

    def _capture_auto_conf(self) -> str:
        """
        Read postgresql.auto.conf contents.

        This file contains ALTER SYSTEM settings.
        """
        auto_conf_path = f"{self.data_directory}/postgresql.auto.conf"

        if self.is_remote:
            return self._read_remote_file(auto_conf_path)
        else:
            path = Path(auto_conf_path)
            if path.exists():
                return path.read_text()
            return ""

    def _capture_conf_hash(self) -> str:
        """
        Get MD5 hash of postgresql.conf to detect manual edits.
        """
        conf_path = f"{self.data_directory}/postgresql.conf"

        if self.is_remote:
            content = self._read_remote_file(conf_path)
        else:
            path = Path(conf_path)
            if path.exists():
                content = path.read_text()
            else:
                content = ""

        return hashlib.md5(content.encode()).hexdigest()

    def _read_remote_file(self, path: str) -> str:
        """Read a file from remote server via SSH."""
        ssh_cmd = self._build_ssh_command(f"cat {path} 2>/dev/null || echo ''")
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""

    def _build_ssh_command(self, cmd: str) -> list:
        """Build SSH command with config."""
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]

        if self.ssh_config.get('key'):
            ssh_cmd.extend(["-i", self.ssh_config['key']])

        if self.ssh_config.get('port', 22) != 22:
            ssh_cmd.extend(["-p", str(self.ssh_config['port'])])

        user = self.ssh_config.get('user', 'ubuntu')
        host = self.ssh_config['host']
        ssh_cmd.append(f"{user}@{host}")
        ssh_cmd.append(cmd)

        return ssh_cmd
