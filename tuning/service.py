"""
ServiceController - Manages PostgreSQL service operations.

Provides:
- Service restart (systemctl)
- Health probe (pg_isready)
- Log retrieval (journalctl)
"""

import subprocess
import time
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    """Configuration for service controller."""
    service_name: str = "postgresql"
    pg_host: str = "localhost"
    pg_port: int = 5432
    probe_timeout: int = 60  # seconds
    probe_interval: int = 2  # seconds
    ssh_host: Optional[str] = None
    ssh_user: str = "ubuntu"
    ssh_key: Optional[str] = None


class ServiceController:
    """
    Controls PostgreSQL service operations.

    Supports both local and remote (SSH) operations.
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()
        self.is_remote = self.config.ssh_host is not None

    def _run_command(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command locally or remotely."""
        if self.is_remote:
            ssh_cmd = ["ssh"]
            if self.config.ssh_key:
                ssh_cmd.extend(["-i", self.config.ssh_key])
            ssh_cmd.extend([
                "-o", "StrictHostKeyChecking=no",
                f"{self.config.ssh_user}@{self.config.ssh_host}",
                cmd
            ])
            return subprocess.run(ssh_cmd, capture_output=True, text=True, check=check)
        else:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)

    def restart(self, timeout: int = 30) -> bool:
        """
        Restart the PostgreSQL service.

        Args:
            timeout: Maximum time to wait for restart

        Returns:
            True if restart succeeded
        """
        try:
            cmd = f"sudo systemctl restart {self.config.service_name}"
            self._run_command(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def stop(self) -> bool:
        """Stop the PostgreSQL service."""
        try:
            cmd = f"sudo systemctl stop {self.config.service_name}"
            self._run_command(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def start(self) -> bool:
        """Start the PostgreSQL service."""
        try:
            cmd = f"sudo systemctl start {self.config.service_name}"
            self._run_command(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def reload(self) -> bool:
        """Reload PostgreSQL configuration (pg_reload_conf equivalent)."""
        try:
            cmd = f"sudo systemctl reload {self.config.service_name}"
            self._run_command(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def status(self) -> str:
        """Get service status."""
        try:
            cmd = f"systemctl is-active {self.config.service_name}"
            result = self._run_command(cmd, check=False)
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def is_running(self) -> bool:
        """Check if service is running."""
        return self.status() == "active"

    def probe_connectivity(self, timeout: Optional[int] = None) -> bool:
        """
        Probe database connectivity using pg_isready.

        Args:
            timeout: Maximum time to wait for connection

        Returns:
            True if database is accepting connections
        """
        timeout = timeout or self.config.probe_timeout
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self._check_pg_ready():
                return True
            time.sleep(self.config.probe_interval)

        return False

    def _check_pg_ready(self) -> bool:
        """Single check if PostgreSQL is ready."""
        try:
            cmd = f"pg_isready -h {self.config.pg_host} -p {self.config.pg_port}"
            result = self._run_command(cmd, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def get_tail_logs(self, lines: int = 50) -> List[str]:
        """
        Get recent service logs from journalctl.

        Args:
            lines: Number of log lines to retrieve

        Returns:
            List of log lines
        """
        try:
            cmd = f"sudo journalctl -u {self.config.service_name} -n {lines} --no-pager"
            result = self._run_command(cmd, check=False)
            return result.stdout.strip().split('\n') if result.stdout else []
        except Exception:
            return []

    def get_postgres_logs(self, lines: int = 50, log_dir: str = "/var/log/postgresql") -> List[str]:
        """
        Get recent PostgreSQL logs from log directory.

        Fallback when journalctl doesn't capture PostgreSQL logs.
        """
        try:
            # Find most recent log file
            cmd = f"ls -t {log_dir}/*.log 2>/dev/null | head -1"
            result = self._run_command(cmd, check=False)
            log_file = result.stdout.strip()

            if log_file:
                cmd = f"tail -n {lines} {log_file}"
                result = self._run_command(cmd, check=False)
                return result.stdout.strip().split('\n') if result.stdout else []
        except Exception:
            pass

        return []

    def get_startup_errors(self, since_seconds: int = 300) -> List[str]:
        """
        Get startup-related error messages.

        Args:
            since_seconds: How far back to look

        Returns:
            List of error log lines
        """
        try:
            cmd = (
                f"sudo journalctl -u {self.config.service_name} "
                f"--since '{since_seconds} seconds ago' --no-pager "
                f"| grep -iE 'FATAL|ERROR|could not|failed'"
            )
            result = self._run_command(cmd, check=False)
            return result.stdout.strip().split('\n') if result.stdout else []
        except Exception:
            return []

    def restart_and_probe(self, timeout: Optional[int] = None) -> tuple:
        """
        Restart service and probe for connectivity.

        Returns:
            (success: bool, logs: List[str])
        """
        timeout = timeout or self.config.probe_timeout

        # Restart
        if not self.restart():
            logs = self.get_tail_logs(100)
            return False, logs

        # Wait for startup
        time.sleep(2)

        # Probe
        if self.probe_connectivity(timeout):
            return True, []
        else:
            logs = self.get_startup_errors()
            if not logs:
                logs = self.get_tail_logs(100)
            return False, logs
