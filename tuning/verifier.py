"""
TuningVerifier - Verifies tuning changes took effect.

Compares expected values against actual values after applying tuning.
"""

import re
import subprocess
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2


class TuningVerifier:
    """
    Verifies that tuning changes took effect.

    Supports:
    - SQL SHOW commands
    - sysctl parameter checks
    - File content verification
    """

    def __init__(self, connection=None, ssh_config=None):
        """
        Initialize verifier.

        Args:
            connection: psycopg2 connection for DB verification
            ssh_config: SSH config for remote verification
        """
        self.conn = connection
        self.ssh_config = ssh_config

    def get_value(self, verification_command: str) -> Optional[str]:
        """
        Execute verification command and return result.

        Args:
            verification_command: Command to execute (SQL or shell)

        Returns:
            The value returned by the command
        """
        cmd_upper = verification_command.upper().strip()

        # SQL commands (SHOW, SELECT)
        if cmd_upper.startswith('SHOW ') or cmd_upper.startswith('SELECT '):
            return self._execute_sql(verification_command)

        # sysctl commands
        elif 'sysctl' in verification_command.lower():
            return self._execute_sysctl(verification_command)

        # Shell commands
        else:
            return self._execute_shell(verification_command)

    def _execute_sql(self, cmd: str) -> Optional[str]:
        """Execute SQL verification command."""
        if not self.conn:
            return None

        try:
            with self.conn.cursor() as cur:
                cur.execute(cmd)
                result = cur.fetchone()
                if result:
                    return str(result[0])
        except Exception:
            pass

        return None

    def _execute_sysctl(self, cmd: str) -> Optional[str]:
        """Execute sysctl verification command."""
        # Extract parameter name from command
        # Handle: "sysctl -n param.name" or "sysctl param.name"
        match = re.search(r'sysctl\s+(?:-n\s+)?([a-z_.]+)', cmd, re.IGNORECASE)
        if not match:
            return None

        param = match.group(1)

        try:
            if self.ssh_config:
                result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no",
                     f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                     f"sysctl -n {param}"],
                    capture_output=True, text=True, timeout=10
                )
            else:
                result = subprocess.run(
                    ["sysctl", "-n", param],
                    capture_output=True, text=True, timeout=10
                )

            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def _execute_shell(self, cmd: str) -> Optional[str]:
        """Execute shell verification command."""
        try:
            if self.ssh_config:
                result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no",
                     f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                     cmd],
                    capture_output=True, text=True, timeout=30
                )
            else:
                result = subprocess.run(
                    cmd, shell=True,
                    capture_output=True, text=True, timeout=30
                )

            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def matches(self, actual: Optional[str], expected: str) -> bool:
        """
        Compare actual value against expected value.

        Handles various comparison modes:
        - Exact match
        - Numeric comparison with units (4GB == 4096MB)
        - Regex patterns
        - Partial matches

        Args:
            actual: The actual value from verification
            expected: The expected value

        Returns:
            True if values match
        """
        if actual is None:
            return False

        actual = actual.strip()
        expected = expected.strip()

        # Exact match
        if actual.lower() == expected.lower():
            return True

        # Numeric comparison (handle units)
        actual_bytes = self._parse_size(actual)
        expected_bytes = self._parse_size(expected)

        if actual_bytes is not None and expected_bytes is not None:
            # Allow 1% tolerance for rounding
            tolerance = max(actual_bytes, expected_bytes) * 0.01
            return abs(actual_bytes - expected_bytes) <= tolerance

        # Check if expected is a regex pattern
        if expected.startswith('/') and expected.endswith('/'):
            pattern = expected[1:-1]
            try:
                return bool(re.search(pattern, actual, re.IGNORECASE))
            except re.error:
                pass

        # Substring match for complex values
        if expected in actual or actual in expected:
            return True

        return False

    def _parse_size(self, value: str) -> Optional[int]:
        """
        Parse a size value with units to bytes.

        Handles PostgreSQL and standard units:
        - B, kB, MB, GB, TB
        - K, M, G, T (without B)
        """
        value = value.strip().upper()

        # Remove trailing 'B' for byte suffix if present
        # Handle cases like "GB", "MB", "KB"
        units = {
            'TB': 1024**4,
            'T': 1024**4,
            'GB': 1024**3,
            'G': 1024**3,
            'MB': 1024**2,
            'M': 1024**2,
            'KB': 1024,
            'K': 1024,
            'B': 1,
            '': 1,
        }

        # Try to extract number and unit
        match = re.match(r'^(\d+(?:\.\d+)?)\s*([TGMKB]*)', value)
        if match:
            try:
                number = float(match.group(1))
                unit = match.group(2) if match.group(2) else ''

                if unit in units:
                    return int(number * units[unit])
            except (ValueError, KeyError):
                pass

        # Try parsing as plain integer
        try:
            return int(value)
        except ValueError:
            pass

        return None

    def verify_chunk(
        self,
        verification_command: str,
        verification_expected: str
    ) -> tuple:
        """
        Verify a tuning chunk's expected outcome.

        Args:
            verification_command: Command to check current value
            verification_expected: Expected value

        Returns:
            (success: bool, actual_value: str, message: str)
        """
        actual = self.get_value(verification_command)

        if actual is None:
            return (
                False,
                "N/A",
                f"Failed to execute verification command: {verification_command}"
            )

        if self.matches(actual, verification_expected):
            return (
                True,
                actual,
                f"Verification passed: {actual} matches expected {verification_expected}"
            )
        else:
            return (
                False,
                actual,
                f"Verification failed: got {actual}, expected {verification_expected}"
            )

    def verify_db_connection(self) -> bool:
        """Verify database connection is alive."""
        if not self.conn:
            return False

        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
        except Exception:
            return False

    def verify_service_running(self, service_name: str = "postgresql") -> bool:
        """Verify a systemd service is running."""
        try:
            cmd = f"systemctl is-active {service_name}"

            if self.ssh_config:
                result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no",
                     f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                     cmd],
                    capture_output=True, text=True, timeout=10
                )
            else:
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True, text=True, timeout=10
                )

            return result.stdout.strip() == "active"
        except Exception:
            return False
