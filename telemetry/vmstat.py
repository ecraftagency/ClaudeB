"""
VMStatCollector - Collects system memory and CPU metrics via vmstat.

Provides:
- Memory usage (free, buffer, cache, swap)
- CPU utilization (user, system, idle, wait)
- Process stats (runnable, blocked)
"""

import subprocess
import threading
from typing import Dict, List, Optional, Any


class VMStatCollector:
    """
    Collects system statistics using vmstat.

    Supports both local and remote (SSH) collection.
    """

    def __init__(self, ssh_config: Optional[Dict] = None):
        self.ssh_config = ssh_config
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._latest: Dict[str, Any] = {}
        self._running = False
        self._lock = threading.Lock()

    def start(self, interval: int = 1):
        """Start continuous vmstat collection."""
        self._running = True

        cmd = f"vmstat {interval}"

        if self.ssh_config:
            full_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                cmd
            ]
        else:
            full_cmd = cmd.split()

        self._process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop vmstat collection."""
        self._running = False

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        if self._thread:
            self._thread.join(timeout=5)

    def _read_loop(self):
        """Read vmstat output continuously."""
        if not self._process:
            return

        header_found = False
        column_names = []

        for line in self._process.stdout:
            if not self._running:
                break

            line = line.strip()
            if not line:
                continue

            # Skip procs/memory/swap header line
            if line.startswith('procs'):
                continue

            # Capture column headers
            if line.startswith('r ') or line.startswith(' r'):
                column_names = line.split()
                header_found = True
                continue

            if not header_found:
                continue

            # Parse data line
            self._parse_line(line, column_names)

    def _parse_line(self, line: str, columns: List[str]):
        """Parse a vmstat data line."""
        parts = line.split()

        if len(parts) < len(columns):
            return

        try:
            data = {}

            # Map columns to values
            for i, col in enumerate(columns):
                if i < len(parts):
                    try:
                        data[col] = int(parts[i])
                    except ValueError:
                        data[col] = parts[i]

            # Build structured result
            result = {
                'procs': {
                    'runnable': data.get('r', 0),
                    'blocked': data.get('b', 0),
                },
                'memory': {
                    'swapped_kb': data.get('swpd', 0),
                    'free_kb': data.get('free', 0),
                    'buffer_kb': data.get('buff', 0),
                    'cache_kb': data.get('cache', 0),
                },
                'swap': {
                    'in_per_sec': data.get('si', 0),
                    'out_per_sec': data.get('so', 0),
                },
                'io': {
                    'blocks_in': data.get('bi', 0),
                    'blocks_out': data.get('bo', 0),
                },
                'system': {
                    'interrupts': data.get('in', 0),
                    'context_switches': data.get('cs', 0),
                },
                'cpu': {
                    'user_pct': data.get('us', 0),
                    'system_pct': data.get('sy', 0),
                    'idle_pct': data.get('id', 0),
                    'wait_pct': data.get('wa', 0),
                    'stolen_pct': data.get('st', 0),
                },
            }

            with self._lock:
                self._latest = result

        except Exception:
            pass

    def get_latest(self) -> Dict[str, Any]:
        """Get latest vmstat readings."""
        with self._lock:
            return self._latest.copy()

    def collect_once(self) -> Dict[str, Any]:
        """Collect a single vmstat sample."""
        cmd = "vmstat 1 2"

        try:
            if self.ssh_config:
                full_cmd = [
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                    cmd
                ]
                result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=10)
            else:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                # Parse last data line (skip first which is since boot)
                if len(lines) >= 3:
                    # Find column header
                    columns = []
                    for line in lines:
                        if line.strip().startswith('r ') or ' r ' in line[:10]:
                            columns = line.split()
                            break

                    # Parse last data line
                    if columns and len(lines) >= 4:
                        self._parse_line(lines[-1], columns)
                        return self._latest.copy()

        except Exception:
            pass

        return {}

    def get_cpu_summary(self) -> Dict[str, float]:
        """Get CPU utilization summary."""
        with self._lock:
            return self._latest.get('cpu', {})

    def get_memory_summary(self) -> Dict[str, int]:
        """Get memory usage summary."""
        with self._lock:
            return self._latest.get('memory', {})
