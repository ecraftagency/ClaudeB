"""
IOStatCollector - Collects disk I/O metrics via iostat.

Provides:
- Per-device read/write throughput
- IOPS and latency
- Queue depth and utilization
"""

import subprocess
import threading
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class IOStatSample:
    """Single iostat sample."""
    device: str
    r_per_sec: float  # reads/sec
    w_per_sec: float  # writes/sec
    rkb_per_sec: float  # KB read/sec
    wkb_per_sec: float  # KB written/sec
    await_ms: float  # avg wait time ms
    r_await_ms: float  # read wait time ms
    w_await_ms: float  # write wait time ms
    util_pct: float  # utilization %


class IOStatCollector:
    """
    Collects disk I/O statistics using iostat.

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
        """Start continuous iostat collection."""
        self._running = True

        cmd = f"iostat -xdm {interval}"

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
        """Stop iostat collection."""
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
        """Read iostat output continuously."""
        if not self._process:
            return

        current_block = []

        for line in self._process.stdout:
            if not self._running:
                break

            line = line.strip()

            # Empty line marks end of block
            if not line:
                if current_block:
                    self._parse_block(current_block)
                    current_block = []
            else:
                current_block.append(line)

    def _parse_block(self, lines: List[str]):
        """Parse a block of iostat output."""
        devices = {}
        header_found = False
        header_indices = {}

        for line in lines:
            # Skip timestamp lines
            if line.startswith('Linux') or '/' in line[:20]:
                continue

            # Find header line
            if line.startswith('Device'):
                header_found = True
                parts = line.split()
                for i, col in enumerate(parts):
                    header_indices[col.lower()] = i
                continue

            if not header_found:
                continue

            # Parse device line
            parts = line.split()
            if len(parts) < 5:
                continue

            device = parts[0]

            try:
                sample = {
                    'device': device,
                    'r_per_sec': self._safe_float(parts, header_indices.get('r/s', 1)),
                    'w_per_sec': self._safe_float(parts, header_indices.get('w/s', 2)),
                    'rkb_per_sec': self._safe_float(parts, header_indices.get('rmb/s', 3)) * 1024,
                    'wkb_per_sec': self._safe_float(parts, header_indices.get('wmb/s', 4)) * 1024,
                    'await_ms': self._safe_float(parts, header_indices.get('await', -3)),
                    'r_await_ms': self._safe_float(parts, header_indices.get('r_await', -4)),
                    'w_await_ms': self._safe_float(parts, header_indices.get('w_await', -3)),
                    'util_pct': self._safe_float(parts, header_indices.get('%util', -1)),
                }
                devices[device] = sample
            except (IndexError, ValueError):
                continue

        if devices:
            with self._lock:
                self._latest = devices

    def _safe_float(self, parts: List[str], index: int) -> float:
        """Safely extract float from parts list."""
        try:
            if 0 <= index < len(parts) or index < 0:
                return float(parts[index])
        except (IndexError, ValueError):
            pass
        return 0.0

    def get_latest(self) -> Dict[str, Any]:
        """Get latest iostat readings."""
        with self._lock:
            return self._latest.copy()

    def collect_once(self) -> Dict[str, Any]:
        """Collect a single iostat sample."""
        cmd = "iostat -xdm 1 2"

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
                # Parse the second block (first is since boot)
                lines = result.stdout.strip().split('\n')
                # Find second Device header
                device_count = 0
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('Device'):
                        device_count += 1
                        if device_count == 2:
                            start_idx = i
                            break

                if device_count >= 2:
                    self._parse_block(lines[start_idx:])
                    return self._latest.copy()

        except Exception:
            pass

        return {}

    def get_device_summary(self, device: str) -> Optional[Dict[str, float]]:
        """Get summary for a specific device."""
        with self._lock:
            return self._latest.get(device)
