"""
TelemetryCollector - Orchestrates time-aligned metric collection.

v2.1 Feature: All metrics synchronized with relative timestamps.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..protocol.result import TelemetryPoint


@dataclass
class CollectorConfig:
    """Configuration for telemetry collection."""
    interval_ms: int = 1000  # Collection interval
    ssh_host: Optional[str] = None
    ssh_user: str = "ubuntu"
    ssh_key: Optional[str] = None


@dataclass
class TimeAlignedSample:
    """A single time-aligned sample from all sources."""
    relative_ts: str  # "T+0ms", "T+1000ms"
    absolute_ts: datetime
    offset_ms: int
    iostat: Dict[str, Any] = field(default_factory=dict)
    vmstat: Dict[str, Any] = field(default_factory=dict)
    pgstat: Dict[str, Any] = field(default_factory=dict)
    sysstat: Dict[str, Any] = field(default_factory=dict)


class TelemetryCollector:
    """
    Master telemetry collector with time alignment.

    Coordinates iostat, vmstat, and pg_stat collection with
    synchronized timestamps for accurate correlation.
    """

    def __init__(
        self,
        connection=None,
        config: Optional[CollectorConfig] = None,
        ssh_config: Optional[Dict[str, Any]] = None,
    ):
        self.conn = connection
        self.config = config or CollectorConfig()

        # Import child collectors
        from .iostat import IOStatCollector
        from .vmstat import VMStatCollector
        from .pgstat import PgStatCollector
        from .sysstat import SysStatCollector

        # Support both config object and ssh_config dict
        effective_ssh_config = ssh_config
        if not effective_ssh_config and self.config.ssh_host:
            effective_ssh_config = {
                'host': self.config.ssh_host,
                'user': self.config.ssh_user,
            }

        self.iostat = IOStatCollector(ssh_config=effective_ssh_config)
        self.vmstat = VMStatCollector(ssh_config=effective_ssh_config)
        self.pgstat = PgStatCollector(connection=connection)
        self.sysstat = SysStatCollector(ssh_config=effective_ssh_config)

        self._samples: List[TimeAlignedSample] = []
        self._start_time: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start background collection."""
        self._samples = []
        self._start_time = time.time()
        self._running = True

        # Start child collectors
        self.iostat.start(interval=self.config.interval_ms // 1000)
        self.vmstat.start(interval=self.config.interval_ms // 1000)

        # Start collection thread
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[TimeAlignedSample]:
        """Stop collection and return samples."""
        self._running = False

        # Stop child collectors
        self.iostat.stop()
        self.vmstat.stop()

        if self._thread:
            self._thread.join(timeout=5)

        return self._samples

    def _collection_loop(self):
        """Background collection loop."""
        while self._running:
            try:
                self._collect_sample()
            except Exception:
                pass

            # Sleep for interval
            time.sleep(self.config.interval_ms / 1000)

    def _collect_sample(self):
        """Collect a single time-aligned sample."""
        now = time.time()
        offset_ms = int((now - self._start_time) * 1000)

        # Collect sysstat less frequently (every 5 samples) to reduce overhead
        sysstat_data = {}
        if len(self._samples) % 5 == 0:
            sysstat_data = self.sysstat.collect_all()

        sample = TimeAlignedSample(
            relative_ts=f"T+{offset_ms}ms",
            absolute_ts=datetime.utcnow(),
            offset_ms=offset_ms,
            iostat=self.iostat.get_latest(),
            vmstat=self.vmstat.get_latest(),
            pgstat=self.pgstat.collect_snapshot(),
            sysstat=sysstat_data,
        )

        self._samples.append(sample)

    def collect_once(self) -> TimeAlignedSample:
        """Collect a single sample (for non-background use)."""
        if self._start_time is None:
            self._start_time = time.time()

        return TimeAlignedSample(
            relative_ts="T+0ms",
            absolute_ts=datetime.utcnow(),
            offset_ms=0,
            iostat=self.iostat.collect_once(),
            vmstat=self.vmstat.collect_once(),
            pgstat=self.pgstat.collect_snapshot(),
            sysstat=self.sysstat.collect_all(),
        )

    def get_samples(self) -> List[TimeAlignedSample]:
        """Get all collected samples."""
        return self._samples.copy()

    def to_telemetry_points(self) -> List[TelemetryPoint]:
        """Convert samples to protocol TelemetryPoint format."""
        points = []

        for sample in self._samples:
            point = TelemetryPoint(
                timestamp=sample.relative_ts,
                metrics={
                    'iostat': sample.iostat,
                    'vmstat': sample.vmstat,
                    'pgstat': sample.pgstat,
                    'sysstat': sample.sysstat,
                }
            )
            points.append(point)

        return points

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated summary of collected telemetry."""
        if not self._samples:
            return {}

        from .aggregator import TelemetryAggregator
        aggregator = TelemetryAggregator()

        return aggregator.aggregate(self._samples)

    def collect_snapshot(self) -> Dict[str, Any]:
        """Collect a single snapshot of all metrics (simple dict format)."""
        try:
            pg_stats = self.pgstat.collect_snapshot() if self.pgstat else {}
        except Exception:
            pg_stats = {}

        try:
            iostat = self.iostat.collect_once() if self.iostat else {}
        except Exception:
            iostat = {}

        try:
            vmstat = self.vmstat.collect_once() if self.vmstat else {}
        except Exception:
            vmstat = {}

        try:
            sysstat = self.sysstat.collect_all() if self.sysstat else {}
        except Exception:
            sysstat = {}

        return {
            'pg_stats': pg_stats,
            'iostat': iostat,
            'vmstat': vmstat,
            'sysstat': sysstat,
        }
