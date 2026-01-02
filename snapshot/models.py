"""
Data models for the snapshot/restore system.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import uuid


@dataclass
class SnapshotData:
    """PostgreSQL configuration state at a point in time."""

    # Identity
    id: str
    name: str                              # "initial", "before_round_1", custom name
    created_at: str                        # ISO timestamp

    # PostgreSQL Configuration
    pg_settings: Dict[str, str]            # All tunable params from pg_settings
    pg_auto_conf: str                      # Contents of postgresql.auto.conf
    pg_conf_hash: str                      # MD5 of postgresql.conf (detect manual edits)

    # Benchmark State (for comparison)
    last_tps: Optional[float] = None       # TPS at snapshot time
    last_latency: Optional[float] = None   # Latency at snapshot time

    # Metadata
    session_name: str = ""                 # Which session created this
    round_num: int = 0                     # Which round (0 = before any tuning)
    trigger: str = "manual"                # "automatic", "manual", "checkpoint"

    @classmethod
    def create(
        cls,
        name: str,
        pg_settings: Dict[str, str],
        pg_auto_conf: str,
        pg_conf_hash: str,
        trigger: str = "manual",
        session_name: str = "",
        round_num: int = 0,
        last_tps: Optional[float] = None,
        last_latency: Optional[float] = None,
    ) -> 'SnapshotData':
        """Create a new snapshot with generated ID and timestamp."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            created_at=datetime.now().isoformat(),
            pg_settings=pg_settings,
            pg_auto_conf=pg_auto_conf,
            pg_conf_hash=pg_conf_hash,
            last_tps=last_tps,
            last_latency=last_latency,
            session_name=session_name,
            round_num=round_num,
            trigger=trigger,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SnapshotData':
        """Create from dictionary (JSON deserialization)."""
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save snapshot to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'SnapshotData':
        """Load snapshot from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class SnapshotInfo:
    """Summary info for listing snapshots."""
    name: str
    created_at: str
    trigger: str
    session_name: str
    round_num: int
    last_tps: Optional[float]
    param_count: int


@dataclass
class SettingChange:
    """A single parameter change for restore preview."""
    param: str
    current_value: str
    snapshot_value: str
    requires_restart: bool = False


@dataclass
class RestorePreview:
    """Preview of what restore would change."""
    changes: List[SettingChange] = field(default_factory=list)
    restart_required: bool = False
    total_changes: int = 0

    def __post_init__(self):
        self.total_changes = len(self.changes)
        self.restart_required = any(c.requires_restart for c in self.changes)


@dataclass
class RestoreResult:
    """Result of a restore operation."""
    success: bool
    changes_applied: int = 0
    restart_required: bool = False
    commands_executed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    preview: Optional[RestorePreview] = None

    @classmethod
    def from_preview(cls, preview: RestorePreview, dry_run: bool = False) -> 'RestoreResult':
        """Create result from preview (for dry-run)."""
        return cls(
            success=True,
            changes_applied=0 if dry_run else preview.total_changes,
            restart_required=preview.restart_required,
            commands_executed=[],
            preview=preview,
        )

    @classmethod
    def failure(cls, error: str) -> 'RestoreResult':
        """Create a failure result."""
        return cls(success=False, error=error)


@dataclass
class SnapshotDiff:
    """Difference between two snapshots."""
    snapshot_a: str
    snapshot_b: str
    changes: List[SettingChange] = field(default_factory=list)
    only_in_a: List[str] = field(default_factory=list)
    only_in_b: List[str] = field(default_factory=list)

    @property
    def total_differences(self) -> int:
        return len(self.changes) + len(self.only_in_a) + len(self.only_in_b)


# Parameters that require PostgreSQL restart (context = 'postmaster')
RESTART_REQUIRED_PARAMS = {
    'shared_buffers',
    'max_connections',
    'wal_buffers',
    'huge_pages',
    'max_prepared_transactions',
    'max_locks_per_transaction',
    'max_pred_locks_per_transaction',
    'max_worker_processes',
    'max_parallel_workers',
    'wal_level',
    'archive_mode',
    'max_wal_senders',
    'max_replication_slots',
    'track_commit_timestamp',
    'ssl',
    'shared_preload_libraries',
}


def requires_restart(param: str) -> bool:
    """Check if a parameter requires restart to take effect."""
    return param in RESTART_REQUIRED_PARAMS
