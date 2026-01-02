"""
Snapshot/Restore system for pg_diagnose.

Provides persistent configuration snapshots that enable safe experimentation
with PostgreSQL tuning. Key features:

- Automatic initial snapshot before first tuning change
- Manual checkpoints via /snapshot command
- One-click restore via /restore command
- Dry-run preview before restore

Scope: PostgreSQL configuration only (pg_settings + postgresql.auto.conf)
Future: OS params (sysctl), PgBouncer config
"""

from .models import SnapshotData, RestoreResult, RestorePreview, SnapshotInfo
from .capture import SnapshotCapture
from .restore import SnapshotRestore
from .manager import SnapshotManager

__all__ = [
    'SnapshotData',
    'RestoreResult',
    'RestorePreview',
    'SnapshotInfo',
    'SnapshotCapture',
    'SnapshotRestore',
    'SnapshotManager',
]
