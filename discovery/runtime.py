"""
RuntimeScanner - Scans PostgreSQL runtime configuration.

Collects current postgresql.conf settings and version information.
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2

from ..protocol.context import RuntimeContext
from .system import SystemScanner


# Key PostgreSQL parameters to collect
PG_IMPORTANT_PARAMS = [
    # Memory
    'shared_buffers',
    'effective_cache_size',
    'work_mem',
    'maintenance_work_mem',
    'wal_buffers',
    'huge_pages',

    # WAL
    'wal_level',
    'max_wal_size',
    'min_wal_size',
    'checkpoint_timeout',
    'checkpoint_completion_target',
    'wal_compression',

    # Connections
    'max_connections',
    'superuser_reserved_connections',

    # Query Planning
    'random_page_cost',
    'effective_io_concurrency',
    'default_statistics_target',

    # Parallel Query
    'max_parallel_workers_per_gather',
    'max_parallel_workers',
    'max_parallel_maintenance_workers',

    # Autovacuum
    'autovacuum',
    'autovacuum_max_workers',
    'autovacuum_naptime',
    'autovacuum_vacuum_scale_factor',
    'autovacuum_analyze_scale_factor',

    # Locking
    'deadlock_timeout',
    'lock_timeout',
    'statement_timeout',

    # Logging
    'log_min_duration_statement',
    'log_checkpoints',
    'log_lock_waits',

    # Replication (if applicable)
    'synchronous_commit',
    'max_wal_senders',
]


class RuntimeScanner:
    """
    Scans PostgreSQL runtime configuration.
    """

    def __init__(self, connection, system_scanner: Optional[SystemScanner] = None):
        self.conn = connection
        self.system_scanner = system_scanner or SystemScanner()

    def scan(self) -> RuntimeContext:
        """Perform full runtime configuration scan."""
        return RuntimeContext(
            active_config=self._get_active_config(),
            os_tuning=self.system_scanner.get_os_tuning(),
            load_average=self.system_scanner.get_load_average(),
            memory_usage=self._get_memory_usage(),
        )

    def _get_active_config(self) -> Dict[str, str]:
        """Get current PostgreSQL configuration values."""
        config = {}

        with self.conn.cursor() as cur:
            for param in PG_IMPORTANT_PARAMS:
                try:
                    cur.execute(f"SHOW {param}")
                    value = cur.fetchone()[0]
                    config[param] = value
                except Exception:
                    pass

        return config

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        mem_info = self.system_scanner._get_memory_info()
        return {
            'used_gb': mem_info.get('used_gb', 0),
            'cached_gb': mem_info.get('cached_gb', 0),
            'available_gb': mem_info.get('available_gb', 0),
        }

    def get_version(self) -> tuple:
        """Get PostgreSQL version (major, full)."""
        with self.conn.cursor() as cur:
            cur.execute("SHOW server_version")
            full_version = cur.fetchone()[0]

            cur.execute("SHOW server_version_num")
            version_num = int(cur.fetchone()[0])

            # Major version is first two digits for PG 10+
            major_version = version_num // 10000

        return major_version, full_version

    def get_all_settings(self) -> List[Dict[str, Any]]:
        """Get all PostgreSQL settings with metadata."""
        settings = []

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    name,
                    setting,
                    unit,
                    category,
                    short_desc,
                    context,
                    vartype,
                    source,
                    boot_val,
                    reset_val
                FROM pg_settings
                ORDER BY category, name
            """)

            for row in cur.fetchall():
                settings.append({
                    'name': row[0],
                    'setting': row[1],
                    'unit': row[2],
                    'category': row[3],
                    'description': row[4],
                    'context': row[5],
                    'type': row[6],
                    'source': row[7],
                    'boot_val': row[8],
                    'reset_val': row[9],
                })

        return settings

    def get_setting(self, param: str) -> Optional[str]:
        """Get a single setting value."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SHOW {param}")
                return cur.fetchone()[0]
        except Exception:
            return None

    def get_pg_stats_database(self) -> Dict[str, Any]:
        """Get pg_stat_database stats for current database."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    numbackends,
                    xact_commit,
                    xact_rollback,
                    blks_read,
                    blks_hit,
                    tup_returned,
                    tup_fetched,
                    tup_inserted,
                    tup_updated,
                    tup_deleted,
                    conflicts,
                    temp_files,
                    temp_bytes,
                    deadlocks,
                    blk_read_time,
                    blk_write_time
                FROM pg_stat_database
                WHERE datname = current_database()
            """)
            row = cur.fetchone()
            if row:
                return {
                    'numbackends': row[0],
                    'xact_commit': row[1],
                    'xact_rollback': row[2],
                    'blks_read': row[3],
                    'blks_hit': row[4],
                    'tup_returned': row[5],
                    'tup_fetched': row[6],
                    'tup_inserted': row[7],
                    'tup_updated': row[8],
                    'tup_deleted': row[9],
                    'conflicts': row[10],
                    'temp_files': row[11],
                    'temp_bytes': row[12],
                    'deadlocks': row[13],
                    'blk_read_time': row[14],
                    'blk_write_time': row[15],
                }
        return {}

    def get_pg_stats_bgwriter(self) -> Dict[str, Any]:
        """Get pg_stat_bgwriter stats."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    checkpoints_timed,
                    checkpoints_req,
                    checkpoint_write_time,
                    checkpoint_sync_time,
                    buffers_checkpoint,
                    buffers_clean,
                    maxwritten_clean,
                    buffers_backend,
                    buffers_backend_fsync,
                    buffers_alloc
                FROM pg_stat_bgwriter
            """)
            row = cur.fetchone()
            if row:
                return {
                    'checkpoints_timed': row[0],
                    'checkpoints_req': row[1],
                    'checkpoint_write_time': row[2],
                    'checkpoint_sync_time': row[3],
                    'buffers_checkpoint': row[4],
                    'buffers_clean': row[5],
                    'maxwritten_clean': row[6],
                    'buffers_backend': row[7],
                    'buffers_backend_fsync': row[8],
                    'buffers_alloc': row[9],
                }
        return {}

    def get_pg_stats_wal(self) -> Optional[Dict[str, Any]]:
        """Get pg_stat_wal stats (PG 14+)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        wal_records,
                        wal_fpi,
                        wal_bytes,
                        wal_buffers_full,
                        wal_write,
                        wal_sync,
                        wal_write_time,
                        wal_sync_time
                    FROM pg_stat_wal
                """)
                row = cur.fetchone()
                if row:
                    return {
                        'wal_records': row[0],
                        'wal_fpi': row[1],
                        'wal_bytes': row[2],
                        'wal_buffers_full': row[3],
                        'wal_write': row[4],
                        'wal_sync': row[5],
                        'wal_write_time': row[6],
                        'wal_sync_time': row[7],
                    }
        except Exception:
            # pg_stat_wal doesn't exist in older versions
            pass
        return None
