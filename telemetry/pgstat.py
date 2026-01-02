"""
PgStatCollector - Collects PostgreSQL performance statistics.

Provides:
- pg_stat_database metrics
- pg_stat_bgwriter metrics
- pg_stat_wal metrics (PG 14+)
- Connection and lock stats
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2


class PgStatCollector:
    """
    Collects PostgreSQL internal statistics.

    Uses pg_stat_* system views for performance metrics.
    """

    def __init__(self, connection=None):
        self.conn = connection
        self._baseline: Optional[Dict[str, Any]] = None

    def set_connection(self, connection):
        """Set or update database connection."""
        self.conn = connection

    def capture_baseline(self):
        """Capture baseline stats for delta calculation."""
        self._baseline = self.collect_snapshot()

    def collect_snapshot(self) -> Dict[str, Any]:
        """Collect current PostgreSQL statistics snapshot."""
        if not self.conn:
            return {}

        return {
            'database': self._get_database_stats(),
            'bgwriter': self._get_bgwriter_stats(),
            'wal': self._get_wal_stats(),
            'connections': self._get_connection_stats(),
            'locks': self._get_lock_stats(),
            'activity': self._get_activity_summary(),
            'statements': self._get_statement_stats(),
            'tables': self._get_table_stats(),
            'table_io': self._get_table_io_stats(),
            'indexes': self._get_index_stats(),
            'replication': self._get_replication_stats(),
            'checkpointer': self._get_checkpointer_stats(),
        }

    def _get_database_stats(self) -> Dict[str, Any]:
        """Get pg_stat_database stats for current database."""
        try:
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
                    blks_read = row[3] or 0
                    blks_hit = row[4] or 0
                    total_blocks = blks_read + blks_hit

                    return {
                        'numbackends': row[0],
                        'xact_commit': row[1],
                        'xact_rollback': row[2],
                        'blks_read': blks_read,
                        'blks_hit': blks_hit,
                        'cache_hit_ratio': (blks_hit / total_blocks * 100) if total_blocks > 0 else 0,
                        'tup_returned': row[5],
                        'tup_fetched': row[6],
                        'tup_inserted': row[7],
                        'tup_updated': row[8],
                        'tup_deleted': row[9],
                        'conflicts': row[10],
                        'temp_files': row[11],
                        'temp_bytes': row[12],
                        'deadlocks': row[13],
                        'blk_read_time_ms': row[14] or 0,
                        'blk_write_time_ms': row[15] or 0,
                    }
        except Exception:
            pass

        return {}

    def _get_bgwriter_stats(self) -> Dict[str, Any]:
        """Get pg_stat_bgwriter stats."""
        try:
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
                    total_checkpoints = (row[0] or 0) + (row[1] or 0)

                    return {
                        'checkpoints_timed': row[0],
                        'checkpoints_req': row[1],
                        'checkpoint_write_time_ms': row[2] or 0,
                        'checkpoint_sync_time_ms': row[3] or 0,
                        'buffers_checkpoint': row[4],
                        'buffers_clean': row[5],
                        'maxwritten_clean': row[6],
                        'buffers_backend': row[7],
                        'buffers_backend_fsync': row[8],
                        'buffers_alloc': row[9],
                        'checkpoint_req_ratio': (row[1] / total_checkpoints * 100) if total_checkpoints > 0 else 0,
                    }
        except Exception:
            pass

        return {}

    def _get_wal_stats(self) -> Optional[Dict[str, Any]]:
        """Get pg_stat_wal stats (PostgreSQL 14+)."""
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
                        'wal_fpi': row[1],  # Full page images
                        'wal_bytes': row[2],
                        'wal_buffers_full': row[3],
                        'wal_write': row[4],
                        'wal_sync': row[5],
                        'wal_write_time_ms': row[6] or 0,
                        'wal_sync_time_ms': row[7] or 0,
                    }
        except Exception:
            # pg_stat_wal doesn't exist in PG < 14
            pass

        return None

    def _get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        try:
            with self.conn.cursor() as cur:
                # Get max connections
                cur.execute("SHOW max_connections")
                max_conn = int(cur.fetchone()[0])

                # Get current connections by state
                cur.execute("""
                    SELECT
                        state,
                        count(*)
                    FROM pg_stat_activity
                    WHERE pid != pg_backend_pid()
                    GROUP BY state
                """)

                states = {}
                total = 0
                for row in cur.fetchall():
                    state = row[0] or 'null'
                    count = row[1]
                    states[state] = count
                    total += count

                return {
                    'max_connections': max_conn,
                    'total_connections': total,
                    'connection_pct': (total / max_conn * 100) if max_conn > 0 else 0,
                    'by_state': states,
                }
        except Exception:
            pass

        return {}

    def _get_lock_stats(self) -> Dict[str, Any]:
        """Get lock statistics."""
        try:
            with self.conn.cursor() as cur:
                # Get lock counts by type
                cur.execute("""
                    SELECT
                        mode,
                        count(*),
                        sum(case when granted then 1 else 0 end) as granted,
                        sum(case when not granted then 1 else 0 end) as waiting
                    FROM pg_locks
                    GROUP BY mode
                """)

                locks = {}
                total_waiting = 0
                for row in cur.fetchall():
                    locks[row[0]] = {
                        'total': row[1],
                        'granted': row[2],
                        'waiting': row[3],
                    }
                    total_waiting += row[3] or 0

                return {
                    'by_mode': locks,
                    'total_waiting': total_waiting,
                }
        except Exception:
            pass

        return {}

    def _get_activity_summary(self) -> Dict[str, Any]:
        """Get summary of current activity."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        count(*) FILTER (WHERE state = 'active') as active,
                        count(*) FILTER (WHERE state = 'idle') as idle,
                        count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_txn,
                        count(*) FILTER (WHERE wait_event_type = 'Lock') as waiting_on_lock,
                        count(*) FILTER (WHERE wait_event_type = 'IO') as waiting_on_io,
                        max(extract(epoch from (now() - xact_start))) as longest_txn_sec,
                        max(extract(epoch from (now() - query_start)))
                            FILTER (WHERE state = 'active') as longest_query_sec
                    FROM pg_stat_activity
                    WHERE pid != pg_backend_pid()
                """)
                row = cur.fetchone()
                if row:
                    return {
                        'active': row[0] or 0,
                        'idle': row[1] or 0,
                        'idle_in_transaction': row[2] or 0,
                        'waiting_on_lock': row[3] or 0,
                        'waiting_on_io': row[4] or 0,
                        'longest_txn_sec': row[5] or 0,
                        'longest_query_sec': row[6] or 0,
                    }
        except Exception:
            pass

        return {}

    def get_delta(self) -> Dict[str, Any]:
        """Get delta from baseline."""
        if not self._baseline:
            return {}

        current = self.collect_snapshot()

        return {
            'database': self._calculate_delta(
                self._baseline.get('database', {}),
                current.get('database', {}),
                ['xact_commit', 'xact_rollback', 'blks_read', 'blks_hit',
                 'tup_returned', 'tup_fetched', 'tup_inserted', 'tup_updated',
                 'tup_deleted', 'temp_files', 'temp_bytes', 'deadlocks']
            ),
            'bgwriter': self._calculate_delta(
                self._baseline.get('bgwriter', {}),
                current.get('bgwriter', {}),
                ['checkpoints_timed', 'checkpoints_req', 'buffers_checkpoint',
                 'buffers_clean', 'buffers_backend', 'buffers_alloc']
            ),
        }

    def _calculate_delta(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
        counters: List[str]
    ) -> Dict[str, Any]:
        """Calculate delta for counter metrics."""
        delta = {}

        for key in counters:
            if key in baseline and key in current:
                delta[key] = current[key] - baseline[key]

        return delta

    def _get_statement_stats(self) -> Optional[Dict[str, Any]]:
        """Get pg_stat_statements stats (if extension installed)."""
        try:
            with self.conn.cursor() as cur:
                # Check if extension exists
                cur.execute("""
                    SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                """)
                if not cur.fetchone():
                    return None

                # Get top queries by total time
                cur.execute("""
                    SELECT
                        queryid,
                        LEFT(query, 200) as query_preview,
                        calls,
                        total_exec_time as total_time_ms,
                        mean_exec_time as mean_time_ms,
                        min_exec_time as min_time_ms,
                        max_exec_time as max_time_ms,
                        stddev_exec_time as stddev_time_ms,
                        rows,
                        shared_blks_hit,
                        shared_blks_read,
                        shared_blks_dirtied,
                        shared_blks_written,
                        local_blks_hit,
                        local_blks_read,
                        temp_blks_read,
                        temp_blks_written,
                        wal_records,
                        wal_bytes
                    FROM pg_stat_statements
                    WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
                    ORDER BY total_exec_time DESC
                    LIMIT 10
                """)

                top_queries = []
                for row in cur.fetchall():
                    shared_hit = row[9] or 0
                    shared_read = row[10] or 0
                    total_shared = shared_hit + shared_read

                    top_queries.append({
                        'queryid': row[0],
                        'query_preview': row[1],
                        'calls': row[2],
                        'total_time_ms': row[3] or 0,
                        'mean_time_ms': row[4] or 0,
                        'min_time_ms': row[5] or 0,
                        'max_time_ms': row[6] or 0,
                        'stddev_time_ms': row[7] or 0,
                        'rows': row[8],
                        'shared_blks_hit': shared_hit,
                        'shared_blks_read': shared_read,
                        'cache_hit_ratio': (shared_hit / total_shared * 100) if total_shared > 0 else 100,
                        'shared_blks_dirtied': row[11],
                        'shared_blks_written': row[12],
                        'temp_blks_read': row[15],
                        'temp_blks_written': row[16],
                        'wal_records': row[17],
                        'wal_bytes': row[18],
                    })

                # Get aggregate stats
                cur.execute("""
                    SELECT
                        count(*) as total_queries,
                        sum(calls) as total_calls,
                        sum(total_exec_time) as total_time_ms,
                        sum(rows) as total_rows,
                        sum(shared_blks_hit) as total_shared_hit,
                        sum(shared_blks_read) as total_shared_read
                    FROM pg_stat_statements
                    WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
                """)
                agg = cur.fetchone()
                total_shared = (agg[4] or 0) + (agg[5] or 0)

                return {
                    'total_queries': agg[0] or 0,
                    'total_calls': agg[1] or 0,
                    'total_time_ms': agg[2] or 0,
                    'total_rows': agg[3] or 0,
                    'overall_cache_hit_ratio': ((agg[4] or 0) / total_shared * 100) if total_shared > 0 else 100,
                    'top_by_time': top_queries,
                }

        except Exception:
            pass

        return None

    def _get_table_stats(self) -> Dict[str, Any]:
        """Get pg_stat_user_tables stats."""
        try:
            with self.conn.cursor() as cur:
                # Get per-table stats (top 20 by activity)
                cur.execute("""
                    SELECT
                        schemaname,
                        relname,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch,
                        n_tup_ins,
                        n_tup_upd,
                        n_tup_del,
                        n_tup_hot_upd,
                        n_live_tup,
                        n_dead_tup,
                        n_mod_since_analyze,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze,
                        vacuum_count,
                        autovacuum_count,
                        analyze_count,
                        autoanalyze_count
                    FROM pg_stat_user_tables
                    ORDER BY (seq_scan + COALESCE(idx_scan, 0)) DESC
                    LIMIT 20
                """)

                tables = []
                for row in cur.fetchall():
                    seq_scan = row[2] or 0
                    idx_scan = row[4] or 0
                    total_scan = seq_scan + idx_scan
                    live_tup = row[10] or 0
                    dead_tup = row[11] or 0
                    total_tup = live_tup + dead_tup

                    tables.append({
                        'schema': row[0],
                        'table': row[1],
                        'seq_scan': seq_scan,
                        'seq_tup_read': row[3] or 0,
                        'idx_scan': idx_scan,
                        'idx_tup_fetch': row[5] or 0,
                        'index_usage_ratio': (idx_scan / total_scan * 100) if total_scan > 0 else 0,
                        'n_tup_ins': row[6] or 0,
                        'n_tup_upd': row[7] or 0,
                        'n_tup_del': row[8] or 0,
                        'n_tup_hot_upd': row[9] or 0,
                        'hot_update_ratio': (row[9] / row[7] * 100) if row[7] and row[7] > 0 else 0,
                        'n_live_tup': live_tup,
                        'n_dead_tup': dead_tup,
                        'dead_tuple_ratio': (dead_tup / total_tup * 100) if total_tup > 0 else 0,
                        'n_mod_since_analyze': row[12] or 0,
                        'last_vacuum': str(row[13]) if row[13] else None,
                        'last_autovacuum': str(row[14]) if row[14] else None,
                        'last_analyze': str(row[15]) if row[15] else None,
                        'vacuum_count': row[17] or 0,
                        'autovacuum_count': row[18] or 0,
                    })

                # Get aggregate stats
                cur.execute("""
                    SELECT
                        sum(seq_scan) as total_seq_scan,
                        sum(idx_scan) as total_idx_scan,
                        sum(n_live_tup) as total_live_tup,
                        sum(n_dead_tup) as total_dead_tup,
                        sum(n_tup_ins) as total_ins,
                        sum(n_tup_upd) as total_upd,
                        sum(n_tup_del) as total_del
                    FROM pg_stat_user_tables
                """)
                agg = cur.fetchone()
                total_scan = (agg[0] or 0) + (agg[1] or 0)
                total_tup = (agg[2] or 0) + (agg[3] or 0)

                return {
                    'total_seq_scan': agg[0] or 0,
                    'total_idx_scan': agg[1] or 0,
                    'overall_index_usage_ratio': ((agg[1] or 0) / total_scan * 100) if total_scan > 0 else 0,
                    'total_live_tup': agg[2] or 0,
                    'total_dead_tup': agg[3] or 0,
                    'overall_dead_tuple_ratio': ((agg[3] or 0) / total_tup * 100) if total_tup > 0 else 0,
                    'total_inserts': agg[4] or 0,
                    'total_updates': agg[5] or 0,
                    'total_deletes': agg[6] or 0,
                    'top_tables': tables,
                }

        except Exception:
            pass

        return {}

    def _get_table_io_stats(self) -> Dict[str, Any]:
        """Get pg_statio_user_tables stats (buffer cache effectiveness)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        schemaname,
                        relname,
                        heap_blks_read,
                        heap_blks_hit,
                        idx_blks_read,
                        idx_blks_hit,
                        toast_blks_read,
                        toast_blks_hit,
                        tidx_blks_read,
                        tidx_blks_hit
                    FROM pg_statio_user_tables
                    WHERE (heap_blks_read + heap_blks_hit) > 0
                    ORDER BY (heap_blks_read + heap_blks_hit) DESC
                    LIMIT 20
                """)

                tables = []
                for row in cur.fetchall():
                    heap_read = row[2] or 0
                    heap_hit = row[3] or 0
                    idx_read = row[4] or 0
                    idx_hit = row[5] or 0
                    total_heap = heap_read + heap_hit
                    total_idx = idx_read + idx_hit

                    tables.append({
                        'schema': row[0],
                        'table': row[1],
                        'heap_blks_read': heap_read,
                        'heap_blks_hit': heap_hit,
                        'heap_cache_hit_ratio': (heap_hit / total_heap * 100) if total_heap > 0 else 100,
                        'idx_blks_read': idx_read,
                        'idx_blks_hit': idx_hit,
                        'idx_cache_hit_ratio': (idx_hit / total_idx * 100) if total_idx > 0 else 100,
                        'toast_blks_read': row[6] or 0,
                        'toast_blks_hit': row[7] or 0,
                    })

                # Get aggregate stats
                cur.execute("""
                    SELECT
                        sum(heap_blks_read) as total_heap_read,
                        sum(heap_blks_hit) as total_heap_hit,
                        sum(idx_blks_read) as total_idx_read,
                        sum(idx_blks_hit) as total_idx_hit
                    FROM pg_statio_user_tables
                """)
                agg = cur.fetchone()
                total_heap = (agg[0] or 0) + (agg[1] or 0)
                total_idx = (agg[2] or 0) + (agg[3] or 0)

                return {
                    'total_heap_blks_read': agg[0] or 0,
                    'total_heap_blks_hit': agg[1] or 0,
                    'heap_cache_hit_ratio': ((agg[1] or 0) / total_heap * 100) if total_heap > 0 else 100,
                    'total_idx_blks_read': agg[2] or 0,
                    'total_idx_blks_hit': agg[3] or 0,
                    'idx_cache_hit_ratio': ((agg[3] or 0) / total_idx * 100) if total_idx > 0 else 100,
                    'top_tables': tables,
                }

        except Exception:
            pass

        return {}

    def _get_index_stats(self) -> Dict[str, Any]:
        """Get pg_stat_user_indexes stats."""
        try:
            with self.conn.cursor() as cur:
                # Most used indexes
                cur.execute("""
                    SELECT
                        schemaname,
                        relname,
                        indexrelname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE idx_scan > 0
                    ORDER BY idx_scan DESC
                    LIMIT 15
                """)

                used_indexes = []
                for row in cur.fetchall():
                    used_indexes.append({
                        'schema': row[0],
                        'table': row[1],
                        'index': row[2],
                        'scans': row[3],
                        'tuples_read': row[4],
                        'tuples_fetched': row[5],
                    })

                # Unused indexes (potential waste)
                cur.execute("""
                    SELECT
                        schemaname,
                        relname,
                        indexrelname,
                        pg_relation_size(indexrelid) as index_size
                    FROM pg_stat_user_indexes
                    WHERE idx_scan = 0
                    AND indexrelname NOT LIKE '%_pkey'
                    ORDER BY pg_relation_size(indexrelid) DESC
                    LIMIT 10
                """)

                unused_indexes = []
                for row in cur.fetchall():
                    unused_indexes.append({
                        'schema': row[0],
                        'table': row[1],
                        'index': row[2],
                        'size_bytes': row[3],
                    })

                # Get aggregate stats
                cur.execute("""
                    SELECT
                        count(*) as total_indexes,
                        count(*) FILTER (WHERE idx_scan = 0) as unused_indexes,
                        sum(idx_scan) as total_scans
                    FROM pg_stat_user_indexes
                """)
                agg = cur.fetchone()

                return {
                    'total_indexes': agg[0] or 0,
                    'unused_indexes': agg[1] or 0,
                    'total_scans': agg[2] or 0,
                    'most_used': used_indexes,
                    'unused': unused_indexes,
                }

        except Exception:
            pass

        return {}

    def _get_replication_stats(self) -> Optional[Dict[str, Any]]:
        """Get pg_stat_replication stats (if any replicas connected)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        pid,
                        usename,
                        application_name,
                        client_addr,
                        state,
                        sent_lsn,
                        write_lsn,
                        flush_lsn,
                        replay_lsn,
                        write_lag,
                        flush_lag,
                        replay_lag,
                        sync_state
                    FROM pg_stat_replication
                """)

                replicas = []
                for row in cur.fetchall():
                    replicas.append({
                        'pid': row[0],
                        'user': row[1],
                        'application': row[2],
                        'client_addr': str(row[3]) if row[3] else None,
                        'state': row[4],
                        'sent_lsn': str(row[5]) if row[5] else None,
                        'write_lsn': str(row[6]) if row[6] else None,
                        'flush_lsn': str(row[7]) if row[7] else None,
                        'replay_lsn': str(row[8]) if row[8] else None,
                        'write_lag': str(row[9]) if row[9] else None,
                        'flush_lag': str(row[10]) if row[10] else None,
                        'replay_lag': str(row[11]) if row[11] else None,
                        'sync_state': row[12],
                    })

                if not replicas:
                    return None

                return {
                    'replica_count': len(replicas),
                    'replicas': replicas,
                }

        except Exception:
            pass

        return None

    def _get_checkpointer_stats(self) -> Optional[Dict[str, Any]]:
        """Get pg_stat_checkpointer stats (PostgreSQL 17+)."""
        try:
            with self.conn.cursor() as cur:
                # Check if pg_stat_checkpointer exists (PG 17+)
                cur.execute("""
                    SELECT 1 FROM pg_catalog.pg_class
                    WHERE relname = 'pg_stat_checkpointer'
                    AND relnamespace = 'pg_catalog'::regnamespace
                """)
                if not cur.fetchone():
                    return None

                cur.execute("""
                    SELECT
                        num_timed,
                        num_requested,
                        restartpoints_timed,
                        restartpoints_req,
                        restartpoints_done,
                        write_time,
                        sync_time,
                        buffers_written
                    FROM pg_stat_checkpointer
                """)
                row = cur.fetchone()
                if row:
                    return {
                        'num_timed': row[0],
                        'num_requested': row[1],
                        'restartpoints_timed': row[2],
                        'restartpoints_req': row[3],
                        'restartpoints_done': row[4],
                        'write_time_ms': row[5] or 0,
                        'sync_time_ms': row[6] or 0,
                        'buffers_written': row[7],
                    }

        except Exception:
            pass

        return None
