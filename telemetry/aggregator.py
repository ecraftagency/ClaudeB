"""
TelemetryAggregator - Aggregates time-series telemetry for AI analysis.

Computes:
- Averages, min, max, percentiles
- Trend detection
- Anomaly identification
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics


@dataclass
class MetricSummary:
    """Summary statistics for a single metric."""
    name: str
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float
    samples: int


class TelemetryAggregator:
    """
    Aggregates time-aligned telemetry samples for AI consumption.

    Produces concise summaries suitable for context-limited LLMs.
    """

    def aggregate(self, samples: List) -> Dict[str, Any]:
        """
        Aggregate multiple TimeAlignedSamples.

        Args:
            samples: List of TimeAlignedSample objects

        Returns:
            Aggregated summary dictionary
        """
        if not samples:
            return {}

        return {
            'duration_ms': self._get_duration(samples),
            'sample_count': len(samples),
            'iostat': self._aggregate_iostat(samples),
            'vmstat': self._aggregate_vmstat(samples),
            'pgstat': self._aggregate_pgstat(samples),
            'sysstat': self._aggregate_sysstat(samples),
        }

    def _get_duration(self, samples: List) -> int:
        """Calculate total duration from samples."""
        if len(samples) < 2:
            return 0
        return samples[-1].offset_ms - samples[0].offset_ms

    def _aggregate_iostat(self, samples: List) -> Dict[str, Any]:
        """Aggregate iostat metrics across samples."""
        # Collect all device metrics
        device_metrics: Dict[str, Dict[str, List[float]]] = {}

        for sample in samples:
            if not sample.iostat:
                continue

            for device, metrics in sample.iostat.items():
                if device not in device_metrics:
                    device_metrics[device] = {
                        'r_per_sec': [],
                        'w_per_sec': [],
                        'rkb_per_sec': [],
                        'wkb_per_sec': [],
                        'await_ms': [],
                        'util_pct': [],
                    }

                for metric in device_metrics[device].keys():
                    if metric in metrics:
                        device_metrics[device][metric].append(metrics[metric])

        # Compute summaries per device
        result = {}
        for device, metrics in device_metrics.items():
            result[device] = {}
            for metric, values in metrics.items():
                if values:
                    result[device][metric] = self._compute_summary(metric, values)

        # Compute totals across all devices
        if device_metrics:
            result['_totals'] = self._compute_totals(device_metrics)

        return result

    def _aggregate_vmstat(self, samples: List) -> Dict[str, Any]:
        """Aggregate vmstat metrics across samples."""
        cpu_metrics = {
            'user_pct': [],
            'system_pct': [],
            'idle_pct': [],
            'wait_pct': [],
        }

        memory_metrics = {
            'free_kb': [],
            'buffer_kb': [],
            'cache_kb': [],
        }

        for sample in samples:
            if not sample.vmstat:
                continue

            cpu = sample.vmstat.get('cpu', {})
            for metric in cpu_metrics.keys():
                if metric in cpu:
                    cpu_metrics[metric].append(cpu[metric])

            memory = sample.vmstat.get('memory', {})
            for metric in memory_metrics.keys():
                if metric in memory:
                    memory_metrics[metric].append(memory[metric])

        return {
            'cpu': {
                metric: self._compute_summary(metric, values)
                for metric, values in cpu_metrics.items() if values
            },
            'memory': {
                metric: self._compute_summary(metric, values)
                for metric, values in memory_metrics.items() if values
            },
        }

    def _aggregate_pgstat(self, samples: List) -> Dict[str, Any]:
        """Aggregate PostgreSQL stats across samples."""
        if len(samples) < 2:
            # Need at least 2 samples for delta
            if samples:
                return samples[-1].pgstat
            return {}

        # Calculate deltas between first and last sample
        first = samples[0].pgstat
        last = samples[-1].pgstat

        if not first or not last:
            return {}

        duration_sec = (samples[-1].offset_ms - samples[0].offset_ms) / 1000
        if duration_sec <= 0:
            duration_sec = 1

        # Calculate rates
        db_first = first.get('database', {})
        db_last = last.get('database', {})

        tps = 0
        if 'xact_commit' in db_first and 'xact_commit' in db_last:
            commits = db_last['xact_commit'] - db_first['xact_commit']
            rollbacks = (db_last.get('xact_rollback', 0) -
                        db_first.get('xact_rollback', 0))
            tps = (commits + rollbacks) / duration_sec

        # Cache hit tracking
        cache_hits = []
        for sample in samples:
            db = sample.pgstat.get('database', {})
            if 'cache_hit_ratio' in db:
                cache_hits.append(db['cache_hit_ratio'])

        # Get table and index stats from last sample
        tables = last.get('tables', {})
        table_io = last.get('table_io', {})
        indexes = last.get('indexes', {})
        statements = last.get('statements', {})

        return {
            'duration_sec': duration_sec,
            'tps': round(tps, 2),
            'cache_hit_ratio': {
                'avg': statistics.mean(cache_hits) if cache_hits else 0,
                'min': min(cache_hits) if cache_hits else 0,
            },
            'delta': {
                'xact_commit': db_last.get('xact_commit', 0) - db_first.get('xact_commit', 0),
                'xact_rollback': db_last.get('xact_rollback', 0) - db_first.get('xact_rollback', 0),
                'blks_read': db_last.get('blks_read', 0) - db_first.get('blks_read', 0),
                'blks_hit': db_last.get('blks_hit', 0) - db_first.get('blks_hit', 0),
                'temp_files': db_last.get('temp_files', 0) - db_first.get('temp_files', 0),
                'deadlocks': db_last.get('deadlocks', 0) - db_first.get('deadlocks', 0),
            },
            'final_state': {
                'connections': last.get('connections', {}),
                'activity': last.get('activity', {}),
            },
            'tables': {
                'total_seq_scan': tables.get('total_seq_scan', 0),
                'total_idx_scan': tables.get('total_idx_scan', 0),
                'index_usage_ratio': tables.get('overall_index_usage_ratio', 0),
                'dead_tuple_ratio': tables.get('overall_dead_tuple_ratio', 0),
                'top_tables': tables.get('top_tables', [])[:5],  # Top 5 only
            },
            'table_io': {
                'heap_cache_hit_ratio': table_io.get('heap_cache_hit_ratio', 0),
                'idx_cache_hit_ratio': table_io.get('idx_cache_hit_ratio', 0),
            },
            'indexes': {
                'total_indexes': indexes.get('total_indexes', 0),
                'unused_indexes': indexes.get('unused_indexes', 0),
                'most_used': indexes.get('most_used', [])[:5],
            },
            'statements': statements,  # Include pg_stat_statements if available
        }

    def _aggregate_sysstat(self, samples: List) -> Dict[str, Any]:
        """Aggregate system stats across samples."""
        # Get samples that have sysstat data (collected less frequently)
        sysstat_samples = [s for s in samples if s.sysstat]

        if not sysstat_samples:
            return {}

        # Collect memory metrics
        mem_used_pct = []
        dirty_kb = []
        swap_used_pct = []
        load_1min = []

        for sample in sysstat_samples:
            meminfo = sample.sysstat.get('meminfo', {})
            if meminfo:
                if 'mem_used_pct' in meminfo:
                    mem_used_pct.append(meminfo['mem_used_pct'])
                if 'dirty_kb' in meminfo:
                    dirty_kb.append(meminfo['dirty_kb'])
                if 'swap_used_pct' in meminfo:
                    swap_used_pct.append(meminfo['swap_used_pct'])

            load_avg = sample.sysstat.get('load_avg', {})
            if load_avg and 'load_1min' in load_avg:
                load_1min.append(load_avg['load_1min'])

        # Get network stats from last sample
        last_sysstat = sysstat_samples[-1].sysstat if sysstat_samples else {}
        network = last_sysstat.get('network', {})
        disk_space = last_sysstat.get('disk_space', {})

        return {
            'memory': {
                'used_pct': self._compute_summary('mem_used_pct', mem_used_pct) if mem_used_pct else {},
                'dirty_kb': self._compute_summary('dirty_kb', dirty_kb) if dirty_kb else {},
                'swap_used_pct': self._compute_summary('swap_used_pct', swap_used_pct) if swap_used_pct else {},
            },
            'load_avg': {
                'load_1min': self._compute_summary('load_1min', load_1min) if load_1min else {},
            },
            'network': {
                'tcp': network.get('tcp', {}),
                'tcp_ext': network.get('tcp_ext', {}),
            },
            'disk_space': disk_space,
        }

    def _compute_summary(self, name: str, values: List[float]) -> Dict[str, float]:
        """Compute summary statistics for a list of values."""
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            'min': round(min(values), 2),
            'max': round(max(values), 2),
            'avg': round(statistics.mean(values), 2),
            'p50': round(sorted_values[int(n * 0.5)], 2),
            'p95': round(sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1], 2),
            'p99': round(sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1], 2),
        }

    def _compute_totals(
        self,
        device_metrics: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Any]:
        """Compute totals across all devices."""
        totals = {
            'total_read_kb_sec': 0,
            'total_write_kb_sec': 0,
            'max_util_pct': 0,
            'avg_await_ms': 0,
        }

        await_values = []
        util_values = []

        for device, metrics in device_metrics.items():
            if 'rkb_per_sec' in metrics and metrics['rkb_per_sec']:
                totals['total_read_kb_sec'] += statistics.mean(metrics['rkb_per_sec'])

            if 'wkb_per_sec' in metrics and metrics['wkb_per_sec']:
                totals['total_write_kb_sec'] += statistics.mean(metrics['wkb_per_sec'])

            if 'util_pct' in metrics and metrics['util_pct']:
                util_values.extend(metrics['util_pct'])

            if 'await_ms' in metrics and metrics['await_ms']:
                await_values.extend(metrics['await_ms'])

        if util_values:
            totals['max_util_pct'] = max(util_values)

        if await_values:
            totals['avg_await_ms'] = statistics.mean(await_values)

        return {k: round(v, 2) for k, v in totals.items()}

    def format_for_ai(self, aggregated: Dict[str, Any]) -> str:
        """
        Format aggregated telemetry as concise text for AI.

        Produces a compact summary suitable for LLM context.
        """
        lines = []

        lines.append(f"Duration: {aggregated.get('duration_ms', 0)}ms")
        lines.append(f"Samples: {aggregated.get('sample_count', 0)}")

        # IO Summary
        iostat = aggregated.get('iostat', {})
        if '_totals' in iostat:
            totals = iostat['_totals']
            lines.append(f"\nI/O Totals:")
            lines.append(f"  Read: {totals.get('total_read_kb_sec', 0):.0f} KB/s")
            lines.append(f"  Write: {totals.get('total_write_kb_sec', 0):.0f} KB/s")
            lines.append(f"  Max Util: {totals.get('max_util_pct', 0):.1f}%")
            lines.append(f"  Avg Await: {totals.get('avg_await_ms', 0):.1f}ms")

        # CPU Summary
        vmstat = aggregated.get('vmstat', {})
        cpu = vmstat.get('cpu', {})
        if cpu:
            lines.append(f"\nCPU:")
            lines.append(f"  User: {cpu.get('user_pct', {}).get('avg', 0):.1f}%")
            lines.append(f"  System: {cpu.get('system_pct', {}).get('avg', 0):.1f}%")
            lines.append(f"  Wait: {cpu.get('wait_pct', {}).get('avg', 0):.1f}%")
            lines.append(f"  Idle: {cpu.get('idle_pct', {}).get('avg', 0):.1f}%")

        # System Stats (memory, load)
        sysstat = aggregated.get('sysstat', {})
        if sysstat:
            mem = sysstat.get('memory', {})
            if mem:
                lines.append(f"\nMemory:")
                if 'used_pct' in mem and mem['used_pct']:
                    lines.append(f"  Used: {mem['used_pct'].get('avg', 0):.1f}%")
                if 'dirty_kb' in mem and mem['dirty_kb']:
                    lines.append(f"  Dirty Pages: {mem['dirty_kb'].get('max', 0) / 1024:.1f} MB (max)")
                if 'swap_used_pct' in mem and mem['swap_used_pct']:
                    swap = mem['swap_used_pct'].get('max', 0)
                    if swap > 0:
                        lines.append(f"  Swap Used: {swap:.1f}%")

            load = sysstat.get('load_avg', {})
            if load and 'load_1min' in load and load['load_1min']:
                lines.append(f"\nLoad Average:")
                lines.append(f"  1min: {load['load_1min'].get('avg', 0):.2f} (max: {load['load_1min'].get('max', 0):.2f})")

            network = sysstat.get('network', {})
            tcp_ext = network.get('tcp_ext', {})
            if tcp_ext:
                retrans = tcp_ext.get('retransmits', 0)
                timeouts = tcp_ext.get('timeouts', 0)
                if retrans > 0 or timeouts > 0:
                    lines.append(f"\nNetwork Issues:")
                    if retrans > 0:
                        lines.append(f"  TCP Retransmits: {retrans}")
                    if timeouts > 0:
                        lines.append(f"  TCP Timeouts: {timeouts}")

        # PostgreSQL Summary
        pgstat = aggregated.get('pgstat', {})
        if pgstat:
            lines.append(f"\nPostgreSQL:")
            lines.append(f"  TPS: {pgstat.get('tps', 0):.0f}")

            cache = pgstat.get('cache_hit_ratio', {})
            lines.append(f"  Cache Hit: {cache.get('avg', 0):.1f}% (min: {cache.get('min', 0):.1f}%)")

            # Table stats
            tables = pgstat.get('tables', {})
            if tables:
                idx_ratio = tables.get('index_usage_ratio', 0)
                dead_ratio = tables.get('dead_tuple_ratio', 0)
                lines.append(f"  Index Usage: {idx_ratio:.1f}%")
                if dead_ratio > 5:  # Only show if significant
                    lines.append(f"  Dead Tuples: {dead_ratio:.1f}%")

            # Table I/O cache
            table_io = pgstat.get('table_io', {})
            if table_io:
                heap_hit = table_io.get('heap_cache_hit_ratio', 0)
                idx_hit = table_io.get('idx_cache_hit_ratio', 0)
                if heap_hit < 99 or idx_hit < 99:  # Only show if not perfect
                    lines.append(f"  Heap Cache Hit: {heap_hit:.1f}%")
                    lines.append(f"  Index Cache Hit: {idx_hit:.1f}%")

            # Index stats
            indexes = pgstat.get('indexes', {})
            if indexes:
                unused = indexes.get('unused_indexes', 0)
                if unused > 0:
                    lines.append(f"  Unused Indexes: {unused}")

            # pg_stat_statements
            statements = pgstat.get('statements', {})
            if statements:
                lines.append(f"\nTop Queries (by time):")
                top_queries = statements.get('top_by_time', [])[:3]
                for i, q in enumerate(top_queries, 1):
                    lines.append(f"  {i}. calls={q.get('calls', 0)}, "
                               f"total={q.get('total_time_ms', 0):.0f}ms, "
                               f"mean={q.get('mean_time_ms', 0):.2f}ms")
                    # Truncate query preview
                    preview = q.get('query_preview', '')[:80]
                    if preview:
                        lines.append(f"     {preview}...")

            delta = pgstat.get('delta', {})
            if delta.get('temp_files', 0) > 0:
                lines.append(f"  Temp Files Created: {delta['temp_files']}")
            if delta.get('deadlocks', 0) > 0:
                lines.append(f"  Deadlocks: {delta['deadlocks']}")

        return '\n'.join(lines)
