"""
Telemetry module - Collects time-aligned performance metrics.

v2.1 Time-Aligned Telemetry:
- Relative timestamps (T+0ms, T+1000ms)
- Synchronized collection across sources
- Aggregation for AI consumption

v2.2 Extended Metrics:
- pg_stat_statements for query-level performance
- pg_stat_user_tables for table stats (seq scan, dead tuples)
- pg_statio_user_tables for table I/O cache hits
- pg_stat_user_indexes for index usage
- Detailed memory info (dirty pages, swap, hugepages)
- Network stats (TCP retransmits, timeouts)
- Load average
"""

from .collector import TelemetryCollector
from .iostat import IOStatCollector
from .vmstat import VMStatCollector
from .pgstat import PgStatCollector
from .sysstat import SysStatCollector
from .aggregator import TelemetryAggregator

__all__ = [
    "TelemetryCollector",
    "IOStatCollector",
    "VMStatCollector",
    "PgStatCollector",
    "SysStatCollector",
    "TelemetryAggregator",
]
