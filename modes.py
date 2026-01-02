"""
Modes - Alternative operation modes for pg_diagnose.

Provides:
- Health check mode (--health)
- Watch mode (--watch)
- Dry-run mode (--dry-run)
- Auto mode (--auto)
- Analyze-only mode (--analyze-only)
"""

import time
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class HealthCheck:
    """Quick health check for PostgreSQL."""

    def __init__(self, conn, ssh_config: Optional[Dict] = None):
        self.conn = conn
        self.ssh_config = ssh_config
        self.console = Console() if RICH_AVAILABLE else None
        self.checks = []

    def run_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': [],
            'recommendations': [],
        }

        # Run each check
        checks = [
            self._check_connection_pool,
            self._check_replication_lag,
            self._check_cache_hit_ratio,
            self._check_long_queries,
            self._check_dead_tuples,
            self._check_disk_space,
            self._check_locks,
            self._check_connections,
        ]

        for check in checks:
            try:
                result = check()
                results['checks'].append(result)
                if result.get('recommendation'):
                    results['recommendations'].append(result['recommendation'])
            except Exception as e:
                results['checks'].append({
                    'name': check.__name__.replace('_check_', ''),
                    'status': 'error',
                    'message': str(e),
                })

        return results

    def _check_connection_pool(self) -> Dict:
        """Check connection pool status."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_stat_activity")
            active = cur.fetchone()[0]

            cur.execute("SHOW max_connections")
            max_conn = int(cur.fetchone()[0])

        usage_pct = (active / max_conn) * 100
        status = 'ok' if usage_pct < 80 else 'warning' if usage_pct < 95 else 'critical'

        return {
            'name': 'Connection Pool',
            'status': status,
            'message': f"{active}/{max_conn} used ({usage_pct:.0f}%)",
            'value': usage_pct,
            'recommendation': f"Consider increasing max_connections" if usage_pct > 80 else None,
        }

    def _check_replication_lag(self) -> Dict:
        """Check replication lag."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    CASE
                        WHEN pg_is_in_recovery() THEN
                            COALESCE(pg_last_wal_receive_lsn() - pg_last_wal_replay_lsn(), 0)
                        ELSE 0
                    END as lag_bytes
            """)
            lag = cur.fetchone()[0] or 0

        status = 'ok' if lag < 1024*1024 else 'warning' if lag < 100*1024*1024 else 'critical'

        if lag < 1024:
            msg = f"{lag} bytes"
        elif lag < 1024*1024:
            msg = f"{lag/1024:.1f} KB"
        else:
            msg = f"{lag/(1024*1024):.1f} MB"

        return {
            'name': 'Replication Lag',
            'status': status,
            'message': msg,
            'value': lag,
            'recommendation': "Check replica connectivity" if lag > 1024*1024 else None,
        }

    def _check_cache_hit_ratio(self) -> Dict:
        """Check buffer cache hit ratio."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    CASE
                        WHEN blks_hit + blks_read = 0 THEN 100
                        ELSE (blks_hit::float / (blks_hit + blks_read) * 100)
                    END as ratio
                FROM pg_stat_database
                WHERE datname = current_database()
            """)
            ratio = cur.fetchone()[0] or 0

        status = 'ok' if ratio >= 99 else 'warning' if ratio >= 95 else 'critical'

        return {
            'name': 'Cache Hit Ratio',
            'status': status,
            'message': f"{ratio:.1f}%",
            'value': ratio,
            'recommendation': "Increase shared_buffers" if ratio < 95 else None,
        }

    def _check_long_queries(self) -> Dict:
        """Check for long-running queries."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT count(*)
                FROM pg_stat_activity
                WHERE state = 'active'
                  AND query_start < now() - interval '30 seconds'
                  AND pid != pg_backend_pid()
            """)
            count = cur.fetchone()[0]

        status = 'ok' if count == 0 else 'warning' if count < 3 else 'critical'

        return {
            'name': 'Long-running Queries',
            'status': status,
            'message': f"{count} queries > 30s",
            'value': count,
            'recommendation': f"Review {count} long-running queries (use /queries)" if count > 0 else None,
        }

    def _check_dead_tuples(self) -> Dict:
        """Check for tables with high dead tuple ratio."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    schemaname || '.' || relname as table_name,
                    n_dead_tup,
                    n_live_tup,
                    CASE WHEN n_live_tup > 0 THEN n_dead_tup::float / n_live_tup * 100 ELSE 0 END as ratio
                FROM pg_stat_user_tables
                WHERE n_dead_tup > 10000
                ORDER BY n_dead_tup DESC
                LIMIT 5
            """)
            rows = cur.fetchall()

        if not rows:
            return {
                'name': 'Dead Tuples',
                'status': 'ok',
                'message': 'Autovacuum OK',
                'value': 0,
            }

        max_ratio = max(r[3] for r in rows)
        status = 'ok' if max_ratio < 10 else 'warning' if max_ratio < 30 else 'critical'

        return {
            'name': 'Dead Tuples',
            'status': status,
            'message': f"{max_ratio:.1f}% max dead ratio",
            'value': max_ratio,
            'recommendation': f"Run VACUUM on high-dead-tuple tables" if max_ratio > 10 else None,
        }

    def _check_disk_space(self) -> Dict:
        """Check disk space usage."""
        if not self.ssh_config:
            return {
                'name': 'Disk Space',
                'status': 'unknown',
                'message': 'SSH not configured',
                'value': 0,
            }

        try:
            cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                   f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                   "df -h / | tail -1 | awk '{print $5}'"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                usage = int(result.stdout.strip().rstrip('%'))
                status = 'ok' if usage < 70 else 'warning' if usage < 85 else 'critical'

                return {
                    'name': 'Disk Space',
                    'status': status,
                    'message': f"{usage}% used",
                    'value': usage,
                    'recommendation': "Free up disk space" if usage > 85 else None,
                }
        except Exception:
            pass

        return {
            'name': 'Disk Space',
            'status': 'unknown',
            'message': 'Could not check',
            'value': 0,
        }

    def _check_locks(self) -> Dict:
        """Check for waiting locks."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_locks WHERE NOT granted")
            waiting = cur.fetchone()[0]

        status = 'ok' if waiting == 0 else 'warning' if waiting < 5 else 'critical'

        return {
            'name': 'Waiting Locks',
            'status': status,
            'message': f"{waiting} waiting",
            'value': waiting,
            'recommendation': "Investigate lock contention" if waiting > 0 else None,
        }

    def _check_connections(self) -> Dict:
        """Check connection states."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT state, count(*)
                FROM pg_stat_activity
                GROUP BY state
            """)
            states = {r[0] or 'null': r[1] for r in cur.fetchall()}

        idle = states.get('idle', 0)
        active = states.get('active', 0)
        idle_txn = states.get('idle in transaction', 0)

        status = 'ok' if idle_txn == 0 else 'warning' if idle_txn < 5 else 'critical'

        return {
            'name': 'Connection States',
            'status': status,
            'message': f"active={active}, idle={idle}, idle_txn={idle_txn}",
            'value': idle_txn,
            'recommendation': f"Fix {idle_txn} idle-in-transaction connections" if idle_txn > 0 else None,
        }

    def display(self, results: Dict):
        """Display health check results."""
        if self.console and RICH_AVAILABLE:
            self._display_rich(results)
        else:
            self._display_plain(results)

    def _display_rich(self, results: Dict):
        """Rich display of health check."""
        # Header
        db_info = f"{self.conn.info.dbname}@{self.conn.info.host}"
        self.console.print(Panel(
            f"[bold]PostgreSQL Health Check[/] @ [cyan]{db_info}[/]",
            border_style="blue"
        ))

        # Checks table
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Status", width=3)
        table.add_column("Check", style="cyan")
        table.add_column("Result", style="green")

        for check in results['checks']:
            status = check['status']
            icon = {
                'ok': '[green]✓[/]',
                'warning': '[yellow]⚠[/]',
                'critical': '[red]✗[/]',
                'unknown': '[dim]?[/]',
                'error': '[red]![/]',
            }.get(status, '[dim]?[/]')

            table.add_row(icon, check['name'], check['message'])

        self.console.print(table)

        # Recommendations
        if results['recommendations']:
            self.console.print()
            self.console.print("[yellow]Recommendations:[/]")
            for i, rec in enumerate(results['recommendations'], 1):
                self.console.print(f"  {i}. {rec}")

    def _display_plain(self, results: Dict):
        """Plain text display of health check."""
        print("\nPostgreSQL Health Check")
        print("=" * 50)

        for check in results['checks']:
            status = check['status']
            icon = {'ok': '✓', 'warning': '⚠', 'critical': '✗'}.get(status, '?')
            print(f"  {icon} {check['name']:25} {check['message']}")

        if results['recommendations']:
            print("\nRecommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}")


class WatchMode:
    """Live monitoring mode."""

    def __init__(self, conn, ssh_config: Optional[Dict] = None, interval: int = 5):
        self.conn = conn
        self.ssh_config = ssh_config
        self.interval = interval
        self.console = Console() if RICH_AVAILABLE else None
        self.events: List[Dict] = []

    def run(self, target_tps: float = 0):
        """Run watch mode until interrupted."""
        if self.console and RICH_AVAILABLE:
            self._run_rich(target_tps)
        else:
            self._run_plain(target_tps)

    def _get_metrics(self) -> Dict:
        """Get current metrics."""
        metrics = {}

        with self.conn.cursor() as cur:
            # Cache hit ratio
            cur.execute("""
                SELECT
                    CASE WHEN blks_hit + blks_read = 0 THEN 100
                    ELSE (blks_hit::float / (blks_hit + blks_read) * 100) END
                FROM pg_stat_database WHERE datname = current_database()
            """)
            metrics['cache_hit'] = cur.fetchone()[0] or 0

            # Active connections
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            metrics['active_conns'] = cur.fetchone()[0]

            # Waiting locks
            cur.execute("SELECT count(*) FROM pg_locks WHERE NOT granted")
            metrics['waiting_locks'] = cur.fetchone()[0]

            # Transactions
            cur.execute("""
                SELECT xact_commit + xact_rollback
                FROM pg_stat_database WHERE datname = current_database()
            """)
            metrics['total_txn'] = cur.fetchone()[0] or 0

            # Checkpoints
            cur.execute("SELECT checkpoints_timed, checkpoints_req FROM pg_stat_bgwriter")
            row = cur.fetchone()
            metrics['checkpoints_timed'] = row[0] if row else 0
            metrics['checkpoints_req'] = row[1] if row else 0

        # Get system metrics via SSH
        if self.ssh_config:
            try:
                cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                       f"{self.ssh_config['user']}@{self.ssh_config['host']}",
                       "cat /proc/loadavg && cat /proc/meminfo | grep -E '^(MemTotal|MemAvailable):'"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        load = lines[0].split()
                        metrics['load_1'] = float(load[0])

                    for line in lines[1:]:
                        if 'MemTotal' in line:
                            metrics['mem_total'] = int(line.split()[1])
                        elif 'MemAvailable' in line:
                            metrics['mem_available'] = int(line.split()[1])

                    if 'mem_total' in metrics and 'mem_available' in metrics:
                        metrics['mem_used_pct'] = (1 - metrics['mem_available'] / metrics['mem_total']) * 100
            except Exception:
                pass

        return metrics

    def _run_rich(self, target_tps: float):
        """Run with rich live display."""
        self.console.print(f"[dim]Live metrics (every {self.interval}s). Press Ctrl+C to stop.[/]")
        self.console.print()

        prev_txn = 0
        prev_time = time.time()

        try:
            while True:
                metrics = self._get_metrics()
                now = time.time()

                # Calculate TPS
                if prev_txn > 0:
                    elapsed = now - prev_time
                    tps = (metrics['total_txn'] - prev_txn) / elapsed if elapsed > 0 else 0
                else:
                    tps = 0

                prev_txn = metrics['total_txn']
                prev_time = now

                # Build display
                timestamp = datetime.now().strftime("%H:%M:%S")

                # TPS bar
                if target_tps > 0:
                    tps_pct = min(tps / target_tps * 100, 100)
                    tps_bar = self._make_bar(tps_pct, 30)
                    tps_str = f"TPS      {tps_bar}  {tps:.0f} / {target_tps:.0f}"
                else:
                    tps_str = f"TPS: {tps:.0f}"

                # Cache bar
                cache_bar = self._make_bar(metrics['cache_hit'], 30)
                cache_str = f"Cache    {cache_bar}  {metrics['cache_hit']:.1f}%"

                # Memory bar
                mem_pct = metrics.get('mem_used_pct', 0)
                mem_bar = self._make_bar(mem_pct, 30)
                mem_str = f"Memory   {mem_bar}  {mem_pct:.0f}%"

                # Print
                self.console.print(
                    f"[dim]{timestamp}[/] | "
                    f"TPS: [cyan]{tps:.0f}[/] | "
                    f"Cache: [{'green' if metrics['cache_hit'] > 99 else 'yellow'}]{metrics['cache_hit']:.1f}%[/] | "
                    f"Active: [cyan]{metrics['active_conns']}[/] | "
                    f"Locks: [{'green' if metrics['waiting_locks'] == 0 else 'red'}]{metrics['waiting_locks']}[/]"
                )

                time.sleep(self.interval)

        except KeyboardInterrupt:
            self.console.print("\n[dim]Watch mode stopped[/]")

    def _run_plain(self, target_tps: float):
        """Run with plain text display."""
        print(f"Live metrics (every {self.interval}s). Ctrl+C to stop.")

        prev_txn = 0
        prev_time = time.time()

        try:
            while True:
                metrics = self._get_metrics()
                now = time.time()

                if prev_txn > 0:
                    elapsed = now - prev_time
                    tps = (metrics['total_txn'] - prev_txn) / elapsed if elapsed > 0 else 0
                else:
                    tps = 0

                prev_txn = metrics['total_txn']
                prev_time = now

                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"{timestamp} | TPS: {tps:.0f} | Cache: {metrics['cache_hit']:.1f}% | Active: {metrics['active_conns']} | Locks: {metrics['waiting_locks']}")

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\nWatch mode stopped")

    def _make_bar(self, pct: float, width: int) -> str:
        """Create a progress bar."""
        filled = int(pct / 100 * width)
        empty = width - filled
        return f"[green]{'█' * filled}[/][dim]{'░' * empty}[/]"


class DryRunMode:
    """Dry-run mode - show recommendations without applying."""

    def __init__(self, conn, agent, ssh_config: Optional[Dict] = None):
        self.conn = conn
        self.agent = agent
        self.ssh_config = ssh_config
        self.console = Console() if RICH_AVAILABLE else None

    def run(self, output_file: Optional[str] = None) -> Dict:
        """
        Analyze and generate recommendations without applying.

        Args:
            output_file: Optional file to write recommendations

        Returns:
            Recommendations dict
        """
        from .discovery.system import SystemScanner, SystemScannerConfig
        from .discovery.schema import SchemaScanner
        from .discovery.runtime import RuntimeScanner
        from .protocol.context import ContextPacket

        # Discovery
        sys_config = None
        if self.ssh_config:
            sys_config = SystemScannerConfig(
                ssh_host=self.ssh_config.get('host'),
                ssh_port=self.ssh_config.get('port', 22),
                ssh_user=self.ssh_config.get('user'),
            )

        system_scanner = SystemScanner(config=sys_config)
        system_context = system_scanner.scan()

        schema_scanner = SchemaScanner(self.conn)
        schema_context = schema_scanner.scan()

        runtime_scanner = RuntimeScanner(self.conn, system_scanner)
        runtime_context = runtime_scanner.scan()

        context_packet = ContextPacket(
            protocol_version="v2",
            timestamp=datetime.utcnow().isoformat(),
            session_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            system_context=system_context,
            runtime_context=runtime_context,
            schema_context=schema_context,
        )

        # Get AI analysis
        first_sight = self.agent.first_sight_analysis(context_packet)

        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'system_overview': first_sight.system_overview,
            'key_observations': first_sight.key_observations,
            'warnings': first_sight.warnings,
            'strategies': [],
        }

        for opt in first_sight.strategy_options:
            recommendations['strategies'].append({
                'name': opt.name,
                'goal': opt.goal,
                'hypothesis': opt.hypothesis,
                'target_kpis': opt.target_kpis,
                'risk_level': opt.risk_level,
            })

        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(recommendations, f, indent=2)

        return recommendations

    def display(self, recommendations: Dict):
        """Display recommendations."""
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel(
                "[bold]DRY RUN MODE[/] - No changes will be applied",
                border_style="yellow"
            ))

            self.console.print("\n[cyan]System Overview:[/]")
            self.console.print(recommendations['system_overview'])

            if recommendations['key_observations']:
                self.console.print("\n[cyan]Key Observations:[/]")
                for obs in recommendations['key_observations']:
                    self.console.print(f"  • {obs}")

            if recommendations['warnings']:
                self.console.print("\n[yellow]Warnings:[/]")
                for w in recommendations['warnings']:
                    self.console.print(f"  ⚠ {w}")

            if recommendations['strategies']:
                self.console.print("\n[cyan]Recommended Strategies:[/]")
                for i, s in enumerate(recommendations['strategies'], 1):
                    risk_color = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'red'}.get(s['risk_level'], 'white')
                    self.console.print(f"\n  {i}. [bold]{s['name']}[/] [[{risk_color}]{s['risk_level']}[/]]")
                    self.console.print(f"     Goal: {s['goal']}")
                    self.console.print(f"     Hypothesis: {s['hypothesis']}")
        else:
            print("\nDRY RUN MODE - No changes will be applied")
            print("\nSystem Overview:")
            print(recommendations['system_overview'])

            if recommendations['strategies']:
                print("\nRecommended Strategies:")
                for i, s in enumerate(recommendations['strategies'], 1):
                    print(f"  {i}. {s['name']} [{s['risk_level']}]")


class AutoMode:
    """Automatic tuning mode."""

    def __init__(self, conn, agent, ssh_config: Optional[Dict] = None,
                 target_tps: float = 0, max_rounds: int = 5, risk_level: str = 'low'):
        self.conn = conn
        self.agent = agent
        self.ssh_config = ssh_config
        self.target_tps = target_tps
        self.max_rounds = max_rounds
        self.risk_level = risk_level.upper()
        self.console = Console() if RICH_AVAILABLE else None

    def run(self) -> Dict:
        """
        Run automatic tuning.

        Returns:
            Results dict
        """
        results = {
            'rounds_completed': 0,
            'changes_applied': [],
            'tps_history': [],
            'final_tps': 0,
            'target_hit': False,
        }

        if self.console:
            self.console.print(Panel(
                f"[bold]AUTO MODE[/]\n"
                f"Target: {self.target_tps:.0f} TPS\n"
                f"Max Rounds: {self.max_rounds}\n"
                f"Risk Level: {self.risk_level}",
                border_style="blue"
            ))

        # Implementation would go here...
        # This is a placeholder - full implementation requires
        # integrating with the main tuning loop

        return results
