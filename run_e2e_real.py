#!/usr/bin/env python3
"""
Real End-to-End Test for pg_diagnose

This script runs ACTUAL benchmarks, makes REAL API calls, and applies
REAL configuration changes to validate pg_diagnose works in production.

WARNING: This test:
- Takes 10-20 minutes to complete
- Modifies PostgreSQL configuration
- Requires database restart for some changes
- Should be run against a test/staging database, not production

Usage:
    python run_e2e_real.py \
        --db-host <IP> \
        --db-port 6432 \
        --db-user postgres \
        --db-password postgres \
        --db-name postgres \
        --ssh-host <DB_SERVER_IP> \
        --ssh-user ubuntu \
        --gemini-api-key <KEY> \
        [--benchmark-duration 60] \
        [--max-rounds 3] \
        [--target-improvement 50]

Example:
    python run_e2e_real.py \
        --db-host 44.249.192.13 \
        --db-port 6432 \
        --db-user postgres \
        --db-password postgres \
        --ssh-host 10.0.0.49 \
        --gemini-api-key AIzaSyCbVsXu7cXn_jAIQrZwq1k2MTSlTMC5_Sk \
        --benchmark-duration 60 \
        --max-rounds 3
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import new UI components
try:
    from pg_diagnose.ui import (
        ConsoleUI,
        WorkflowPhase,
    )
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result from a real benchmark run."""
    tps: float
    latency_avg_ms: float
    latency_stddev_ms: float
    transactions: int
    duration_seconds: int
    raw_output: str
    success: bool
    error: Optional[str] = None


@dataclass
class TuningRound:
    """Record of a tuning round."""
    round_num: int
    tps_before: float
    tps_after: float
    improvement_pct: float
    changes_applied: List[Dict[str, Any]]
    ai_rationale: str
    benchmark_duration: int


@dataclass
class E2ETestResult:
    """Complete E2E test result."""
    success: bool
    baseline_tps: float
    final_tps: float
    total_improvement_pct: float
    target_improvement_pct: float
    target_achieved: bool
    rounds_executed: int
    total_duration_seconds: float
    rounds: List[TuningRound] = field(default_factory=list)
    error: Optional[str] = None


class RealE2ETest:
    """
    Real End-to-End test that runs actual benchmarks and applies real changes.

    v2.4: Enhanced with new UI components for better progress visibility.
    """

    def __init__(
        self,
        db_host: str,
        db_port: int,
        db_user: str,
        db_password: str,
        db_name: str,
        ssh_host: str,
        ssh_user: str,
        gemini_api_key: str,
        benchmark_duration: int = 60,
        max_rounds: int = 3,
        target_improvement: float = 50.0,
        pgbench_scale: int = 100,
        pgbench_clients: int = 32,
        pgbench_threads: int = 8,
        ssh_key: Optional[str] = None,
        verbose: bool = True,
    ):
        self.db_host = db_host
        self.db_port = db_port
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.gemini_api_key = gemini_api_key
        self.benchmark_duration = benchmark_duration
        self.max_rounds = max_rounds
        self.target_improvement = target_improvement
        self.pgbench_scale = pgbench_scale
        self.pgbench_clients = pgbench_clients
        self.pgbench_threads = pgbench_threads
        self.verbose = verbose

        # Track state
        self.baseline_tps: float = 0
        self.current_tps: float = 0
        self.rounds: List[TuningRound] = []
        self.initial_config: Dict[str, str] = {}

        # v2.4: Enhanced UI
        self.ui = ConsoleUI(quiet=not verbose) if UI_AVAILABLE else None

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def run(self) -> E2ETestResult:
        """
        Run the complete E2E test.

        Returns:
            E2ETestResult with all details
        """
        start_time = time.time()

        try:
            # v2.4: Print banner
            if self.ui:
                self.ui.print_banner()

            self.log("=" * 60)
            self.log("REAL END-TO-END TEST - pg_diagnose")
            self.log("=" * 60)
            self.log(f"Database: {self.db_host}:{self.db_port}/{self.db_name}")
            self.log(f"SSH: {self.ssh_user}@{self.ssh_host}")
            self.log(f"Benchmark duration: {self.benchmark_duration}s")
            self.log(f"Max rounds: {self.max_rounds}")
            self.log(f"Target improvement: {self.target_improvement}%")
            self.log("")

            # Step 1: Verify connectivity
            if self.ui:
                self.ui.print_phase_header(WorkflowPhase.CONNECT, "Verifying connectivity")
            else:
                self.log("Step 1: Verifying connectivity...")

            self._verify_db_connection()
            self._verify_ssh_connection()
            self.log("Connectivity OK", "SUCCESS")

            # Step 2: Capture initial configuration
            if self.ui:
                self.ui.print_phase_header(WorkflowPhase.DISCOVER, "Capturing configuration")
            else:
                self.log("\nStep 2: Capturing initial configuration...")

            self.initial_config = self._capture_pg_settings()
            self.log(f"Captured {len(self.initial_config)} settings", "SUCCESS")

            # Step 3: Initialize pgbench if needed
            self.log("\nInitializing pgbench...")
            self._init_pgbench()
            self.log("pgbench initialized", "SUCCESS")

            # Step 4: Run baseline benchmark
            if self.ui:
                self.ui.print_phase_header(WorkflowPhase.BASELINE, f"Running {self.benchmark_duration}s benchmark")
                self.ui.start_ai_thinking(f"Running baseline benchmark...", self.benchmark_duration)

            baseline_result = self._run_benchmark()

            if self.ui:
                self.ui.stop_ai_thinking("Baseline complete")

            if not baseline_result.success:
                raise Exception(f"Baseline benchmark failed: {baseline_result.error}")

            self.baseline_tps = baseline_result.tps
            self.current_tps = baseline_result.tps
            target_tps = self.baseline_tps * (1 + self.target_improvement / 100)

            # v2.4: Update UI with TPS info
            if self.ui:
                self.ui.set_tps(self.current_tps, target_tps, self.baseline_tps)
                self.ui.show_status()

            self.log(f"Baseline TPS: {self.baseline_tps:,.0f}", "SUCCESS")
            self.log(f"Target TPS: {target_tps:,.0f} (+{self.target_improvement}%)")

            # Step 5: Tuning loop
            if self.ui:
                self.ui.print_phase_header(WorkflowPhase.TUNING, "AI tuning loop")
            else:
                self.log("\nStep 5: Starting tuning loop...")

            for round_num in range(1, self.max_rounds + 1):
                # v2.4: Enhanced round header
                if self.ui:
                    self.ui.print_round_header(round_num, self.max_rounds)
                else:
                    self.log(f"\n{'='*40}")
                    self.log(f"ROUND {round_num}/{self.max_rounds}")
                    self.log(f"{'='*40}")

                # Get AI recommendations
                if self.ui:
                    self.ui.start_ai_thinking("AI generating recommendations...", 30)

                recommendations = self._get_ai_recommendations(round_num)

                if self.ui:
                    if recommendations:
                        self.ui.stop_ai_thinking(f"{len(recommendations)} recommendations ready")
                    else:
                        self.ui.stop_ai_thinking("No recommendations")

                if not recommendations:
                    self.log("No recommendations from AI, stopping", "WARNING")
                    break

                # v2.4: Show change summary
                requires_restart = any(r.get('requires_restart', False) for r in recommendations)
                if self.ui:
                    self.ui.print_change_summary(recommendations, requires_restart)
                else:
                    self.log(f"Applying {len(recommendations)} changes...")

                applied = self._apply_changes(recommendations)

                if not applied:
                    self.log("No changes applied, stopping", "WARNING")
                    break

                # Run benchmark
                if self.ui:
                    self.ui.start_ai_thinking(f"Running benchmark...", self.benchmark_duration)

                result = self._run_benchmark()

                if self.ui:
                    self.ui.stop_ai_thinking("Benchmark complete")

                if not result.success:
                    self.log(f"Benchmark failed: {result.error}", "ERROR")
                    continue

                # Record round
                tps_before = self.current_tps
                tps_after = result.tps
                improvement = ((tps_after - tps_before) / tps_before) * 100 if tps_before > 0 else 0

                round_record = TuningRound(
                    round_num=round_num,
                    tps_before=tps_before,
                    tps_after=tps_after,
                    improvement_pct=improvement,
                    changes_applied=recommendations,
                    ai_rationale="AI-generated tuning",
                    benchmark_duration=self.benchmark_duration,
                )
                self.rounds.append(round_record)

                self.current_tps = tps_after
                total_improvement = ((self.current_tps - self.baseline_tps) / self.baseline_tps) * 100

                # v2.4: Enhanced TPS comparison display
                if self.ui:
                    self.ui.print_tps_comparison(tps_before, tps_after, target_tps)
                    self.ui.add_completed_round(round_num, improvement)
                    self.ui.set_tps(self.current_tps, target_tps, self.baseline_tps)
                    self.ui.show_status_with_timeline()
                else:
                    self.log(f"Round {round_num} TPS: {tps_after:,.0f} ({improvement:+.1f}% from previous)", "SUCCESS")
                    self.log(f"Total improvement: {total_improvement:+.1f}%")

                # Check if target achieved
                if self.current_tps >= target_tps:
                    self.log(f"\nTARGET ACHIEVED! TPS {self.current_tps:,.0f} >= {target_tps:,.0f}", "SUCCESS")
                    if self.ui:
                        self.ui.print_command_hint("Use /export to save results")
                    break

            # Calculate final results
            elapsed = time.time() - start_time
            total_improvement = ((self.current_tps - self.baseline_tps) / self.baseline_tps) * 100
            target_achieved = self.current_tps >= target_tps

            self.log("\n" + "=" * 60)
            self.log("TEST COMPLETE")
            self.log("=" * 60)
            self.log(f"Baseline TPS: {self.baseline_tps:,.0f}")
            self.log(f"Final TPS: {self.current_tps:,.0f}")
            self.log(f"Total improvement: {total_improvement:+.1f}%")
            self.log(f"Target ({self.target_improvement}%): {'ACHIEVED' if target_achieved else 'NOT ACHIEVED'}")
            self.log(f"Rounds executed: {len(self.rounds)}")
            self.log(f"Total duration: {elapsed:.0f}s ({elapsed/60:.1f} min)")

            return E2ETestResult(
                success=True,
                baseline_tps=self.baseline_tps,
                final_tps=self.current_tps,
                total_improvement_pct=total_improvement,
                target_improvement_pct=self.target_improvement,
                target_achieved=target_achieved,
                rounds_executed=len(self.rounds),
                total_duration_seconds=elapsed,
                rounds=self.rounds,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            self.log(f"\nTEST FAILED: {e}", "ERROR")
            import traceback
            traceback.print_exc()

            return E2ETestResult(
                success=False,
                baseline_tps=self.baseline_tps,
                final_tps=self.current_tps,
                total_improvement_pct=0,
                target_improvement_pct=self.target_improvement,
                target_achieved=False,
                rounds_executed=len(self.rounds),
                total_duration_seconds=elapsed,
                rounds=self.rounds,
                error=str(e),
            )

    def _verify_db_connection(self):
        """Verify database connection works."""
        import psycopg2
        conn = psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_password,
            dbname=self.db_name,
            connect_timeout=10,
        )
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        self.log(f"  Database: {version[:50]}...")
        conn.close()

    def _verify_ssh_connection(self):
        """Verify SSH connection to database server."""
        result = self._ssh_exec("echo 'SSH OK'")
        if "SSH OK" not in result:
            raise Exception(f"SSH connection failed: {result}")
        self.log(f"  SSH: Connected to {self.ssh_host}")

    def _ssh_exec(self, cmd: str, timeout: int = 30) -> str:
        """Execute command on database server via SSH."""
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]

        if self.ssh_key:
            ssh_cmd.extend(["-i", self.ssh_key])

        ssh_cmd.append(f"{self.ssh_user}@{self.ssh_host}")
        ssh_cmd.append(cmd)

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0 and result.stderr:
            return f"ERROR: {result.stderr}"

        return result.stdout

    def _capture_pg_settings(self) -> Dict[str, str]:
        """Capture current PostgreSQL settings."""
        import psycopg2
        conn = psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_password,
            dbname=self.db_name,
        )

        settings = {}
        cur = conn.cursor()
        cur.execute("""
            SELECT name, setting, unit, context
            FROM pg_settings
            WHERE context IN ('postmaster', 'sighup', 'superuser', 'user')
            ORDER BY name
        """)

        for name, setting, unit, context in cur.fetchall():
            settings[name] = setting

        conn.close()
        return settings

    def _init_pgbench(self):
        """Initialize pgbench tables if needed."""
        # Check if pgbench tables exist
        import psycopg2
        conn = psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_password,
            dbname=self.db_name,
        )

        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'pgbench_accounts'
            )
        """)
        exists = cur.fetchone()[0]
        conn.close()

        if not exists:
            self.log(f"  Initializing pgbench with scale {self.pgbench_scale}...")
            # Run pgbench init on the client (where pgbench is installed)
            cmd = (
                f"PGPASSWORD={self.db_password} pgbench "
                f"-h {self.db_host} -p {self.db_port} -U {self.db_user} "
                f"-i -s {self.pgbench_scale} {self.db_name}"
            )
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise Exception(f"pgbench init failed: {result.stderr}")
        else:
            self.log("  pgbench tables already exist")

    def _run_benchmark(self) -> BenchmarkResult:
        """Run pgbench benchmark."""
        cmd = (
            f"PGPASSWORD={self.db_password} pgbench "
            f"-h {self.db_host} -p {self.db_port} -U {self.db_user} "
            f"-c {self.pgbench_clients} -j {self.pgbench_threads} "
            f"-T {self.benchmark_duration} -P 10 "
            f"{self.db_name}"
        )

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.benchmark_duration + 60,
            )

            output = result.stdout + result.stderr

            # Parse TPS from output
            tps = self._parse_tps(output)
            latency = self._parse_latency(output)
            transactions = self._parse_transactions(output)

            if tps <= 0:
                return BenchmarkResult(
                    tps=0,
                    latency_avg_ms=0,
                    latency_stddev_ms=0,
                    transactions=0,
                    duration_seconds=self.benchmark_duration,
                    raw_output=output,
                    success=False,
                    error="Failed to parse TPS from output",
                )

            return BenchmarkResult(
                tps=tps,
                latency_avg_ms=latency[0],
                latency_stddev_ms=latency[1],
                transactions=transactions,
                duration_seconds=self.benchmark_duration,
                raw_output=output,
                success=True,
            )

        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                tps=0,
                latency_avg_ms=0,
                latency_stddev_ms=0,
                transactions=0,
                duration_seconds=self.benchmark_duration,
                raw_output="",
                success=False,
                error="Benchmark timed out",
            )
        except Exception as e:
            return BenchmarkResult(
                tps=0,
                latency_avg_ms=0,
                latency_stddev_ms=0,
                transactions=0,
                duration_seconds=self.benchmark_duration,
                raw_output="",
                success=False,
                error=str(e),
            )

    def _parse_tps(self, output: str) -> float:
        """Parse TPS from pgbench output."""
        import re
        # Look for "tps = 1234.567890 (including connections establishing)"
        match = re.search(r'tps = ([\d.]+)', output)
        if match:
            return float(match.group(1))
        return 0

    def _parse_latency(self, output: str) -> Tuple[float, float]:
        """Parse latency from pgbench output."""
        import re
        avg = 0.0
        stddev = 0.0

        # "latency average = 6.123 ms"
        match = re.search(r'latency average = ([\d.]+)', output)
        if match:
            avg = float(match.group(1))

        # "latency stddev = 3.456 ms"
        match = re.search(r'latency stddev = ([\d.]+)', output)
        if match:
            stddev = float(match.group(1))

        return avg, stddev

    def _parse_transactions(self, output: str) -> int:
        """Parse transaction count from pgbench output."""
        import re
        match = re.search(r'number of transactions actually processed: (\d+)', output)
        if match:
            return int(match.group(1))
        return 0

    def _get_ai_recommendations(self, round_num: int) -> List[Dict[str, Any]]:
        """Get tuning recommendations from Gemini AI."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-3-flash-preview')

            # Get current config for context
            current_config = self._capture_pg_settings()

            # Build history of applied changes
            applied_changes_text = ""
            if self.rounds:
                applied_changes_text = "\n\nPREVIOUSLY APPLIED CHANGES (DO NOT REPEAT THESE):\n"
                for prev_round in self.rounds:
                    applied_changes_text += f"\nRound {prev_round.round_num}:"
                    applied_changes_text += f" TPS went from {prev_round.tps_before:,.0f} to {prev_round.tps_after:,.0f} ({prev_round.improvement_pct:+.1f}%)"
                    for change in prev_round.changes_applied:
                        applied_changes_text += f"\n  - {change.get('param')} = {change.get('value')}"

            # Build prompt
            prompt = f"""You are a PostgreSQL performance tuning expert. Your goal is to improve TPS (transactions per second).

CURRENT SITUATION:
- Baseline TPS: {self.baseline_tps:,.0f}
- Current TPS: {self.current_tps:,.0f}
- Target TPS: {self.baseline_tps * (1 + self.target_improvement/100):,.0f} (+{self.target_improvement}% improvement needed)
- Tuning round: {round_num} of {self.max_rounds}
{applied_changes_text}

CURRENT POSTGRESQL SETTINGS:
- shared_buffers: {current_config.get('shared_buffers', 'unknown')}
- effective_cache_size: {current_config.get('effective_cache_size', 'unknown')}
- work_mem: {current_config.get('work_mem', 'unknown')}
- maintenance_work_mem: {current_config.get('maintenance_work_mem', 'unknown')}
- wal_buffers: {current_config.get('wal_buffers', 'unknown')}
- checkpoint_completion_target: {current_config.get('checkpoint_completion_target', 'unknown')}
- random_page_cost: {current_config.get('random_page_cost', 'unknown')}
- effective_io_concurrency: {current_config.get('effective_io_concurrency', 'unknown')}
- max_parallel_workers_per_gather: {current_config.get('max_parallel_workers_per_gather', 'unknown')}
- max_wal_size: {current_config.get('max_wal_size', 'unknown')}
- min_wal_size: {current_config.get('min_wal_size', 'unknown')}
- synchronous_commit: {current_config.get('synchronous_commit', 'unknown')}
- wal_compression: {current_config.get('wal_compression', 'unknown')}
- huge_pages: {current_config.get('huge_pages', 'unknown')}

INSTRUCTIONS:
1. Analyze the current settings and previous changes (if any)
2. Suggest 2-4 NEW tuning changes that have NOT been applied before
3. Focus on different aspects of PostgreSQL tuning each round:
   - Round 1: Memory settings (shared_buffers, effective_cache_size, work_mem)
   - Round 2: WAL and checkpoint settings (max_wal_size, wal_buffers, checkpoint settings)
   - Round 3: I/O and parallelism (effective_io_concurrency, random_page_cost, parallel workers)
4. Do NOT suggest the same parameter values that were already applied in previous rounds

Return ONLY a valid JSON array with this exact format (no markdown, no explanation, no code blocks):
[
  {{"param": "parameter_name", "value": "new_value", "requires_restart": true_or_false}},
  {{"param": "another_param", "value": "another_value", "requires_restart": true_or_false}}
]
"""

            response = model.generate_content(prompt)
            text = response.text.strip()

            # Clean up response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            recommendations = json.loads(text)
            return recommendations

        except Exception as e:
            self.log(f"AI recommendation failed: {e}", "ERROR")
            return []

    def _apply_changes(self, recommendations: List[Dict[str, Any]]) -> bool:
        """Apply tuning changes to PostgreSQL."""
        requires_restart = False
        applied = []

        for rec in recommendations:
            param = rec.get('param')
            value = rec.get('value')
            needs_restart = rec.get('requires_restart', False)

            if not param or not value:
                continue

            # Apply via ALTER SYSTEM
            self.log(f"  Setting {param} = {value}")

            cmd = f"sudo -u postgres psql -c \"ALTER SYSTEM SET {param} = '{value}';\""
            result = self._ssh_exec(cmd)

            if "ERROR" in result:
                self.log(f"  Failed to set {param}: {result}", "WARNING")
                continue

            applied.append(rec)
            if needs_restart:
                requires_restart = True

        if not applied:
            return False

        # Reload or restart
        if requires_restart:
            self.log("  Restarting PostgreSQL (changes require restart)...")
            self._ssh_exec("sudo systemctl restart postgresql")
            time.sleep(5)  # Wait for restart
        else:
            self.log("  Reloading PostgreSQL configuration...")
            self._ssh_exec("sudo -u postgres psql -c 'SELECT pg_reload_conf();'")
            time.sleep(2)

        return True

    def restore_initial_config(self):
        """Restore initial configuration (for cleanup)."""
        self.log("\nRestoring initial configuration...")

        # Reset all ALTER SYSTEM settings
        cmd = "sudo -u postgres psql -c 'ALTER SYSTEM RESET ALL;'"
        self._ssh_exec(cmd)

        # Restart to apply
        self._ssh_exec("sudo systemctl restart postgresql")
        time.sleep(5)

        self.log("Configuration restored", "SUCCESS")


def main():
    parser = argparse.ArgumentParser(
        description="Run real E2E test for pg_diagnose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using config file (recommended):
    python run_e2e_real.py --config config.toml

    # Create example config:
    python run_e2e_real.py --init-config

    # Override specific values:
    python run_e2e_real.py --config config.toml --max-rounds 5

    # Legacy mode (all flags):
    python run_e2e_real.py --db-host 10.0.0.1 --ssh-host 10.0.0.1 ...
        """
    )

    # Config file
    parser.add_argument("-c", "--config", help="Path to TOML config file")
    parser.add_argument("--init-config", action="store_true", help="Create example config.toml")
    parser.add_argument("--show-config", action="store_true", help="Show loaded config and exit")

    # Database connection (optional overrides)
    parser.add_argument("--db-host", help="Database host (or pgcat host)")
    parser.add_argument("--db-port", type=int, help="Database port")
    parser.add_argument("--db-user", help="Database user")
    parser.add_argument("--db-password", help="Database password")
    parser.add_argument("--db-name", help="Database name")

    # SSH connection (optional overrides)
    parser.add_argument("--ssh-host", help="SSH host (database server IP)")
    parser.add_argument("--ssh-user", help="SSH user")
    parser.add_argument("--ssh-key", help="SSH private key path")

    # AI (optional override)
    parser.add_argument("--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY)")

    # Benchmark settings (optional overrides)
    parser.add_argument("--benchmark-duration", type=int, help="Benchmark duration in seconds")
    parser.add_argument("--max-rounds", type=int, help="Maximum tuning rounds")
    parser.add_argument("--target-improvement", type=float, help="Target improvement percentage")
    parser.add_argument("--pgbench-scale", type=int, help="pgbench scale factor")
    parser.add_argument("--pgbench-clients", type=int, help="pgbench clients")
    parser.add_argument("--pgbench-threads", type=int, help="pgbench threads")

    # Options
    parser.add_argument("--restore-after", action="store_true", help="Restore initial config after test")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Handle --init-config
    if args.init_config:
        try:
            from pg_diagnose.config import create_example_config
            path = create_example_config()
            print(f"Created example config: {path}")
            print("Edit the file and run: python run_e2e_real.py --config config.toml")
            sys.exit(0)
        except FileExistsError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ImportError:
            # Fallback if config module not found
            print("Creating config.toml...")
            Path("config.toml").write_text("""# pg_diagnose Configuration

[database]
host = "localhost"
port = 5432
user = "postgres"
password = "postgres"
name = "postgres"

[ssh]
host = ""  # Set to database server IP for remote tuning
user = "ubuntu"

[benchmark]
duration = 60
scale = 100
clients = 32
threads = 8

[tuning]
max_rounds = 3
target_improvement = 20.0

[ai]
provider = "gemini"
model = "gemini-3-flash-preview"
# api_key = ""  # Or set GEMINI_API_KEY env var
""")
            print("Created config.toml - edit and run: python run_e2e_real.py -c config.toml")
            sys.exit(0)

    # Load config
    try:
        from pg_diagnose.config import Config
        config = Config.load(args.config)
        config.override_from_args(args)
    except ImportError:
        # Fallback to args-only mode
        config = None
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Handle --show-config
    if args.show_config:
        if config:
            print(config.summary())
        else:
            print("No config loaded (using command-line args)")
        sys.exit(0)

    # Validate we have required values
    if config:
        # Use config values
        db_host = config.database.host
        db_port = config.database.port
        db_user = config.database.user
        db_password = config.database.password
        db_name = config.database.name
        ssh_host = config.ssh.host
        ssh_user = config.ssh.user
        ssh_key = config.ssh.key
        gemini_api_key = config.ai.api_key
        benchmark_duration = config.benchmark.duration
        max_rounds = config.tuning.max_rounds
        target_improvement = config.tuning.target_improvement
        pgbench_scale = config.benchmark.scale
        pgbench_clients = config.benchmark.clients
        pgbench_threads = config.benchmark.threads
        verbose = not config.output.quiet
    else:
        # Fallback to args (legacy mode)
        db_host = args.db_host
        db_port = args.db_port or 5432
        db_user = args.db_user or "postgres"
        db_password = args.db_password or ""
        db_name = args.db_name or "postgres"
        ssh_host = args.ssh_host
        ssh_user = args.ssh_user or "ubuntu"
        ssh_key = args.ssh_key
        gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        benchmark_duration = args.benchmark_duration or 60
        max_rounds = args.max_rounds or 3
        target_improvement = args.target_improvement or 20.0
        pgbench_scale = args.pgbench_scale or 100
        pgbench_clients = args.pgbench_clients or 32
        pgbench_threads = args.pgbench_threads or 8
        verbose = not args.quiet

    # Validate required fields
    errors = []
    if not db_host:
        errors.append("Database host required (--db-host or config)")
    if not ssh_host:
        errors.append("SSH host required (--ssh-host or config)")
    if not gemini_api_key:
        errors.append("Gemini API key required (--gemini-api-key or GEMINI_API_KEY)")

    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        print("\nRun with --init-config to create a config file")
        sys.exit(1)

    # Show config summary
    if verbose and config:
        print(config.summary())
        print()

    # Run test
    test = RealE2ETest(
        db_host=db_host,
        db_port=db_port,
        db_user=db_user,
        db_password=db_password,
        db_name=db_name,
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        gemini_api_key=gemini_api_key,
        benchmark_duration=benchmark_duration,
        max_rounds=max_rounds,
        target_improvement=target_improvement,
        pgbench_scale=pgbench_scale,
        pgbench_clients=pgbench_clients,
        pgbench_threads=pgbench_threads,
        verbose=verbose,
    )

    result = test.run()

    # Restore if requested
    if args.restore_after:
        test.restore_initial_config()

    # Exit code
    if result.success and result.target_achieved:
        sys.exit(0)
    elif result.success:
        sys.exit(1)  # Test ran but target not achieved
    else:
        sys.exit(2)  # Test failed


if __name__ == "__main__":
    main()
