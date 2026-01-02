"""
Golden Test Data - Real PostgreSQL tuning knowledge for mock testing.

This module contains realistic, battle-tested PostgreSQL tuning configurations
that the mock AI agent returns. This is NOT random data - it's based on
actual PostgreSQL performance tuning best practices.

The data is organized by:
1. System profiles (small, medium, large)
2. Workload types (oltp, olap, mixed)
3. Tuning phases (baseline, round1, round2, round3)

Each configuration includes:
- PostgreSQL settings with rationale
- OS settings if applicable
- Custom SQL for benchmarking
- Expected improvements
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class SystemProfile(str, Enum):
    """System size profiles."""
    SMALL = "small"      # 4 cores, 8GB RAM, SSD
    MEDIUM = "medium"    # 8 cores, 32GB RAM, NVMe
    LARGE = "large"      # 16+ cores, 64GB+ RAM, NVMe RAID


class WorkloadType(str, Enum):
    """Workload type profiles."""
    OLTP = "oltp"        # High concurrency, short transactions
    OLAP = "olap"        # Complex queries, large scans
    MIXED = "mixed"      # Combination


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

STRATEGIES = {
    "balanced_tps": {
        "name": "Balanced TPS Optimization",
        "hypothesis": "Optimize memory allocation and WAL settings for balanced read/write performance",
        "target_kpis": {
            "tps_improvement": 50,
            "latency_p99_max": 10,
        },
        "suitable_for": [WorkloadType.OLTP, WorkloadType.MIXED],
        "risk_level": "low",
    },
    "memory_focused": {
        "name": "Memory-Centric Optimization",
        "hypothesis": "Maximize buffer cache hit ratio by aggressive memory tuning",
        "target_kpis": {
            "tps_improvement": 40,
            "cache_hit_ratio": 99,
        },
        "suitable_for": [WorkloadType.OLTP, WorkloadType.OLAP],
        "risk_level": "low",
    },
    "wal_optimized": {
        "name": "WAL Write Optimization",
        "hypothesis": "Reduce WAL bottleneck for write-heavy workloads",
        "target_kpis": {
            "tps_improvement": 60,
            "wal_write_time_reduction": 50,
        },
        "suitable_for": [WorkloadType.OLTP],
        "risk_level": "medium",
    },
    "parallel_query": {
        "name": "Parallel Query Optimization",
        "hypothesis": "Enable aggressive parallelism for analytical queries",
        "target_kpis": {
            "query_time_reduction": 70,
            "cpu_utilization": 80,
        },
        "suitable_for": [WorkloadType.OLAP],
        "risk_level": "low",
    },
}


# =============================================================================
# TUNING CONFIGURATIONS BY ROUND
# =============================================================================

# Configurations are sized for MEDIUM profile (8 cores, 32GB RAM)
# Scaled appropriately in get_tuning_for_profile()

TUNING_ROUNDS = {
    "balanced_tps": {
        0: {  # Initial/Round 0 - Conservative baseline tuning
            "name": "Initial Memory Configuration",
            "changes": [
                {
                    "name": "Shared Buffers Optimization",
                    "category": "memory",
                    "pg_configs": [
                        "ALTER SYSTEM SET shared_buffers = '8GB';",
                        "ALTER SYSTEM SET effective_cache_size = '24GB';",
                    ],
                    "rationale": "Set shared_buffers to 25% of RAM, effective_cache_size to 75%",
                    "requires_restart": True,
                },
                {
                    "name": "Work Memory",
                    "category": "memory",
                    "pg_configs": [
                        "ALTER SYSTEM SET work_mem = '256MB';",
                        "ALTER SYSTEM SET maintenance_work_mem = '2GB';",
                    ],
                    "rationale": "Increase work_mem for complex sorts, maintenance_work_mem for VACUUM/CREATE INDEX",
                    "requires_restart": False,
                },
            ],
            "expected_improvement": 20,
        },
        1: {  # Round 1 - WAL tuning
            "name": "WAL and Checkpoint Optimization",
            "changes": [
                {
                    "name": "WAL Buffers",
                    "category": "wal",
                    "pg_configs": [
                        "ALTER SYSTEM SET wal_buffers = '64MB';",
                        "ALTER SYSTEM SET wal_writer_delay = '10ms';",
                    ],
                    "rationale": "Larger WAL buffers reduce disk I/O, shorter delay improves throughput",
                    "requires_restart": True,
                },
                {
                    "name": "Checkpoint Tuning",
                    "category": "checkpoint",
                    "pg_configs": [
                        "ALTER SYSTEM SET checkpoint_completion_target = '0.9';",
                        "ALTER SYSTEM SET max_wal_size = '4GB';",
                        "ALTER SYSTEM SET min_wal_size = '1GB';",
                    ],
                    "rationale": "Spread checkpoint I/O over longer period, allow more WAL before checkpoint",
                    "requires_restart": False,
                },
            ],
            "expected_improvement": 15,
        },
        2: {  # Round 2 - I/O and planner
            "name": "I/O and Planner Optimization",
            "changes": [
                {
                    "name": "I/O Concurrency",
                    "category": "io",
                    "pg_configs": [
                        "ALTER SYSTEM SET effective_io_concurrency = '200';",
                        "ALTER SYSTEM SET random_page_cost = '1.1';",
                    ],
                    "rationale": "NVMe can handle 200+ concurrent I/O ops, reduce random_page_cost for SSD",
                    "requires_restart": False,
                },
                {
                    "name": "Planner Statistics",
                    "category": "planner",
                    "pg_configs": [
                        "ALTER SYSTEM SET default_statistics_target = '200';",
                    ],
                    "rationale": "More accurate statistics for better query plans",
                    "requires_restart": False,
                },
            ],
            "expected_improvement": 10,
        },
        3: {  # Round 3 - Parallelism
            "name": "Parallel Query Optimization",
            "changes": [
                {
                    "name": "Parallel Workers",
                    "category": "parallelism",
                    "pg_configs": [
                        "ALTER SYSTEM SET max_parallel_workers_per_gather = '4';",
                        "ALTER SYSTEM SET max_parallel_workers = '8';",
                        "ALTER SYSTEM SET max_worker_processes = '12';",
                    ],
                    "rationale": "Enable parallel query execution for large scans",
                    "requires_restart": True,
                },
                {
                    "name": "Parallel Cost Reduction",
                    "category": "parallelism",
                    "pg_configs": [
                        "ALTER SYSTEM SET parallel_tuple_cost = '0.01';",
                        "ALTER SYSTEM SET parallel_setup_cost = '100';",
                    ],
                    "rationale": "Lower thresholds to encourage parallel execution",
                    "requires_restart": False,
                },
            ],
            "expected_improvement": 10,
        },
    },
    "wal_optimized": {
        0: {
            "name": "Aggressive WAL Configuration",
            "changes": [
                {
                    "name": "WAL Level and Compression",
                    "category": "wal",
                    "pg_configs": [
                        "ALTER SYSTEM SET wal_level = 'replica';",
                        "ALTER SYSTEM SET wal_compression = 'on';",
                        "ALTER SYSTEM SET wal_buffers = '128MB';",
                    ],
                    "rationale": "Compress WAL to reduce I/O, larger buffers for burst writes",
                    "requires_restart": True,
                },
            ],
            "expected_improvement": 30,
        },
        1: {
            "name": "Synchronous Commit Tuning",
            "changes": [
                {
                    "name": "Async Commit (if acceptable)",
                    "category": "wal",
                    "pg_configs": [
                        "ALTER SYSTEM SET synchronous_commit = 'off';",
                        "ALTER SYSTEM SET commit_delay = '10';",
                        "ALTER SYSTEM SET commit_siblings = '5';",
                    ],
                    "rationale": "Batch commits for higher throughput (small data loss risk on crash)",
                    "requires_restart": False,
                },
            ],
            "expected_improvement": 25,
        },
    },
}


# =============================================================================
# BENCHMARK CONFIGURATIONS
# =============================================================================

BENCHMARK_CONFIGS = {
    "standard_oltp": {
        "name": "Standard OLTP Benchmark",
        "type": "pgbench",
        "scale": 100,
        "clients": 32,
        "threads": 8,
        "duration_seconds": 60,
        "custom_sql": None,  # Use built-in TPC-B
    },
    "read_heavy": {
        "name": "Read-Heavy Benchmark",
        "type": "pgbench",
        "scale": 100,
        "clients": 64,
        "threads": 8,
        "duration_seconds": 60,
        "custom_sql": """
-- Read-heavy custom script
\\set aid random(1, 100000 * :scale)
\\set bid random(1, 1 * :scale)
\\set tid random(1, 10 * :scale)
SELECT abalance FROM pgbench_accounts WHERE aid = :aid;
SELECT bbalance FROM pgbench_branches WHERE bid = :bid;
SELECT tbalance FROM pgbench_tellers WHERE tid = :tid;
""",
    },
    "write_heavy": {
        "name": "Write-Heavy Benchmark",
        "type": "pgbench",
        "scale": 100,
        "clients": 16,
        "threads": 4,
        "duration_seconds": 60,
        "custom_sql": """
-- Write-heavy custom script
\\set aid random(1, 100000 * :scale)
\\set bid random(1, 1 * :scale)
\\set tid random(1, 10 * :scale)
\\set delta random(-5000, 5000)
BEGIN;
UPDATE pgbench_accounts SET abalance = abalance + :delta WHERE aid = :aid;
INSERT INTO pgbench_history (tid, bid, aid, delta, mtime) VALUES (:tid, :bid, :aid, :delta, CURRENT_TIMESTAMP);
END;
""",
    },
    "mixed_workload": {
        "name": "Mixed Workload Benchmark",
        "type": "pgbench",
        "scale": 100,
        "clients": 48,
        "threads": 8,
        "duration_seconds": 60,
        "custom_sql": """
-- Mixed workload: 70% reads, 30% writes
\\set aid random(1, 100000 * :scale)
\\set bid random(1, 1 * :scale)
\\set tid random(1, 10 * :scale)
\\set delta random(-5000, 5000)
\\set op random(1, 10)
SELECT CASE WHEN :op <= 7 THEN (
    SELECT abalance FROM pgbench_accounts WHERE aid = :aid
) ELSE (
    UPDATE pgbench_accounts SET abalance = abalance + :delta WHERE aid = :aid RETURNING abalance
) END;
""",
    },
    "analytical": {
        "name": "Analytical Query Benchmark",
        "type": "custom",
        "scale": 100,
        "clients": 8,
        "threads": 4,
        "duration_seconds": 60,
        "custom_sql": """
-- Analytical queries
SELECT
    bid,
    COUNT(*) as tx_count,
    SUM(delta) as total_delta,
    AVG(delta) as avg_delta,
    MIN(mtime) as first_tx,
    MAX(mtime) as last_tx
FROM pgbench_history
GROUP BY bid
ORDER BY tx_count DESC
LIMIT 10;

SELECT
    date_trunc('minute', mtime) as minute,
    COUNT(*) as tx_count,
    SUM(delta) as volume
FROM pgbench_history
WHERE mtime > NOW() - INTERVAL '1 hour'
GROUP BY 1
ORDER BY 1;
""",
    },
}


# =============================================================================
# OS TUNING RECOMMENDATIONS
# =============================================================================

OS_TUNING = {
    "sysctl": {
        "vm.swappiness": {"value": "10", "rationale": "Reduce swapping, keep data in RAM"},
        "vm.dirty_ratio": {"value": "40", "rationale": "Allow more dirty pages before flush"},
        "vm.dirty_background_ratio": {"value": "10", "rationale": "Start background writeback earlier"},
        "vm.dirty_expire_centisecs": {"value": "500", "rationale": "Flush dirty pages after 5 seconds"},
        "kernel.shmmax": {"value": "17179869184", "rationale": "16GB shared memory for PostgreSQL"},
        "kernel.shmall": {"value": "4194304", "rationale": "Total shared memory pages"},
        "net.core.somaxconn": {"value": "65535", "rationale": "Connection backlog for high concurrency"},
        "net.ipv4.tcp_max_syn_backlog": {"value": "65535", "rationale": "SYN queue for connections"},
    },
    "thp": {
        "enabled": "never",
        "defrag": "never",
        "rationale": "THP causes latency spikes in PostgreSQL",
    },
    "io_scheduler": {
        "nvme": "none",
        "ssd": "deadline",
        "hdd": "deadline",
        "rationale": "NVMe needs no scheduler, deadline for others",
    },
}


# =============================================================================
# MOCK RESULT PROGRESSION
# =============================================================================

# Simulates realistic TPS progression through tuning rounds
MOCK_TPS_PROGRESSION = {
    "balanced_tps": {
        "baseline": 5000,
        "round_0": 6000,    # +20% from initial tuning
        "round_1": 6900,    # +15% from WAL tuning
        "round_2": 7590,    # +10% from I/O tuning
        "round_3": 8350,    # +10% from parallelism
        "target": 9000,     # 80% improvement target
    },
    "wal_optimized": {
        "baseline": 5000,
        "round_0": 6500,    # +30% from aggressive WAL
        "round_1": 8125,    # +25% from async commit
        "target": 9000,
    },
    "error_scenario": {
        "baseline": 0,      # Simulate 0 TPS error
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_strategies_for_workload(workload: WorkloadType) -> List[Dict[str, Any]]:
    """Get suitable strategies for a workload type."""
    result = []
    for key, strategy in STRATEGIES.items():
        if workload in strategy["suitable_for"]:
            result.append({
                "id": key,
                **strategy
            })
    return result


def get_tuning_for_round(strategy: str, round_num: int) -> Dict[str, Any]:
    """Get tuning configuration for a specific strategy and round."""
    strategy_rounds = TUNING_ROUNDS.get(strategy, TUNING_ROUNDS["balanced_tps"])

    # If round exceeds available rounds, return last round
    max_round = max(strategy_rounds.keys())
    actual_round = min(round_num, max_round)

    return strategy_rounds.get(actual_round, strategy_rounds[0])


def get_benchmark_for_workload(workload: WorkloadType) -> Dict[str, Any]:
    """Get appropriate benchmark config for workload."""
    mapping = {
        WorkloadType.OLTP: "standard_oltp",
        WorkloadType.OLAP: "analytical",
        WorkloadType.MIXED: "mixed_workload",
    }
    key = mapping.get(workload, "standard_oltp")
    return BENCHMARK_CONFIGS[key]


def get_expected_tps(strategy: str, round_num: int) -> int:
    """Get expected TPS for a strategy at a given round."""
    progression = MOCK_TPS_PROGRESSION.get(strategy, MOCK_TPS_PROGRESSION["balanced_tps"])

    if round_num == 0:
        return progression.get("baseline", 5000)

    round_key = f"round_{round_num - 1}"
    return progression.get(round_key, progression.get("baseline", 5000))


def scale_config_for_profile(config: Dict, profile: SystemProfile) -> Dict:
    """Scale configuration values based on system profile."""
    # Scaling factors
    factors = {
        SystemProfile.SMALL: 0.25,   # 8GB RAM
        SystemProfile.MEDIUM: 1.0,   # 32GB RAM (baseline)
        SystemProfile.LARGE: 2.0,    # 64GB RAM
    }

    factor = factors.get(profile, 1.0)

    # Scale memory-related values
    scaled = config.copy()
    # Implementation would parse and scale values like '8GB' -> '2GB' for small

    return scaled


# =============================================================================
# FIRST SIGHT / DISCOVERY DATA
# =============================================================================

MOCK_FIRST_SIGHT = {
    "system_info": {
        "pg_version": "15.4",
        "os": "Ubuntu 22.04",
        "cpu_cores": 8,
        "ram_gb": 32,
        "storage_type": "NVMe",
    },
    "current_config": {
        "shared_buffers": "128MB",
        "effective_cache_size": "4GB",
        "work_mem": "4MB",
        "maintenance_work_mem": "64MB",
        "max_connections": "100",
        "wal_buffers": "-1",
        "checkpoint_completion_target": "0.5",
        "max_wal_size": "1GB",
    },
    "observations": [
        "shared_buffers is at default (128MB) - significantly undersized for 32GB RAM",
        "work_mem is at default (4MB) - may cause disk sorts for complex queries",
        "checkpoint_completion_target at 0.5 - checkpoints may be spiky",
        "max_connections at default - consider PgBouncer for connection pooling",
    ],
    "recommended_strategies": ["balanced_tps", "memory_focused"],
}


# =============================================================================
# ANALYSIS RESPONSE TEMPLATES
# =============================================================================

def generate_analysis_response(
    round_num: int,
    current_tps: float,
    target_tps: float,
    strategy: str,
) -> Dict[str, Any]:
    """Generate a realistic analysis response for a tuning round."""

    tuning_config = get_tuning_for_round(strategy, round_num)
    improvement_pct = ((current_tps - 5000) / 5000) * 100 if current_tps > 0 else 0
    gap_pct = ((target_tps - current_tps) / target_tps) * 100 if target_tps > 0 else 100

    # Build tuning chunks from config
    tuning_chunks = []
    for change in tuning_config.get("changes", []):
        tuning_chunks.append({
            "name": change["name"],
            "category": change["category"],
            "apply_commands": change["pg_configs"],
            "rationale": change["rationale"],
            "requires_restart": change.get("requires_restart", False),
        })

    return {
        "analysis": {
            "current_tps": current_tps,
            "target_tps": target_tps,
            "improvement_so_far": f"{improvement_pct:.1f}%",
            "gap_to_target": f"{gap_pct:.1f}%",
            "round": round_num,
        },
        "observations": [
            f"Current TPS: {current_tps:,.0f} ({improvement_pct:.1f}% improvement from baseline)",
            f"Target: {target_tps:,.0f} ({gap_pct:.1f}% gap remaining)",
            f"Applying {tuning_config['name']} for this round",
        ],
        "tuning_chunks": tuning_chunks,
        "expected_improvement": tuning_config.get("expected_improvement", 10),
        "confidence": 0.85 if round_num < 3 else 0.70,
    }


# =============================================================================
# DATA VALIDATION LAYER
# =============================================================================
# This layer validates golden data integrity to help distinguish between:
# - Tool bugs (pg_diagnose logic errors)
# - Mock data bugs (golden_data inconsistencies)
# - Test code bugs (test_mode.py issues)

class GoldenDataValidationError(Exception):
    """Raised when golden data validation fails."""
    pass


class GoldenDataValidator:
    """
    Validates golden data integrity.

    Run independently before tests to ensure mock data is consistent.
    If validation passes but tests fail, the bug is in the tool, not mock data.

    Usage:
        validator = GoldenDataValidator()
        errors = validator.validate_all()
        if errors:
            print("Golden data is corrupt!")
            for error in errors:
                print(f"  - {error}")
    """

    # Valid SQL command prefixes for PostgreSQL tuning
    VALID_SQL_PREFIXES = (
        'ALTER SYSTEM',
        'SELECT pg_reload_conf',
        'SELECT pg_',
        'SET ',
        'SHOW ',
        'CHECKPOINT',
        'VACUUM',
        'ANALYZE',
    )

    # Required fields in tuning changes
    REQUIRED_TUNING_FIELDS = {'name', 'category', 'pg_configs', 'rationale'}

    # Required fields in benchmark configs
    REQUIRED_BENCHMARK_FIELDS = {'scale', 'clients', 'threads', 'duration_seconds'}

    # Valid categories for tuning changes
    VALID_CATEGORIES = {
        'memory', 'wal', 'checkpoint', 'io', 'planner',
        'parallelism', 'vacuum', 'connection', 'misc'
    }

    def __init__(self):
        self.errors: List[str] = []

    def validate_all(self) -> List[str]:
        """
        Run all validations and return list of errors.

        Returns empty list if all validations pass.
        """
        self.errors = []

        self._validate_tps_progression()
        self._validate_tuning_rounds()
        self._validate_benchmark_configs()
        self._validate_strategies()
        self._validate_cross_references()

        return self.errors

    def _validate_tps_progression(self):
        """Validate TPS progression is monotonically increasing."""
        for scenario, progression in MOCK_TPS_PROGRESSION.items():
            if scenario == "error_scenario":
                continue  # Skip error scenario - intentionally broken

            # Extract TPS values in order
            baseline = progression.get("baseline", 0)
            target = progression.get("target", baseline)

            # Collect round values
            tps_sequence = [("baseline", baseline)]
            for i in range(10):  # Check up to 10 rounds
                key = f"round_{i}"
                if key in progression:
                    tps_sequence.append((key, progression[key]))

            # Validate monotonic increase
            prev_tps = 0
            prev_key = None
            for key, tps in tps_sequence:
                if tps < prev_tps:
                    self.errors.append(
                        f"TPS_PROGRESSION[{scenario}]: {key} ({tps}) < {prev_key} ({prev_tps}) - "
                        f"TPS should increase through rounds"
                    )
                prev_tps = tps
                prev_key = key

            # Validate baseline is reasonable
            if baseline <= 0 and scenario != "error_scenario":
                self.errors.append(
                    f"TPS_PROGRESSION[{scenario}]: baseline is {baseline}, expected > 0"
                )

            # Validate target is achievable
            if target > 0:
                max_tps = max(v for k, v in progression.items() if isinstance(v, (int, float)))
                if max_tps < target * 0.9:  # Within 10% of target
                    self.errors.append(
                        f"TPS_PROGRESSION[{scenario}]: max TPS ({max_tps}) doesn't reach "
                        f"90% of target ({target})"
                    )

    def _validate_tuning_rounds(self):
        """Validate tuning rounds have valid structure and SQL commands."""
        for strategy, rounds in TUNING_ROUNDS.items():
            if not isinstance(rounds, dict):
                self.errors.append(f"TUNING_ROUNDS[{strategy}]: expected dict, got {type(rounds)}")
                continue

            for round_num, config in rounds.items():
                prefix = f"TUNING_ROUNDS[{strategy}][{round_num}]"

                # Check required fields
                if "name" not in config:
                    self.errors.append(f"{prefix}: missing 'name' field")
                if "changes" not in config:
                    self.errors.append(f"{prefix}: missing 'changes' field")
                    continue
                if "expected_improvement" not in config:
                    self.errors.append(f"{prefix}: missing 'expected_improvement' field")

                # Validate expected_improvement is reasonable
                exp_imp = config.get("expected_improvement", 0)
                if not 0 <= exp_imp <= 100:
                    self.errors.append(
                        f"{prefix}: expected_improvement ({exp_imp}) should be 0-100%"
                    )

                # Validate each change
                for i, change in enumerate(config.get("changes", [])):
                    change_prefix = f"{prefix}.changes[{i}]"
                    self._validate_tuning_change(change, change_prefix)

    def _validate_tuning_change(self, change: Dict, prefix: str):
        """Validate a single tuning change."""
        # Check required fields
        missing = self.REQUIRED_TUNING_FIELDS - set(change.keys())
        if missing:
            self.errors.append(f"{prefix}: missing fields {missing}")

        # Validate category
        category = change.get("category", "")
        if category and category not in self.VALID_CATEGORIES:
            self.errors.append(
                f"{prefix}: invalid category '{category}', expected one of {self.VALID_CATEGORIES}"
            )

        # Validate SQL commands
        for j, cmd in enumerate(change.get("pg_configs", [])):
            if not self._is_valid_sql_command(cmd):
                self.errors.append(
                    f"{prefix}.pg_configs[{j}]: invalid SQL command '{cmd[:50]}...'"
                )

        # Validate requires_restart is boolean
        requires_restart = change.get("requires_restart")
        if requires_restart is not None and not isinstance(requires_restart, bool):
            self.errors.append(
                f"{prefix}: requires_restart should be bool, got {type(requires_restart)}"
            )

    def _is_valid_sql_command(self, cmd: str) -> bool:
        """Check if command is a valid PostgreSQL tuning command."""
        cmd = cmd.strip()
        for prefix in self.VALID_SQL_PREFIXES:
            if cmd.upper().startswith(prefix.upper()):
                return True
        return False

    def _validate_benchmark_configs(self):
        """Validate benchmark configurations."""
        for name, config in BENCHMARK_CONFIGS.items():
            prefix = f"BENCHMARK_CONFIGS[{name}]"

            # Check required fields
            missing = self.REQUIRED_BENCHMARK_FIELDS - set(config.keys())
            if missing:
                self.errors.append(f"{prefix}: missing fields {missing}")

            # Validate numeric ranges
            if config.get("scale", 0) <= 0:
                self.errors.append(f"{prefix}: scale should be > 0")
            if config.get("clients", 0) <= 0:
                self.errors.append(f"{prefix}: clients should be > 0")
            if config.get("threads", 0) <= 0:
                self.errors.append(f"{prefix}: threads should be > 0")
            if config.get("duration_seconds", 0) <= 0:
                self.errors.append(f"{prefix}: duration_seconds should be > 0")

            # Validate clients >= threads (common pgbench constraint)
            if config.get("clients", 1) < config.get("threads", 1):
                self.errors.append(
                    f"{prefix}: clients ({config.get('clients')}) should be >= "
                    f"threads ({config.get('threads')})"
                )

    def _validate_strategies(self):
        """Validate strategy definitions."""
        for name, strategy in STRATEGIES.items():
            prefix = f"STRATEGIES[{name}]"

            if "name" not in strategy:
                self.errors.append(f"{prefix}: missing 'name' field")
            if "hypothesis" not in strategy:
                self.errors.append(f"{prefix}: missing 'hypothesis' field")
            if "suitable_for" not in strategy:
                self.errors.append(f"{prefix}: missing 'suitable_for' field")
            elif not isinstance(strategy.get("suitable_for"), list):
                self.errors.append(f"{prefix}: 'suitable_for' should be a list")

    # Scenarios that are test-only (don't need TUNING_ROUNDS)
    TEST_ONLY_SCENARIOS = {"error_scenario", "snapshot_restore"}

    def _validate_cross_references(self):
        """Validate cross-references between data structures."""
        # Every strategy in TPS_PROGRESSION should have TUNING_ROUNDS
        # (except test-only scenarios)
        for scenario in MOCK_TPS_PROGRESSION.keys():
            if scenario in self.TEST_ONLY_SCENARIOS:
                continue
            if scenario not in TUNING_ROUNDS:
                self.errors.append(
                    f"MOCK_TPS_PROGRESSION has '{scenario}' but TUNING_ROUNDS doesn't"
                )

        # Validate MOCK_FIRST_SIGHT references valid strategies
        recommended = MOCK_FIRST_SIGHT.get("recommended_strategies", [])
        for strategy in recommended:
            if strategy not in STRATEGIES:
                self.errors.append(
                    f"MOCK_FIRST_SIGHT.recommended_strategies: '{strategy}' not in STRATEGIES"
                )


def validate_golden_data(raise_on_error: bool = False) -> List[str]:
    """
    Validate all golden data and return errors.

    Args:
        raise_on_error: If True, raise GoldenDataValidationError on first error

    Returns:
        List of validation error messages (empty if valid)

    Usage:
        # Quick check
        errors = validate_golden_data()
        assert not errors, f"Golden data corrupt: {errors}"

        # Or raise exception
        validate_golden_data(raise_on_error=True)
    """
    validator = GoldenDataValidator()
    errors = validator.validate_all()

    if errors and raise_on_error:
        raise GoldenDataValidationError(
            f"Golden data validation failed with {len(errors)} error(s):\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    return errors


def run_validation_report():
    """Run validation and print a detailed report."""
    print("=" * 60)
    print("GOLDEN DATA VALIDATION REPORT")
    print("=" * 60)
    print()

    validator = GoldenDataValidator()
    errors = validator.validate_all()

    # Summary
    print(f"Scenarios validated: {len(MOCK_TPS_PROGRESSION)}")
    print(f"Tuning strategies: {len(TUNING_ROUNDS)}")
    print(f"Benchmark configs: {len(BENCHMARK_CONFIGS)}")
    print(f"Strategies: {len(STRATEGIES)}")
    print()

    if errors:
        print(f"VALIDATION FAILED - {len(errors)} error(s) found:")
        print("-" * 60)
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print()
        return False
    else:
        print("VALIDATION PASSED - All golden data is consistent")
        print()
        return True


# =============================================================================
# SNAPSHOT TEST DATA
# =============================================================================
# Realistic PostgreSQL configuration states for snapshot/restore testing

SNAPSHOT_TEST_DATA = {
    # Initial state - before any tuning (PostgreSQL defaults + basic setup)
    "initial_pg_settings": {
        "shared_buffers": "128MB",
        "effective_cache_size": "4GB",
        "work_mem": "4MB",
        "maintenance_work_mem": "64MB",
        "wal_buffers": "-1",
        "checkpoint_completion_target": "0.9",
        "max_connections": "100",
        "random_page_cost": "4",
        "effective_io_concurrency": "1",
        "max_parallel_workers_per_gather": "2",
        "max_parallel_workers": "8",
        "max_wal_size": "1GB",
        "min_wal_size": "80MB",
        "checkpoint_timeout": "5min",
        "default_statistics_target": "100",
        "wal_compression": "off",
    },

    # Tuned state - after optimization
    "tuned_pg_settings": {
        "shared_buffers": "8GB",
        "effective_cache_size": "24GB",
        "work_mem": "64MB",
        "maintenance_work_mem": "2GB",
        "wal_buffers": "64MB",
        "checkpoint_completion_target": "0.9",
        "max_connections": "200",
        "random_page_cost": "1.1",
        "effective_io_concurrency": "200",
        "max_parallel_workers_per_gather": "4",
        "max_parallel_workers": "8",
        "max_wal_size": "4GB",
        "min_wal_size": "1GB",
        "checkpoint_timeout": "15min",
        "default_statistics_target": "200",
        "wal_compression": "on",
    },

    # Contents of postgresql.auto.conf before any tuning
    "initial_auto_conf": "",

    # Contents of postgresql.auto.conf after tuning
    "tuned_auto_conf": """# Do not edit this file manually!
# It will be overwritten by ALTER SYSTEM command.
shared_buffers = '8GB'
effective_cache_size = '24GB'
work_mem = '64MB'
maintenance_work_mem = '2GB'
wal_buffers = '64MB'
random_page_cost = '1.1'
effective_io_concurrency = '200'
max_wal_size = '4GB'
min_wal_size = '1GB'
checkpoint_timeout = '15min'
wal_compression = 'on'
""",

    # MD5 hashes for config file tracking
    "initial_conf_hash": "abc123def456",
    "tuned_conf_hash": "789xyz012abc",
}

# TPS progression for snapshot restore scenarios
MOCK_TPS_PROGRESSION["snapshot_restore"] = {
    "baseline": 5000,
    "round_1": 6000,
    "round_2": 6500,
    "after_restore": 5000,      # Back to baseline after restore
    "round_1_retry": 6200,      # Slightly different due to variance
}


# =============================================================================
# AUTO-VALIDATION ON IMPORT (optional)
# =============================================================================
# Set PG_DIAGNOSE_VALIDATE_MOCKS=1 to validate on import

import os
if os.environ.get('PG_DIAGNOSE_VALIDATE_MOCKS', '').lower() in ('1', 'true', 'yes'):
    errors = validate_golden_data()
    if errors:
        import warnings
        warnings.warn(
            f"Golden data validation failed with {len(errors)} error(s). "
            f"Set PG_DIAGNOSE_VALIDATE_MOCKS=0 to disable. Errors: {errors[:3]}..."
        )
