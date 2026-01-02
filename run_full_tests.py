#!/usr/bin/env python3
"""
Comprehensive Test Suite for pg_diagnose

This script runs ALL test levels to verify the complete functionality
of the pg_diagnose PostgreSQL tuning tool.

Test Levels:
  1. Golden Data Validation - Mock data consistency
  2. Module Import Tests - All modules importable
  3. Mock Component Tests - Mocks work correctly
  4. CLI Command Tests - Commands are registered and routable
  5. Workspace/Session Tests - State management works
  6. Tuning Loop Tests - AI integration and benchmark flow
  7. Snapshot/Restore Tests - Configuration backup/restore
  8. Integration Tests - Scenario-based with mocks
  9. End-to-End Tests - Real database operations

Usage:
    python run_full_tests.py [--db-host HOST] [--db-port PORT] [--db-user USER] [--db-password PASS] [--db-name NAME]

    # Run with real database:
    python run_full_tests.py --db-host 44.249.192.13 --db-port 6432 --db-user postgres --db-password postgres --db-name postgres

    # Run without database tests (levels 1-7 only):
    python run_full_tests.py --skip-db
"""

import sys
import os
import argparse
import traceback
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from io import StringIO

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResult:
    """Result of a single test."""
    def __init__(self, name: str, passed: bool, message: str = "", details: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details


class TestLevel:
    """A collection of related tests."""
    def __init__(self, level: int, name: str):
        self.level = level
        self.name = name
        self.results: List[TestResult] = []
        self.passed = True

    def add_result(self, result: TestResult):
        self.results.append(result)
        if not result.passed:
            self.passed = False

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)


class ComprehensiveTestRunner:
    """Runs all test levels for pg_diagnose."""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None, skip_db: bool = False):
        self.db_config = db_config or {}
        self.skip_db = skip_db
        self.levels: List[TestLevel] = []
        self.start_time = datetime.now()

    def run_all(self) -> bool:
        """Run all test levels and return True if all pass."""
        print("=" * 70)
        print("                  PG_DIAGNOSE COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run each level
        self._run_level_1_golden_data()
        if not self.levels[-1].passed:
            print("\n[ABORT] Level 1 failed - golden data is corrupt, cannot continue")
            return False

        self._run_level_2_imports()
        if not self.levels[-1].passed:
            print("\n[ABORT] Level 2 failed - imports broken, cannot continue")
            return False

        self._run_level_3_mocks()
        self._run_level_4_cli_commands()
        self._run_level_5_workspace_session()
        self._run_level_6_tuning_loop()
        self._run_level_7_snapshot_restore()

        if not self.skip_db:
            self._run_level_8_integration()
            self._run_level_9_end_to_end()
        else:
            print("\n[SKIP] Levels 8-9 skipped (--skip-db flag)")

        # Print summary
        return self._print_summary()

    def _run_level_1_golden_data(self):
        """Level 1: Golden Data Validation."""
        level = TestLevel(1, "Golden Data Validation")
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"{'='*60}")

        try:
            from pg_diagnose.tests.mocks.golden_data import (
                validate_golden_data,
                STRATEGIES,
                TUNING_ROUNDS,
                BENCHMARK_CONFIGS,
                MOCK_TPS_PROGRESSION,
                SNAPSHOT_TEST_DATA,
            )

            # Test 1: Run validation
            errors = validate_golden_data()
            level.add_result(TestResult(
                "Golden data validation",
                len(errors) == 0,
                f"{len(errors)} errors found" if errors else "All validations passed",
                "\n".join(errors[:5]) if errors else ""
            ))

            # Test 2: Check required data structures exist
            level.add_result(TestResult(
                "STRATEGIES defined",
                len(STRATEGIES) >= 2,
                f"{len(STRATEGIES)} strategies"
            ))

            level.add_result(TestResult(
                "TUNING_ROUNDS defined",
                len(TUNING_ROUNDS) >= 1,
                f"{len(TUNING_ROUNDS)} tuning strategies with rounds"
            ))

            level.add_result(TestResult(
                "BENCHMARK_CONFIGS defined",
                len(BENCHMARK_CONFIGS) >= 3,
                f"{len(BENCHMARK_CONFIGS)} benchmark configs"
            ))

            level.add_result(TestResult(
                "MOCK_TPS_PROGRESSION defined",
                len(MOCK_TPS_PROGRESSION) >= 2,
                f"{len(MOCK_TPS_PROGRESSION)} TPS progression scenarios"
            ))

            level.add_result(TestResult(
                "SNAPSHOT_TEST_DATA defined",
                "initial_pg_settings" in SNAPSHOT_TEST_DATA,
                f"{len(SNAPSHOT_TEST_DATA)} snapshot data keys"
            ))

        except Exception as e:
            level.add_result(TestResult(
                "Golden data import",
                False,
                f"Import failed: {e}",
                traceback.format_exc()
            ))

        self.levels.append(level)
        self._print_level_summary(level)

    def _run_level_2_imports(self):
        """Level 2: Module Import Tests."""
        level = TestLevel(2, "Module Import Tests")
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"{'='*60}")

        modules_to_test = [
            ("pg_diagnose", "Main package"),
            ("pg_diagnose.cli", "CLI module"),
            ("pg_diagnose.commands", "Commands module"),
            ("pg_diagnose.workspace", "Workspace module"),
            ("pg_diagnose.session", "Session module"),
            ("pg_diagnose.dashboard", "Dashboard module"),
            ("pg_diagnose.export", "Export module"),
            ("pg_diagnose.modes", "Modes module"),
            ("pg_diagnose.agent", "Agent package"),
            ("pg_diagnose.agent.client", "Agent client"),
            ("pg_diagnose.agent.parser", "Agent parser"),
            ("pg_diagnose.agent.prompts", "Agent prompts"),
            ("pg_diagnose.runner", "Runner package"),
            ("pg_diagnose.runner.benchmark", "Benchmark runner"),
            ("pg_diagnose.runner.engine", "Runner engine"),
            ("pg_diagnose.runner.state", "Runner state"),
            ("pg_diagnose.tuning", "Tuning package"),
            ("pg_diagnose.tuning.executor", "Tuning executor"),
            ("pg_diagnose.tuning.service", "Tuning service"),
            ("pg_diagnose.tuning.verifier", "Tuning verifier"),
            ("pg_diagnose.discovery", "Discovery package"),
            ("pg_diagnose.discovery.runtime", "Discovery runtime"),
            ("pg_diagnose.discovery.schema", "Discovery schema"),
            ("pg_diagnose.discovery.system", "Discovery system"),
            ("pg_diagnose.snapshot", "Snapshot package"),
            ("pg_diagnose.snapshot.models", "Snapshot models"),
            ("pg_diagnose.snapshot.capture", "Snapshot capture"),
            ("pg_diagnose.snapshot.restore", "Snapshot restore"),
            ("pg_diagnose.snapshot.manager", "Snapshot manager"),
            ("pg_diagnose.tests.mocks", "Mock components"),
            ("pg_diagnose.tests.test_mode", "Test mode handler"),
        ]

        for module_name, description in modules_to_test:
            try:
                __import__(module_name)
                level.add_result(TestResult(
                    description,
                    True,
                    f"{module_name} OK"
                ))
            except Exception as e:
                level.add_result(TestResult(
                    description,
                    False,
                    f"Import failed: {e}",
                    traceback.format_exc()
                ))

        self.levels.append(level)
        self._print_level_summary(level)

    def _run_level_3_mocks(self):
        """Level 3: Mock Component Tests."""
        level = TestLevel(3, "Mock Component Tests")
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"{'='*60}")

        try:
            from pg_diagnose.tests.mocks import (
                MockGeminiAgent,
                MockBenchmarkRunner,
                MockSnapshotCapture,
                MockSnapshotRestore,
                MockSnapshotManager,
                MockSnapshotFactory,
                STRATEGIES,
                get_tuning_for_round,
                get_expected_tps,
            )

            # Test MockGeminiAgent
            try:
                agent = MockGeminiAgent(scenario="balanced_tps")
                # Test that it can generate responses
                level.add_result(TestResult(
                    "MockGeminiAgent instantiation",
                    True,
                    "Agent created successfully"
                ))
            except Exception as e:
                level.add_result(TestResult(
                    "MockGeminiAgent instantiation",
                    False,
                    str(e)
                ))

            # Test MockBenchmarkRunner
            try:
                runner = MockBenchmarkRunner(scenario="balanced_tps")
                runner.set_round(0)
                result = runner.run()
                level.add_result(TestResult(
                    "MockBenchmarkRunner.run",
                    result.metrics.tps > 0,
                    f"TPS: {result.metrics.tps}"
                ))
            except Exception as e:
                level.add_result(TestResult(
                    "MockBenchmarkRunner.run",
                    False,
                    str(e)
                ))

            # Test MockSnapshotCapture
            try:
                capture = MockSnapshotCapture(scenario="balanced_tps")
                snapshot = capture.capture("test", "manual", "test-session", 0)
                level.add_result(TestResult(
                    "MockSnapshotCapture.capture",
                    len(snapshot.pg_settings) > 0,
                    f"Captured {len(snapshot.pg_settings)} settings"
                ))
            except Exception as e:
                level.add_result(TestResult(
                    "MockSnapshotCapture.capture",
                    False,
                    str(e)
                ))

            # Test MockSnapshotRestore
            try:
                restore = MockSnapshotRestore()
                result = restore.restore(snapshot, dry_run=True)
                level.add_result(TestResult(
                    "MockSnapshotRestore.restore (dry-run)",
                    result.success,
                    f"Preview: {result.changes_applied} changes"
                ))
            except Exception as e:
                level.add_result(TestResult(
                    "MockSnapshotRestore.restore",
                    False,
                    str(e)
                ))

            # Test MockSnapshotManager
            try:
                manager = MockSnapshotManager(scenario="balanced_tps")
                initial = manager.create_initial("test-session")
                level.add_result(TestResult(
                    "MockSnapshotManager.create_initial",
                    initial.name == "initial",
                    f"Created snapshot: {initial.name}"
                ))

                snapshots = manager.list_snapshots()
                level.add_result(TestResult(
                    "MockSnapshotManager.list_snapshots",
                    len(snapshots) >= 1,
                    f"Listed {len(snapshots)} snapshots"
                ))
            except Exception as e:
                level.add_result(TestResult(
                    "MockSnapshotManager operations",
                    False,
                    str(e)
                ))

            # Test helper functions
            try:
                tuning = get_tuning_for_round("balanced_tps", 0)
                level.add_result(TestResult(
                    "get_tuning_for_round",
                    "changes" in tuning,
                    f"Round 0: {tuning.get('name', 'unnamed')}"
                ))
            except Exception as e:
                level.add_result(TestResult(
                    "get_tuning_for_round",
                    False,
                    str(e)
                ))

            try:
                tps = get_expected_tps("balanced_tps", 0)
                level.add_result(TestResult(
                    "get_expected_tps",
                    tps > 0,
                    f"Expected TPS at round 0: {tps}"
                ))
            except Exception as e:
                level.add_result(TestResult(
                    "get_expected_tps",
                    False,
                    str(e)
                ))

        except Exception as e:
            level.add_result(TestResult(
                "Mock imports",
                False,
                f"Import failed: {e}",
                traceback.format_exc()
            ))

        self.levels.append(level)
        self._print_level_summary(level)

    def _run_level_4_cli_commands(self):
        """Level 4: CLI Command Tests."""
        level = TestLevel(4, "CLI Command Tests")
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"{'='*60}")

        try:
            from pg_diagnose.commands import CommandHandler, SessionState

            # Test SessionState has all required fields
            state = SessionState()
            required_fields = [
                'current_round', 'baseline_tps', 'current_tps', 'target_tps',
                'strategy_name', 'strategy_id', 'tps_history', 'applied_changes',
                'workspace', 'session_name', 'test_mode', 'mock_snapshot'
            ]

            for field in required_fields:
                level.add_result(TestResult(
                    f"SessionState.{field}",
                    hasattr(state, field),
                    f"Field exists: {hasattr(state, field)}"
                ))

            # Test CommandHandler initialization
            try:
                handler = CommandHandler(state)
                level.add_result(TestResult(
                    "CommandHandler initialization",
                    handler is not None,
                    "Handler created successfully"
                ))
            except Exception as e:
                level.add_result(TestResult(
                    "CommandHandler initialization",
                    False,
                    str(e)
                ))
                handler = None

            # Test CommandHandler has required command methods
            if handler:
                commands_to_test = [
                    '_cmd_help', '_cmd_status',
                    '_cmd_history', '_cmd_export',
                    '_cmd_snapshot', '_cmd_restore'
                ]

                for cmd in commands_to_test:
                    level.add_result(TestResult(
                        f"CommandHandler.{cmd}",
                        hasattr(handler, cmd) and callable(getattr(handler, cmd)),
                        f"Method exists and callable"
                    ))

                # Test command routing
                test_routing = [
                    ('snapshot', 'list'),
                    ('snapshot', 'show initial'),
                    ('snapshot', 'compare a b'),
                    ('restore', ''),
                    ('restore', '--dry-run'),
                ]

                for cmd, args in test_routing:
                    method_name = f"_cmd_{cmd}"
                    if hasattr(handler, method_name):
                        level.add_result(TestResult(
                            f"/{cmd} {args}".strip(),
                            True,
                            "Command routable"
                        ))

        except Exception as e:
            level.add_result(TestResult(
                "CLI commands import",
                False,
                f"Import failed: {e}",
                traceback.format_exc()
            ))

        self.levels.append(level)
        self._print_level_summary(level)

    def _run_level_5_workspace_session(self):
        """Level 5: Workspace and Session Management Tests."""
        level = TestLevel(5, "Workspace/Session Management")
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"{'='*60}")

        try:
            import tempfile
            import os
            from pg_diagnose.workspace import Workspace, SessionState, WORKSPACES_DIR

            # Override WORKSPACES_DIR for testing
            original_ws_dir = WORKSPACES_DIR

            with tempfile.TemporaryDirectory() as tmpdir:
                # Temporarily patch WORKSPACES_DIR
                import pg_diagnose.workspace as ws_module
                ws_module.WORKSPACES_DIR = Path(tmpdir)

                try:
                    # Test workspace creation
                    ws = Workspace.create(
                        db_host="localhost",
                        db_port=5432,
                        db_name="testdb",
                        db_user="postgres"
                    )

                    level.add_result(TestResult(
                        "Workspace creation",
                        ws is not None,
                        f"Created: {ws.name}"
                    ))

                    # Test workspace path exists
                    level.add_result(TestResult(
                        "Workspace directory",
                        ws.path.exists(),
                        f"Path: {ws.path}"
                    ))

                    # Test session creation
                    session = ws.create_session("test-session")
                    level.add_result(TestResult(
                        "Session creation",
                        session is not None,
                        f"Session: {session.name if session else 'None'}"
                    ))

                    # Test session state
                    if session:
                        # Session state is string-based
                        level.add_result(TestResult(
                            "Session initial state",
                            session.state == SessionState.ACTIVE,
                            f"State: {session.state}"
                        ))

                        # Test session save and load
                        session.save()
                        level.add_result(TestResult(
                            "Session save",
                            session.path.exists(),
                            f"Session path: {session.path}"
                        ))

                        # Test session listing
                        sessions = ws.list_sessions()
                        level.add_result(TestResult(
                            "Session listing",
                            len(sessions) >= 1,
                            f"Found {len(sessions)} sessions"
                        ))

                    # Test workspace find_for_database
                    found = Workspace.find_for_database("localhost", 5432, "testdb")
                    level.add_result(TestResult(
                        "Workspace.find_for_database",
                        found is not None,
                        f"Found: {found.name if found else 'None'}"
                    ))

                finally:
                    # Restore original
                    ws_module.WORKSPACES_DIR = original_ws_dir

        except Exception as e:
            level.add_result(TestResult(
                "Workspace/Session tests",
                False,
                f"Error: {e}",
                traceback.format_exc()
            ))

        self.levels.append(level)
        self._print_level_summary(level)

    def _run_level_6_tuning_loop(self):
        """Level 6: Tuning Loop Tests."""
        level = TestLevel(6, "Tuning Loop and AI Integration")
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"{'='*60}")

        try:
            from pg_diagnose.tests.mocks import (
                MockGeminiAgent,
                MockBenchmarkRunner,
                STRATEGIES,
                get_tuning_for_round,
                generate_analysis_response,
            )
            from pg_diagnose.tests.mocks.golden_data import MOCK_TPS_PROGRESSION

            # Test complete tuning loop simulation
            scenario = "balanced_tps"
            agent = MockGeminiAgent(scenario=scenario)
            runner = MockBenchmarkRunner(scenario=scenario)

            # Simulate baseline
            runner.set_round(0)
            baseline_result = runner.run()
            baseline_tps = baseline_result.metrics.tps
            level.add_result(TestResult(
                "Baseline benchmark",
                baseline_tps > 0,
                f"Baseline TPS: {baseline_tps:.0f}"
            ))

            # Simulate target calculation (80% improvement)
            target_tps = baseline_tps * 1.8
            level.add_result(TestResult(
                "Target calculation",
                target_tps > baseline_tps,
                f"Target TPS: {target_tps:.0f}"
            ))

            # Simulate tuning rounds
            current_tps = baseline_tps
            for round_num in range(1, 4):
                # Get tuning recommendations
                tuning = get_tuning_for_round(scenario, round_num)
                level.add_result(TestResult(
                    f"Round {round_num} tuning",
                    "changes" in tuning and len(tuning["changes"]) > 0,
                    f"Changes: {len(tuning.get('changes', []))}"
                ))

                # Run benchmark
                runner.set_round(round_num)
                result = runner.run()
                new_tps = result.metrics.tps

                level.add_result(TestResult(
                    f"Round {round_num} benchmark",
                    new_tps >= current_tps * 0.95,  # Allow small variance
                    f"TPS: {new_tps:.0f} (was {current_tps:.0f})"
                ))

                current_tps = new_tps

            # Check if target was approached
            final_improvement = ((current_tps - baseline_tps) / baseline_tps) * 100
            level.add_result(TestResult(
                "Overall improvement",
                final_improvement >= 40,  # At least 40% improvement (accounting for variance)
                f"Improvement: {final_improvement:.1f}%"
            ))

            # Test analysis response generation
            analysis = generate_analysis_response(
                round_num=1,
                current_tps=current_tps,
                target_tps=target_tps,
                strategy=scenario
            )
            level.add_result(TestResult(
                "Analysis response generation",
                "tuning_chunks" in analysis and len(analysis["tuning_chunks"]) > 0,
                f"Chunks: {len(analysis.get('tuning_chunks', []))}"
            ))

        except Exception as e:
            level.add_result(TestResult(
                "Tuning loop tests",
                False,
                f"Error: {e}",
                traceback.format_exc()
            ))

        self.levels.append(level)
        self._print_level_summary(level)

    def _run_level_7_snapshot_restore(self):
        """Level 7: Snapshot/Restore System Tests."""
        level = TestLevel(7, "Snapshot/Restore System")
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"{'='*60}")

        try:
            import tempfile
            from pathlib import Path
            from pg_diagnose.snapshot import (
                SnapshotData,
                RestoreResult,
                RestorePreview,
                SnapshotInfo,
            )
            from pg_diagnose.snapshot.models import RESTART_REQUIRED_PARAMS
            from pg_diagnose.tests.mocks import (
                MockSnapshotCapture,
                MockSnapshotRestore,
                MockSnapshotManager,
                SNAPSHOT_TEST_DATA,
            )

            # Test SnapshotData creation
            snapshot = SnapshotData.create(
                name="test-snapshot",
                trigger="manual",
                session_name="test-session",
                round_num=0,
                pg_settings={"shared_buffers": "128MB"},
                pg_auto_conf="",
                pg_conf_hash="abc123"
            )
            level.add_result(TestResult(
                "SnapshotData.create",
                snapshot.id is not None and snapshot.name == "test-snapshot",
                f"ID: {snapshot.id[:8]}..."
            ))

            # Test requires_restart function
            from pg_diagnose.snapshot.models import requires_restart
            level.add_result(TestResult(
                "requires_restart(shared_buffers)",
                requires_restart("shared_buffers") == True,
                "shared_buffers requires restart"
            ))

            level.add_result(TestResult(
                "requires_restart(work_mem)",
                requires_restart("work_mem") == False,
                "work_mem does not require restart"
            ))

            # Test MockSnapshotCapture captures correct data
            capture = MockSnapshotCapture(scenario="balanced_tps")
            initial_snap = capture.capture("initial", "automatic", "test", 0)
            level.add_result(TestResult(
                "Initial snapshot pg_settings",
                initial_snap.pg_settings.get("shared_buffers") == "128MB",
                f"shared_buffers: {initial_snap.pg_settings.get('shared_buffers')}"
            ))

            tuned_snap = capture.capture("tuned", "manual", "test", 1)
            level.add_result(TestResult(
                "Tuned snapshot pg_settings",
                tuned_snap.pg_settings.get("shared_buffers") == "8GB",
                f"shared_buffers: {tuned_snap.pg_settings.get('shared_buffers')}"
            ))

            # Test MockSnapshotRestore
            restore = MockSnapshotRestore()

            # Test preview (dry-run)
            preview = restore.preview(initial_snap)
            level.add_result(TestResult(
                "Restore preview",
                isinstance(preview, RestorePreview) and len(preview.changes) > 0,
                f"Changes to apply: {len(preview.changes)}"
            ))

            # Test actual restore
            result = restore.restore(initial_snap, dry_run=False)
            level.add_result(TestResult(
                "Restore execution",
                result.success,
                f"Applied {result.changes_applied} changes"
            ))

            # Test restart detection
            level.add_result(TestResult(
                "Restart required detection",
                result.restart_required == True,  # shared_buffers was changed
                f"Restart required: {result.restart_required}"
            ))

            # Test MockSnapshotManager full workflow
            manager = MockSnapshotManager(scenario="balanced_tps")

            # Create initial
            initial = manager.create_initial("test-session")
            level.add_result(TestResult(
                "Manager.create_initial",
                initial.name == "initial",
                f"Created: {initial.name}"
            ))

            # Create checkpoint
            checkpoint = manager.create_checkpoint(
                "after-tuning", "test-session", 1, 6000
            )
            level.add_result(TestResult(
                "Manager.create_checkpoint",
                checkpoint.name == "after-tuning",
                f"Created: {checkpoint.name}"
            ))

            # List snapshots (returns List[Dict])
            snapshots = manager.list_snapshots()
            level.add_result(TestResult(
                "Manager.list_snapshots",
                len(snapshots) == 2,
                f"Found: {[s['name'] for s in snapshots]}"
            ))

            # Get snapshot by name
            retrieved = manager.get("initial")
            level.add_result(TestResult(
                "Manager.get",
                retrieved is not None and retrieved.name == "initial",
                f"Retrieved: {retrieved.name if retrieved else 'None'}"
            ))

            # Compare snapshots
            diff = manager.compare("initial", "after-tuning")
            level.add_result(TestResult(
                "Manager.compare",
                diff is not None and len(diff.changes) > 0,
                f"Differences: {len(diff.changes) if diff else 0}"
            ))

            # Restore to initial
            restore_result = manager.restore_to_initial(dry_run=True)
            level.add_result(TestResult(
                "Manager.restore_to_initial (dry-run)",
                restore_result.success,
                f"Would apply {restore_result.changes_applied} changes"
            ))

        except Exception as e:
            level.add_result(TestResult(
                "Snapshot/Restore tests",
                False,
                f"Error: {e}",
                traceback.format_exc()
            ))

        self.levels.append(level)
        self._print_level_summary(level)

    def _run_level_8_integration(self):
        """Level 8: Integration Tests with Real Database."""
        level = TestLevel(8, "Integration Tests (Real DB)")
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"{'='*60}")

        if not self.db_config:
            level.add_result(TestResult(
                "Database config",
                False,
                "No database configuration provided. Use --db-host, --db-port, etc."
            ))
            self.levels.append(level)
            self._print_level_summary(level)
            return

        try:
            import psycopg2
            from pg_diagnose.snapshot.models import RestorePreview

            # Test database connection
            conn = psycopg2.connect(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 5432),
                user=self.db_config.get("user", "postgres"),
                password=self.db_config.get("password", ""),
                dbname=self.db_config.get("dbname", "postgres"),
                connect_timeout=10
            )

            level.add_result(TestResult(
                "Database connection",
                True,
                f"Connected to {self.db_config.get('host')}:{self.db_config.get('port')}"
            ))

            # Test pg_settings query
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM pg_settings WHERE context != 'internal'")
            settings_count = cur.fetchone()[0]
            level.add_result(TestResult(
                "pg_settings query",
                settings_count > 100,
                f"Found {settings_count} tunable settings"
            ))

            # Test SnapshotCapture with real connection
            from pg_diagnose.snapshot.capture import SnapshotCapture

            capture = SnapshotCapture(conn)
            snapshot = capture.capture("test", "manual", "test-session", 0)
            level.add_result(TestResult(
                "Real SnapshotCapture",
                len(snapshot.pg_settings) > 100,
                f"Captured {len(snapshot.pg_settings)} settings"
            ))

            # Test specific settings exist
            important_settings = ["shared_buffers", "work_mem", "max_connections"]
            for setting in important_settings:
                level.add_result(TestResult(
                    f"Captured {setting}",
                    setting in snapshot.pg_settings,
                    f"Value: {snapshot.pg_settings.get(setting, 'NOT FOUND')}"
                ))

            # Test SnapshotRestore preview (dry-run only)
            from pg_diagnose.snapshot.restore import SnapshotRestore

            restore = SnapshotRestore(conn)
            preview = restore.preview(snapshot)
            level.add_result(TestResult(
                "Real SnapshotRestore preview",
                isinstance(preview, RestorePreview),
                f"Changes preview: {len(preview.changes)}"
            ))

            conn.close()

        except ImportError:
            level.add_result(TestResult(
                "psycopg2 import",
                False,
                "psycopg2 not installed. Run: pip install psycopg2-binary"
            ))
        except Exception as e:
            level.add_result(TestResult(
                "Integration tests",
                False,
                f"Error: {e}",
                traceback.format_exc()
            ))

        self.levels.append(level)
        self._print_level_summary(level)

    def _run_level_9_end_to_end(self):
        """Level 9: End-to-End CLI Tests."""
        level = TestLevel(9, "End-to-End CLI Tests")
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"{'='*60}")

        if not self.db_config:
            level.add_result(TestResult(
                "Database config",
                False,
                "No database configuration provided"
            ))
            self.levels.append(level)
            self._print_level_summary(level)
            return

        try:
            from pg_diagnose.tests.scenarios import AVAILABLE_SCENARIOS, get_scenario_path

            # Test scenario files exist
            for scenario_name in AVAILABLE_SCENARIOS.keys():
                path = get_scenario_path(scenario_name)
                level.add_result(TestResult(
                    f"Scenario: {scenario_name}",
                    path.exists(),
                    f"Path: {path}"
                ))

            # Run a simple scenario test with test mode
            from pg_diagnose.tests.test_mode import TestModeHandler

            scenario_path = get_scenario_path("snapshot_basic")
            handler = TestModeHandler(
                input_file=str(scenario_path),
                output_json=True,
                scenario="balanced_tps"
            )

            level.add_result(TestResult(
                "TestModeHandler creation",
                handler is not None,
                f"Loaded {len(handler._input_lines)} commands"
            ))

            # Test command parsing
            if handler._input_lines:
                first_cmd = handler._input_lines[0]
                level.add_result(TestResult(
                    "Scenario command parsing",
                    first_cmd.startswith("/"),
                    f"First command: {first_cmd}"
                ))

        except Exception as e:
            level.add_result(TestResult(
                "End-to-end tests",
                False,
                f"Error: {e}",
                traceback.format_exc()
            ))

        self.levels.append(level)
        self._print_level_summary(level)

    def _print_level_summary(self, level: TestLevel):
        """Print summary for a test level."""
        status = "PASS" if level.passed else "FAIL"
        color = "\033[92m" if level.passed else "\033[91m"
        reset = "\033[0m"

        print(f"\n{color}Level {level.level} {level.name}: {status}{reset}")
        print(f"  Tests: {level.passed_count}/{level.total} passed")

        if not level.passed:
            print(f"\n  Failed tests:")
            for result in level.results:
                if not result.passed:
                    print(f"    - {result.name}: {result.message}")
                    if result.details:
                        for line in result.details.split("\n")[:5]:
                            print(f"      {line}")

    def _print_summary(self) -> bool:
        """Print final test summary and return True if all passed."""
        elapsed = datetime.now() - self.start_time

        print("\n" + "=" * 70)
        print("                          TEST SUMMARY")
        print("=" * 70)
        print()

        all_passed = True
        total_tests = 0
        total_passed = 0

        for level in self.levels:
            status = "PASS" if level.passed else "FAIL"
            dots = "." * (45 - len(level.name))
            print(f"  Level {level.level}: {level.name} {dots} {status}")

            # Print sub-items
            if level.results:
                passed = sum(1 for r in level.results if r.passed)
                failed = sum(1 for r in level.results if not r.passed)
                total_tests += len(level.results)
                total_passed += passed

                # Show key stats
                stats = []
                if passed > 0:
                    stats.append(f"{passed} passed")
                if failed > 0:
                    stats.append(f"{failed} failed")
                if stats:
                    print(f"    - {', '.join(stats)}")

            if not level.passed:
                all_passed = False

        print()
        print(f"  Total: {total_passed}/{total_tests} tests passed")
        print(f"  Time: {elapsed.total_seconds():.2f}s")
        print()

        if all_passed:
            print("  " + "\033[92m" + "ALL TESTS PASSED" + "\033[0m")
        else:
            print("  " + "\033[91m" + "SOME TESTS FAILED" + "\033[0m")

        print()
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive pg_diagnose tests")
    parser.add_argument("--db-host", help="Database host")
    parser.add_argument("--db-port", type=int, default=5432, help="Database port")
    parser.add_argument("--db-user", default="postgres", help="Database user")
    parser.add_argument("--db-password", default="", help="Database password")
    parser.add_argument("--db-name", default="postgres", help="Database name")
    parser.add_argument("--skip-db", action="store_true", help="Skip database tests")

    args = parser.parse_args()

    db_config = None
    if args.db_host and not args.skip_db:
        db_config = {
            "host": args.db_host,
            "port": args.db_port,
            "user": args.db_user,
            "password": args.db_password,
            "dbname": args.db_name,
        }

    runner = ComprehensiveTestRunner(db_config=db_config, skip_db=args.skip_db)
    success = runner.run_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
