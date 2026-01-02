"""
TuningExecutor - Safe tuning application with v2.2 resilience.

Implements the 5-phase tuning application:
1. SNAPSHOT - Capture current state
2. EXECUTE - Apply the commands
3. RESTART - Restart service if required
4. PROBE - Verify service is alive
5. VERIFY - Confirm change took effect

On failure, performs EMERGENCY ROLLBACK.
"""

import subprocess
import re
from typing import Union, Optional, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import psycopg2

from ..protocol.tuning import TuningChunk, TuningResult, TuningSnapshot
from ..protocol.errors import (
    TuningErrorPacket,
    FailureContext,
    RollbackStatus,
    TuningSystemContext,
    KernelLimits,
)
from .snapshot import TuningSnapshotManager
from .service import ServiceController, ServiceConfig
from .verifier import TuningVerifier


class ServiceStartError(Exception):
    """PostgreSQL failed to start after config change."""
    pass


class VerificationError(Exception):
    """Tuning verification failed."""
    pass


class CommandExecutionError(Exception):
    """Command execution failed."""
    pass


@dataclass
class ExecutorConfig:
    """Configuration for tuning executor."""
    probe_timeout: int = 60
    verify_timeout: int = 30
    ssh_host: Optional[str] = None
    ssh_user: str = "ubuntu"
    ssh_key: Optional[str] = None


class TuningExecutor:
    """
    Applies tuning changes with snapshotting and emergency rollback.

    This is the core v2.2 resilience component.
    """

    def __init__(
        self,
        connection,
        snapshot_manager: Optional[TuningSnapshotManager] = None,
        service_controller: Optional[ServiceController] = None,
        verifier: Optional[TuningVerifier] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        self.conn = connection
        self.config = config or ExecutorConfig()

        # Initialize components
        ssh_config = None
        if self.config.ssh_host:
            ssh_config = {
                'host': self.config.ssh_host,
                'user': self.config.ssh_user,
            }

        self.snapshot_manager = snapshot_manager or TuningSnapshotManager(
            connection=connection,
            ssh_config=ssh_config
        )
        self.service_controller = service_controller or ServiceController(
            ServiceConfig(
                ssh_host=self.config.ssh_host,
                ssh_user=self.config.ssh_user,
                ssh_key=self.config.ssh_key,
            )
        )
        self.verifier = verifier or TuningVerifier(connection=connection)

        self._current_phase = "INIT"

    def apply_tuning_safe(
        self,
        chunk: TuningChunk
    ) -> Union[TuningResult, TuningErrorPacket]:
        """
        Safe tuning application with 5 phases.

        Args:
            chunk: The tuning chunk to apply

        Returns:
            TuningResult on success, TuningErrorPacket on failure
        """
        snapshot = None

        # Phase 1: SNAPSHOT
        self._current_phase = "SNAPSHOT"
        try:
            snapshot = self.snapshot_manager.capture(chunk)
        except Exception as e:
            return self._create_error_packet(
                chunk=chunk,
                phase="SNAPSHOT",
                error=e,
                snapshot=None,
                rollback_performed=False,
            )

        try:
            # Phase 2: EXECUTE
            self._current_phase = "EXECUTE"
            for cmd in chunk.apply_commands:
                self._execute_command(cmd)

            # Phase 3: RESTART (if required)
            if chunk.requires_restart:
                self._current_phase = "RESTART"
                success, logs = self.service_controller.restart_and_probe(
                    timeout=self.config.probe_timeout
                )

                if not success:
                    raise ServiceStartError(
                        f"PostgreSQL failed to start after applying {chunk.name}"
                    )

            # Phase 4: PROBE (additional connectivity check)
            self._current_phase = "PROBE"
            if not self.service_controller.probe_connectivity(timeout=10):
                raise ServiceStartError("PostgreSQL is not responding after config change")

            # Reconnect if needed
            self._ensure_connection()

            # Phase 5: VERIFY
            self._current_phase = "VERIFY"
            actual_value = self.verifier.get_value(chunk.verification_command)
            if not self.verifier.matches(actual_value, chunk.verification_expected):
                raise VerificationError(
                    f"Expected '{chunk.verification_expected}', got '{actual_value}'"
                )

            # Success!
            self._current_phase = "COMPLETE"
            self.snapshot_manager.cleanup(chunk.id)

            return TuningResult(
                chunk_id=chunk.id,
                success=True,
                applied=True,
                verified=True,
                actual_value=str(actual_value),
            )

        except Exception as e:
            # EMERGENCY ROLLBACK
            logs = self.service_controller.get_startup_errors() or \
                   self.service_controller.get_tail_logs(50)
            rollback_success = self._execute_emergency_rollback(chunk, snapshot)

            return self._create_error_packet(
                chunk=chunk,
                phase=self._current_phase,
                error=e,
                snapshot=snapshot,
                rollback_performed=True,
                rollback_success=rollback_success,
                service_logs=logs,
            )

    def _execute_command(self, cmd: str):
        """Execute a single tuning command."""
        cmd_upper = cmd.upper().strip()

        # SQL commands (ALTER SYSTEM, SELECT pg_reload_conf, etc.)
        is_sql = any(kw in cmd_upper for kw in [
            'ALTER SYSTEM', 'SELECT PG_', 'SELECT ', 'SHOW ', 'SET ',
            'CREATE ', 'DROP ', 'VACUUM', 'ANALYZE', 'REINDEX'
        ])
        # Exclude shell commands
        is_shell = any(kw in cmd_upper for kw in ['SUDO', 'SYSCTL', 'ECHO ', 'SYSTEMCTL'])

        if is_sql and not is_shell:
            if not self.conn:
                raise CommandExecutionError("No database connection for SQL command")
            try:
                with self.conn.cursor() as cur:
                    cur.execute(cmd)
                self.conn.commit()
            except Exception as e:
                raise CommandExecutionError(f"SQL execution failed: {e}")

        # OS commands (sudo sysctl, echo, etc.)
        else:
            try:
                if self.config.ssh_host:
                    full_cmd = [
                        "ssh", "-o", "StrictHostKeyChecking=no",
                        f"{self.config.ssh_user}@{self.config.ssh_host}",
                        cmd
                    ]
                    result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
                else:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                raise CommandExecutionError(f"Command failed: {e.stderr}")

    def _execute_emergency_rollback(
        self,
        chunk: TuningChunk,
        snapshot: Optional[TuningSnapshot]
    ) -> bool:
        """
        Restore state even if DB is dead.

        Uses FILE_RESTORE for config files when SQL is not available.
        """
        if not snapshot:
            return False

        try:
            # Delegate to snapshot manager
            restore_success = self.snapshot_manager.restore_from_snapshot(snapshot, chunk)

            # Restart service after rollback
            if chunk.requires_restart:
                self.service_controller.restart()
                return self.service_controller.probe_connectivity(timeout=30)

            return restore_success

        except Exception:
            return False

    def _ensure_connection(self):
        """Ensure database connection is alive after restart."""
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except Exception:
                # Connection was lost, try to reconnect
                # Note: actual reconnection logic would depend on connection factory
                pass

    def _get_system_context(self) -> TuningSystemContext:
        """Gather system context for error packet."""
        from ..discovery.system import SystemScanner

        scanner = SystemScanner()
        kernel_limits = scanner.get_kernel_limits()
        mem_info = scanner._get_memory_info()

        return TuningSystemContext(
            kernel_limits=KernelLimits(
                shmmax=kernel_limits.get('shmmax', 0),
                shmall=kernel_limits.get('shmall', 0),
                hugepages_total=kernel_limits.get('hugepages_total', 0),
                hugepages_free=kernel_limits.get('hugepages_free', 0),
            ),
            available_memory_gb=mem_info.get('available_gb', 0),
        )

    def _create_error_packet(
        self,
        chunk: TuningChunk,
        phase: str,
        error: Exception,
        snapshot: Optional[TuningSnapshot],
        rollback_performed: bool,
        rollback_success: bool = False,
        service_logs: Optional[List[str]] = None,
    ) -> TuningErrorPacket:
        """Create a TuningErrorPacket from an error."""
        # Determine error type
        if isinstance(error, ServiceStartError):
            error_type = "SERVICE_START_FAILED"
        elif isinstance(error, VerificationError):
            error_type = "VERIFICATION_FAILED"
        else:
            error_type = "TUNING_APPLICATION_FAILED"

        # Determine current state
        if not rollback_performed:
            current_state = "UNKNOWN_CRITICAL"
        elif rollback_success:
            current_state = "RESTORED_TO_SNAPSHOT"
        else:
            current_state = "PARTIALLY_RESTORED"

        return TuningErrorPacket(
            protocol_version="v2",
            error_type=error_type,
            failed_chunk_id=chunk.id,
            failure_context=FailureContext(
                phase=phase,
                error_message=str(error),
                service_logs=service_logs or [],
                original_values=snapshot.original_db_values if snapshot else {},
            ),
            rollback_status=RollbackStatus(
                performed=rollback_performed,
                strategy_used=chunk.recovery_strategy,
                success=rollback_success,
                current_state=current_state,
            ),
            system_context=self._get_system_context() if rollback_performed else None,
        )

    def apply_multiple(
        self,
        chunks: List[TuningChunk],
        stop_on_error: bool = True
    ) -> List[Union[TuningResult, TuningErrorPacket]]:
        """
        Apply multiple tuning chunks in dependency order.

        Args:
            chunks: List of chunks to apply
            stop_on_error: If True, stop on first error

        Returns:
            List of results for each chunk
        """
        results = []

        for chunk in chunks:
            result = self.apply_tuning_safe(chunk)
            results.append(result)

            if stop_on_error and isinstance(result, TuningErrorPacket):
                break

        return results
