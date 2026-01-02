"""
Error Protocols - Runner → Agent for error recovery.

ErrorPacket: For SDL validation/execution errors
TuningErrorPacket: For tuning application failures (v2.2)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class ErrorType(str, Enum):
    """Types of errors that can occur."""
    SDL_VALIDATION = "SDL_VALIDATION"
    SQL_EXECUTION = "SQL_EXECUTION"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    TIMEOUT = "TIMEOUT"
    CONNECTION = "CONNECTION"


class TuningErrorType(str, Enum):
    """Types of tuning errors."""
    TUNING_APPLICATION_FAILED = "TUNING_APPLICATION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    SERVICE_START_FAILED = "SERVICE_START_FAILED"


class Phase(str, Enum):
    """Phases where errors can occur."""
    SETUP = "SETUP"
    BENCHMARK = "BENCHMARK"
    TEARDOWN = "TEARDOWN"
    TUNING = "TUNING"
    SNAPSHOT = "SNAPSHOT"
    EXECUTE = "EXECUTE"
    RESTART = "RESTART"
    PROBE = "PROBE"
    VERIFY = "VERIFY"


class RollbackState(str, Enum):
    """Current state after rollback attempt."""
    RESTORED_TO_SNAPSHOT = "RESTORED_TO_SNAPSHOT"
    PARTIALLY_RESTORED = "PARTIALLY_RESTORED"
    UNKNOWN_CRITICAL = "UNKNOWN_CRITICAL"


@dataclass
class ErrorDetails:
    """Details about an error."""
    message: str
    code: Optional[str] = None       # PostgreSQL error code
    hint: Optional[str] = None       # PostgreSQL hint
    failed_sql: Optional[str] = None
    validation_violations: List[str] = field(default_factory=list)


@dataclass
class AttemptedAction:
    """What the Runner tried to do."""
    type: str  # execute_sql, validate_sdl, run_pgbench, apply_tuning
    input: Any = None


@dataclass
class PreviousAttempt:
    """Record of a previous repair attempt."""
    fix_applied: str
    still_failed: bool


@dataclass
class RepairContext:
    """Context for repair attempts."""
    attempt_number: int = 1
    previous_attempts: List[PreviousAttempt] = field(default_factory=list)


@dataclass
class ErrorContext:
    """Context about when/where error occurred."""
    strategy_id: str
    session_id: str
    occurred_at: str = ""
    phase: str = "BENCHMARK"

    def __post_init__(self):
        if not self.occurred_at:
            self.occurred_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class ErrorPacket:
    """
    Error packet for SDL validation/execution errors.

    Sent from Runner to Agent when benchmark fails.
    """
    protocol_version: str = "v2"
    error_type: str = "SQL_EXECUTION"

    context: Optional[ErrorContext] = None
    error_details: Optional[ErrorDetails] = None
    attempted_action: Optional[AttemptedAction] = None
    repair_context: Optional[RepairContext] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "protocol_version": self.protocol_version,
            "error_type": self.error_type,
        }
        if self.context:
            result["context"] = asdict(self.context)
        if self.error_details:
            result["error_details"] = {
                k: v for k, v in asdict(self.error_details).items()
                if v is not None and v != []
            }
        if self.attempted_action:
            result["attempted_action"] = asdict(self.attempted_action)
        if self.repair_context:
            result["repair_context"] = asdict(self.repair_context)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class FailureContext:
    """Context about a tuning failure."""
    phase: str  # SNAPSHOT, EXECUTE, RESTART, PROBE, VERIFY
    error_message: str
    service_logs: List[str] = field(default_factory=list)
    original_values: Dict[str, str] = field(default_factory=dict)


@dataclass
class KernelLimits:
    """Kernel memory limits for context."""
    shmmax: int = 0
    shmall: int = 0
    hugepages_total: int = 0
    hugepages_free: int = 0


@dataclass
class TuningSystemContext:
    """System context for tuning error recovery."""
    kernel_limits: Optional[KernelLimits] = None
    available_memory_gb: float = 0.0


@dataclass
class RollbackStatus:
    """Status of rollback attempt."""
    performed: bool = False
    strategy_used: str = ""  # SQL_REVERT, FILE_RESTORE, OS_REVERT
    success: bool = False
    current_state: str = "UNKNOWN_CRITICAL"


@dataclass
class TuningErrorPacket:
    """
    Error packet for tuning failures (v2.2).

    Sent from Runner to Agent when tuning causes system failure.
    """
    protocol_version: str = "v2"
    error_type: str = "SERVICE_START_FAILED"

    failed_chunk_id: str = ""
    failure_context: Optional[FailureContext] = None
    rollback_status: Optional[RollbackStatus] = None
    system_context: Optional[TuningSystemContext] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "protocol_version": self.protocol_version,
            "error_type": self.error_type,
            "failed_chunk_id": self.failed_chunk_id,
        }
        if self.failure_context:
            result["failure_context"] = {
                k: v for k, v in asdict(self.failure_context).items()
                if v is not None and v != []
            }
        if self.rollback_status:
            result["rollback_status"] = asdict(self.rollback_status)
        if self.system_context:
            sc = {}
            if self.system_context.kernel_limits:
                sc["kernel_limits"] = asdict(self.system_context.kernel_limits)
            sc["available_memory_gb"] = self.system_context.available_memory_gb
            result["system_context"] = sc
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TuningErrorPacket":
        """Create from dictionary."""
        if "failure_context" in data and data["failure_context"]:
            data["failure_context"] = FailureContext(**data["failure_context"])
        if "rollback_status" in data and data["rollback_status"]:
            data["rollback_status"] = RollbackStatus(**data["rollback_status"])
        if "system_context" in data and data["system_context"]:
            sc = data["system_context"]
            if "kernel_limits" in sc and sc["kernel_limits"]:
                sc["kernel_limits"] = KernelLimits(**sc["kernel_limits"])
            data["system_context"] = TuningSystemContext(**sc)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
