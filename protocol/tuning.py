"""
Tuning Protocol - Agent → Runner for configuration changes.

Includes v2.2 Resilience features: recovery_strategy, target_config_file.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class RecoveryStrategy(str, Enum):
    """How to recover if tuning fails."""
    SQL_REVERT = "SQL_REVERT"      # DB is alive, use ALTER SYSTEM RESET
    FILE_RESTORE = "FILE_RESTORE"  # DB dead, restore config file from backup
    OS_REVERT = "OS_REVERT"        # OS-level sysctl rollback


class RiskLevel(str, Enum):
    """Risk level of a tuning chunk."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Priority(str, Enum):
    """Priority of a tuning chunk."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ChunkCategory(str, Enum):
    """Category of tuning chunk."""
    OS = "OS"
    DATABASE = "DATABASE"
    DISK = "DISK"
    NETWORK = "NETWORK"
    ARCHITECTURE = "ARCHITECTURE"


@dataclass
class Diagnosis:
    """AI's diagnosis of performance issues."""
    summary: str
    root_cause: str
    bottleneck_category: str  # CPU, IO, MEMORY, LOCKING, NETWORK, CONFIG
    confidence_score: float   # 0.0 - 1.0
    observations: List[str] = field(default_factory=list)
    positive_notes: List[str] = field(default_factory=list)
    remediation_approach: Optional[str] = None


@dataclass
class TuningChunk:
    """
    A single atomic tuning change with v2.2 resilience features.
    """
    id: str = ""
    category: str = ""  # OS, DATABASE, DISK, NETWORK, ARCHITECTURE
    name: str = ""
    description: str = ""
    rationale: str = ""  # Why this change is recommended
    purpose: str = ""
    expected_impact: str = ""

    # Execution commands
    apply_commands: List[str] = field(default_factory=list)

    # Logical rollback (SQL/Command level)
    rollback_commands: List[str] = field(default_factory=list)

    # Verification
    verification_command: str = ""
    verification_expected: str = ""

    # === RESILIENCE & RECOVERY (v2.2) ===
    requires_restart: bool = False
    recovery_strategy: str = "SQL_REVERT"  # SQL_REVERT, FILE_RESTORE, OS_REVERT
    target_config_file: Optional[str] = None  # For FILE_RESTORE

    # Safety metadata
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH
    priority: str = "MEDIUM"  # HIGH, MEDIUM, LOW

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None and v != []}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TuningChunk":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class NextAction:
    """Recommended next action after analysis."""
    type: str  # apply_tuning, run_new_strategy, conclude_session
    rationale: str
    new_strategy: Optional[Dict[str, Any]] = None


@dataclass
class TuningProposal:
    """
    Complete tuning proposal from Agent to Runner.
    """
    protocol_version: str = "v2"
    strategy_id: str = ""
    session_id: str = ""
    generated_at: str = ""
    response_type: str = "tuning_proposal"  # or "remedial_tuning"

    # Analysis fields (used by parser)
    analysis_summary: str = ""
    bottleneck_type: str = ""
    confidence: float = 0.0

    diagnosis: Optional[Diagnosis] = None
    tuning_chunks: List[TuningChunk] = field(default_factory=list)
    next_action: Optional[NextAction] = None

    # Expected results
    expected_improvement: Optional["ExpectedImprovement"] = None
    verification_benchmark: Optional["VerificationBenchmark"] = None

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "protocol_version": self.protocol_version,
            "strategy_id": self.strategy_id,
            "session_id": self.session_id,
            "generated_at": self.generated_at,
            "response_type": self.response_type,
        }
        if self.diagnosis:
            result["diagnosis"] = asdict(self.diagnosis)
        if self.tuning_chunks:
            result["tuning_chunks"] = [c.to_dict() for c in self.tuning_chunks]
        if self.next_action:
            result["next_action"] = asdict(self.next_action)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TuningProposal":
        """Create from dictionary (parsed from AI response)."""
        if "diagnosis" in data and data["diagnosis"]:
            data["diagnosis"] = Diagnosis(**data["diagnosis"])
        if "tuning_chunks" in data:
            data["tuning_chunks"] = [
                TuningChunk.from_dict(c) for c in data["tuning_chunks"]
            ]
        if "next_action" in data and data["next_action"]:
            data["next_action"] = NextAction(**data["next_action"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_ordered_chunks(self) -> List[TuningChunk]:
        """Get chunks in dependency order."""
        # Build dependency graph
        chunks_by_id = {c.id: c for c in self.tuning_chunks}
        ordered = []
        visited = set()

        def visit(chunk_id: str):
            if chunk_id in visited:
                return
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                return
            for dep_id in chunk.depends_on:
                visit(dep_id)
            visited.add(chunk_id)
            ordered.append(chunk)

        for chunk in self.tuning_chunks:
            visit(chunk.id)

        return ordered


@dataclass
class ExpectedImprovement:
    """Expected improvement from tuning."""
    tps_increase_pct: float = 0.0
    latency_reduction_pct: float = 0.0


@dataclass
class VerificationBenchmark:
    """Parameters for verification benchmark after tuning."""
    duration_seconds: int = 30
    clients: int = 10


@dataclass
class TuningResult:
    """Result of applying a single tuning chunk."""
    chunk_id: str = ""
    success: bool = False
    applied: bool = False
    verified: bool = False
    actual_value: str = ""
    error_message: str = ""
    reason: str = ""


@dataclass
class TuningSnapshot:
    """
    Captures system state before a TuningChunk is applied.

    Used for emergency rollback in v2.2.
    """
    chunk_id: str
    timestamp: datetime

    # Original DB param values (SHOW [param])
    original_db_values: Dict[str, str] = field(default_factory=dict)

    # Original OS param values (sysctl -n [param])
    original_os_values: Dict[str, str] = field(default_factory=dict)

    # Path to config file backup (for FILE_RESTORE)
    config_file_backup_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "timestamp": self.timestamp.isoformat(),
            "original_db_values": self.original_db_values,
            "original_os_values": self.original_os_values,
            "config_file_backup_path": self.config_file_backup_path,
        }
