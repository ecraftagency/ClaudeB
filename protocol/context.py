"""
ContextPacket - The input signal from Runner to Agent.

Contains deep discovery information about the database system.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class TableStats:
    """Statistics for a single table."""
    schema: str
    rows: int
    size_mb: float
    bloat_ratio: float
    index_count: int
    fk_inbound: int
    fk_outbound: int
    last_vacuum: Optional[str]
    last_analyze: Optional[str]
    n_tup_ins: int = 0
    n_tup_upd: int = 0
    n_tup_del: int = 0
    n_dead_tup: int = 0


@dataclass
class IndexStats:
    """Statistics for a single index."""
    table_name: str
    index_type: str  # btree, hash, gin, gist, brin
    size_mb: float
    idx_scan: int
    idx_tup_read: int
    idx_tup_fetch: int = 0
    columns: List[str] = field(default_factory=list)
    is_unique: bool = False
    is_primary: bool = False
    definition: str = ""


@dataclass
class ForeignKeyInfo:
    """Detailed foreign key information."""
    constraint_name: str
    from_table: str
    from_columns: List[str]
    to_table: str
    to_columns: List[str]
    on_delete: str = "NO ACTION"
    on_update: str = "NO ACTION"


@dataclass
class TriggerInfo:
    """Trigger definition."""
    name: str
    table_name: str
    event: str  # INSERT, UPDATE, DELETE, TRUNCATE
    timing: str  # BEFORE, AFTER, INSTEAD OF
    function_name: str
    enabled: bool = True
    definition: str = ""


@dataclass
class FunctionInfo:
    """Stored procedure/function information."""
    name: str
    schema: str
    language: str  # plpgsql, sql, plpython, etc.
    return_type: str
    argument_types: List[str] = field(default_factory=list)
    is_trigger_func: bool = False
    volatility: str = "VOLATILE"  # VOLATILE, STABLE, IMMUTABLE
    definition: str = ""


@dataclass
class ExtensionInfo:
    """Installed PostgreSQL extension."""
    name: str
    version: str
    schema: str = "public"


@dataclass
class HeuristicHints:
    """Inferred patterns for AI guidance."""
    has_version_column: bool = False
    has_audit_tables: bool = False
    high_write_volume_tables: List[str] = field(default_factory=list)
    hot_update_tables: List[str] = field(default_factory=list)
    unused_indexes: List[str] = field(default_factory=list)
    missing_pk_tables: List[str] = field(default_factory=list)
    wide_tables: List[str] = field(default_factory=list)
    partitioned_tables: List[str] = field(default_factory=list)


@dataclass
class SystemContext:
    """System hardware and OS information."""
    cpu_architecture: str  # x86_64, aarch64
    cpu_cores: int
    cpu_model: str
    ram_total_gb: float
    storage_topology: str  # single_disk, raid0, raid1, raid10, nvme_raid10
    storage_devices: List[Dict[str, Any]]
    os_version: str
    kernel_version: str
    pg_version: int
    pg_version_full: str


@dataclass
class RuntimeContext:
    """Current PostgreSQL and OS runtime configuration."""
    active_config: Dict[str, str]
    os_tuning: Dict[str, str]
    load_average: List[float]  # 1m, 5m, 15m
    memory_usage: Dict[str, float]  # used_gb, cached_gb, available_gb


@dataclass
class SchemaSummary:
    """Summary for large schemas (when table count > 50)."""
    total_tables: int
    detailed_tables: int
    omitted_tables: int
    omitted_total_size_gb: float
    omitted_total_rows: int
    omitted_categories: Dict[str, int] = field(default_factory=dict)


@dataclass
class SchemaContext:
    """Database schema information."""
    database_name: str
    database_size_gb: float
    ddl_summary: List[str]
    table_statistics: Dict[str, TableStats]
    index_statistics: Dict[str, IndexStats]
    heuristic_hints: HeuristicHints
    schema_summary: Optional[SchemaSummary] = None
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)
    triggers: List[TriggerInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    extensions: List[ExtensionInfo] = field(default_factory=list)


@dataclass
class PreviousIteration:
    """Results from the previous iteration for comparison."""
    strategy_id: str
    primary_kpi_value: float
    primary_kpi_name: str
    applied_tuning_ids: List[str]


@dataclass
class ContextPacket:
    """
    The complete context packet sent from Runner to Agent.

    This is the input signal that enables contextual reasoning.
    """
    protocol_version: str = "v2"
    timestamp: str = ""
    session_id: str = ""

    system_context: Optional[SystemContext] = None
    runtime_context: Optional[RuntimeContext] = None
    schema_context: Optional[SchemaContext] = None

    previous_iteration: Optional[PreviousIteration] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def convert(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items() if v is not None}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        return convert(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextPacket":
        """Create from dictionary."""
        # Handle nested dataclasses
        if "system_context" in data and data["system_context"]:
            data["system_context"] = SystemContext(**data["system_context"])
        if "runtime_context" in data and data["runtime_context"]:
            data["runtime_context"] = RuntimeContext(**data["runtime_context"])
        if "schema_context" in data and data["schema_context"]:
            sc = data["schema_context"]
            # Convert nested dicts to dataclasses
            if "table_statistics" in sc:
                sc["table_statistics"] = {
                    k: TableStats(**v) for k, v in sc["table_statistics"].items()
                }
            if "index_statistics" in sc:
                sc["index_statistics"] = {
                    k: IndexStats(**v) for k, v in sc["index_statistics"].items()
                }
            if "heuristic_hints" in sc:
                sc["heuristic_hints"] = HeuristicHints(**sc["heuristic_hints"])
            if "schema_summary" in sc and sc["schema_summary"]:
                sc["schema_summary"] = SchemaSummary(**sc["schema_summary"])
            data["schema_context"] = SchemaContext(**sc)
        if "previous_iteration" in data and data["previous_iteration"]:
            data["previous_iteration"] = PreviousIteration(**data["previous_iteration"])
        return cls(**data)
