"""
FirstSight Protocol - Initial system analysis before benchmarking.

This is the AI's first look at the system to provide:
- Brief system/schema overview
- Multiple benchmark strategy suggestions for user selection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json


@dataclass
class StrategyOption:
    """A suggested benchmark strategy for user selection."""
    id: str
    name: str
    goal: str
    hypothesis: str
    target_kpis: Dict[str, Any]
    rationale: str
    estimated_duration_minutes: int = 5
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH


@dataclass
class FirstSightResponse:
    """
    AI's first analysis of the system before any benchmarking.

    Provides concise overview and strategy options for user to choose.
    """
    protocol_version: str = "v2"

    # Brief overviews (markdown formatted, concise)
    system_overview: str = ""
    schema_overview: str = ""

    # Key observations the AI noticed
    key_observations: List[str] = field(default_factory=list)

    # Potential concerns or warnings
    warnings: List[str] = field(default_factory=list)

    # Strategy options for user to choose
    strategy_options: List[StrategyOption] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "protocol_version": self.protocol_version,
            "system_overview": self.system_overview,
            "schema_overview": self.schema_overview,
            "key_observations": self.key_observations,
            "warnings": self.warnings,
            "strategy_options": [
                {
                    "id": s.id,
                    "name": s.name,
                    "goal": s.goal,
                    "hypothesis": s.hypothesis,
                    "target_kpis": s.target_kpis,
                    "rationale": s.rationale,
                    "estimated_duration_minutes": s.estimated_duration_minutes,
                    "risk_level": s.risk_level,
                }
                for s in self.strategy_options
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FirstSightResponse":
        """Create from dictionary."""
        strategy_options = []
        for s in data.get("strategy_options", []):
            strategy_options.append(StrategyOption(
                id=s.get("id", ""),
                name=s.get("name", ""),
                goal=s.get("goal", ""),
                hypothesis=s.get("hypothesis", ""),
                target_kpis=s.get("target_kpis", {}),
                rationale=s.get("rationale", ""),
                estimated_duration_minutes=s.get("estimated_duration_minutes", 5),
                risk_level=s.get("risk_level", "LOW"),
            ))

        return cls(
            protocol_version=data.get("protocol_version", "v2"),
            system_overview=data.get("system_overview", ""),
            schema_overview=data.get("schema_overview", ""),
            key_observations=data.get("key_observations", []),
            warnings=data.get("warnings", []),
            strategy_options=strategy_options,
        )
