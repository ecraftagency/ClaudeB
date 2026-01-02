"""
SessionConclusion Protocol - v2.3 Capacity Aware & Human-in-the-Loop

This protocol is triggered when the AI determines that software tuning
can no longer improve performance due to hardware limitations.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class BottleneckResource(str, Enum):
    """Hardware resource that is saturated."""
    CPU = "CPU"
    IO_THROUGHPUT = "IO_THROUGHPUT"
    IOPS = "IOPS"
    MEMORY = "MEMORY"
    NETWORK = "NETWORK"
    NONE = "NONE"


class ScalingAction(str, Enum):
    """Recommended scaling action."""
    SCALE_UP = "SCALE_UP"
    SCALE_OUT = "SCALE_OUT"
    OPTIMIZE_APP = "OPTIMIZE_APP"
    NONE_NEEDED = "NONE_NEEDED"


@dataclass
class TuningSummary:
    """Summary of tuning iterations performed."""
    total_iterations: int = 0
    baseline_tps: float = 0.0
    final_tps: float = 0.0
    improvement_pct: float = 0.0
    key_changes_applied: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.baseline_tps > 0 and self.final_tps > 0:
            self.improvement_pct = ((self.final_tps - self.baseline_tps) / self.baseline_tps) * 100


@dataclass
class ScalingRecommendation:
    """Capacity planning recommendation."""
    action: str = "NONE_NEEDED"  # SCALE_UP, SCALE_OUT, OPTIMIZE_APP, NONE_NEEDED
    details: str = ""


@dataclass
class HardwareSaturationAnalysis:
    """Analysis of hardware saturation ('The Verdict')."""
    is_saturated: bool = False
    bottleneck_resource: str = "NONE"  # CPU, IO_THROUGHPUT, IOPS, MEMORY, NETWORK, NONE
    evidence: List[str] = field(default_factory=list)
    scaling_recommendation: Optional[ScalingRecommendation] = None

    def __post_init__(self):
        if self.scaling_recommendation is None:
            self.scaling_recommendation = ScalingRecommendation()


@dataclass
class SessionConclusion:
    """
    Session Conclusion protocol - triggered when AI determines tuning is complete.

    This can happen due to:
    1. Hardware saturation (no more software tuning possible)
    2. Success criteria met
    3. Diminishing returns (< 2% improvement over last 2 iterations)
    """
    protocol_version: str = "v2"
    response_type: str = "conclude_session"
    session_id: str = ""
    concluded_at: str = ""

    # Tuning efficacy report
    tuning_summary: Optional[TuningSummary] = None

    # Hardware Saturation Analysis ("The Verdict")
    hardware_saturation_analysis: Optional[HardwareSaturationAnalysis] = None

    # Final executive summary in markdown
    final_report_markdown: str = ""

    # Conclusion reason
    conclusion_reason: str = ""  # SATURATION, SUCCESS, DIMINISHING_RETURNS, MAX_ITERATIONS

    def __post_init__(self):
        if not self.concluded_at:
            self.concluded_at = datetime.utcnow().isoformat() + "Z"
        if self.tuning_summary is None:
            self.tuning_summary = TuningSummary()
        if self.hardware_saturation_analysis is None:
            self.hardware_saturation_analysis = HardwareSaturationAnalysis()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "protocol_version": self.protocol_version,
            "response_type": self.response_type,
            "session_id": self.session_id,
            "concluded_at": self.concluded_at,
            "conclusion_reason": self.conclusion_reason,
            "final_report_markdown": self.final_report_markdown,
        }
        if self.tuning_summary:
            result["tuning_summary"] = asdict(self.tuning_summary)
        if self.hardware_saturation_analysis:
            hsa = asdict(self.hardware_saturation_analysis)
            result["hardware_saturation_analysis"] = hsa
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionConclusion":
        """Create from dictionary (parsed from AI response)."""
        if "tuning_summary" in data and data["tuning_summary"]:
            data["tuning_summary"] = TuningSummary(**data["tuning_summary"])

        if "hardware_saturation_analysis" in data and data["hardware_saturation_analysis"]:
            hsa = data["hardware_saturation_analysis"]
            if "scaling_recommendation" in hsa and hsa["scaling_recommendation"]:
                hsa["scaling_recommendation"] = ScalingRecommendation(**hsa["scaling_recommendation"])
            data["hardware_saturation_analysis"] = HardwareSaturationAnalysis(**hsa)

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def is_hardware_limited(self) -> bool:
        """Check if session concluded due to hardware saturation."""
        if self.hardware_saturation_analysis:
            return self.hardware_saturation_analysis.is_saturated
        return False

    def get_scaling_advice(self) -> str:
        """Get human-readable scaling advice."""
        if not self.hardware_saturation_analysis:
            return "No scaling analysis available."

        hsa = self.hardware_saturation_analysis
        if not hsa.is_saturated:
            return "System is not saturated. Further tuning may be possible."

        rec = hsa.scaling_recommendation
        if rec:
            return f"Recommendation: {rec.action} - {rec.details}"
        return f"Bottleneck: {hsa.bottleneck_resource}"
