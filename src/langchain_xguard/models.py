"""Data models for XGuard detection results and configuration."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RiskLevel(Enum):
    """Risk level enumeration for detection results."""
    
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def to_score(self) -> float:
        """Convert risk level to numeric score."""
        mapping = {
            self.SAFE: 0.0,
            self.LOW: 0.3,
            self.MEDIUM: 0.5,
            self.HIGH: 0.75,
            self.CRITICAL: 1.0,
        }
        return mapping[self]
    
    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Create RiskLevel from numeric score."""
        if score < 0.2:
            return cls.SAFE
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MEDIUM
        elif score < 0.8:
            return cls.HIGH
        else:
            return cls.CRITICAL


class Action(Enum):
    """Action types for policy enforcement."""
    
    BLOCK = "block"
    REWRITE = "rewrite"
    MASK = "mask"
    LOG_ONLY = "log_only"
    ALLOW = "allow"


class DetectionCategory(BaseModel):
    """Detection result for a single category."""
    
    category: str = Field(..., description="Category name (e.g., jailbreak, pii, toxicity)")
    score: float = Field(..., ge=0.0, le=1.0, description="Risk score")
    level: RiskLevel = Field(..., description="Risk level")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    matched_patterns: Optional[List[str]] = Field(default=None, description="Matched patterns if any")


class DetectionResult(BaseModel):
    """Complete detection result from XGuard API."""
    
    is_safe: bool = Field(..., description="Overall safety verdict")
    overall_level: RiskLevel = Field(..., description="Overall risk level")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    categories: List[DetectionCategory] = Field(default_factory=list, description="Per-category results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    session_id: Optional[str] = Field(default=None, description="Session ID for context-aware detection")
    timestamp: Optional[float] = Field(default=None, description="Unix timestamp of detection")
    
    def should_block(self, threshold: float = 0.7) -> bool:
        """Check if result should trigger blocking action."""
        return self.overall_score >= threshold
    
    def get_highest_risk_category(self) -> Optional[DetectionCategory]:
        """Get the category with highest risk score."""
        if not self.categories:
            return None
        return max(self.categories, key=lambda c: c.score)


class PolicyThresholds(BaseModel):
    """Threshold configuration for a policy."""
    
    jailbreak: float = Field(default=0.7, ge=0.0, le=1.0)
    pii: float = Field(default=0.8, ge=0.0, le=1.0)
    toxicity: float = Field(default=0.6, ge=0.0, le=1.0)
    compliance: float = Field(default=0.75, ge=0.0, le=1.0)
    custom: Dict[str, float] = Field(default_factory=dict)
    
    def get_threshold(self, category: str) -> float:
        """Get threshold for a specific category."""
        if hasattr(self, category):
            return getattr(self, category)
        return self.custom.get(category, 0.7)


class PolicyConfig(BaseModel):
    """Configuration for a single policy."""
    
    name: str = Field(..., description="Policy name")
    input_action: Action = Field(default=Action.BLOCK, description="Action for input detection")
    output_action: Action = Field(default=Action.MASK, description="Action for output detection")
    input_thresholds: PolicyThresholds = Field(default_factory=PolicyThresholds)
    output_thresholds: PolicyThresholds = Field(default_factory=PolicyThresholds)
    fallback_message: str = Field(
        default="Your request cannot be processed due to safety concerns.",
        description="Message to show when blocked"
    )
    fallback_llm: Optional[str] = Field(default=None, description="Fallback LLM for rewrite action")
    enable_context: bool = Field(default=True, description="Enable context-aware detection")
    context_window: int = Field(default=5, description="Number of turns to include in context")
    cache_enabled: bool = Field(default=True, description="Enable local caching")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")


class StreamChunk(BaseModel):
    """Represents a chunk in streaming mode."""
    
    content: str = Field(..., description="Chunk content")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
