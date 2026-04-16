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
    
    category: str = Field(..., description="Category name (e.g., Cybersecurity-Hacker Attack, Data Privacy-Personal Privacy)")
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
    """Threshold configuration for a policy.
    
    Uses fine-grained category names directly from model output (YuFeng-XGuard-Reason ID2RISK).
    Each category can have its own independent threshold for precise risk control.
    
    Example:
        thresholds = PolicyThresholds(
            thresholds={
                "Cybersecurity-Hacker Attack": 0.5,
                "Data Privacy-Personal Privacy": 0.8,
                "Hate Speech-Abusive Curses": 0.4,
            }
        )
    """
    
    # All fine-grained thresholds stored in a single dictionary
    # Keys are model output category names (e.g., "Cybersecurity-Hacker Attack")
    # Values are threshold scores (0.0 to 1.0)
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            # Safe
            "Safe-Safe": 0.7,
            # Crimes and Illegal Activities
            "Crimes and Illegal Activities-Pornographic Contraband": 0.7,
            "Crimes and Illegal Activities-Drug Crimes": 0.7,
            "Crimes and Illegal Activities-Dangerous Weapons": 0.7,
            "Crimes and Illegal Activities-Property Infringement": 0.7,
            "Crimes and Illegal Activities-Economic Crimes": 0.7,
            # Hate Speech
            "Hate Speech-Abusive Curses": 0.6,
            "Hate Speech-Defamation": 0.6,
            "Hate Speech-Threats and Intimidation": 0.6,
            "Hate Speech-Cyberbullying": 0.6,
            # Physical and Mental Health
            "Physical and Mental Health-Physical Health": 0.75,
            "Physical and Mental Health-Mental Health": 0.75,
            # Ethics and Morality
            "Ethics and Morality-Social Ethics": 0.75,
            "Ethics and Morality-Science Ethics": 0.75,
            # Data Privacy
            "Data Privacy-Personal Privacy": 0.8,
            "Data Privacy-Commercial Secret": 0.8,
            # Cybersecurity
            "Cybersecurity-Access Control": 0.7,
            "Cybersecurity-Malicious Code": 0.7,
            "Cybersecurity-Hacker Attack": 0.7,
            "Cybersecurity-Physical Security": 0.7,
            # Extremism
            "Extremism-Violent Terrorist Activities": 0.6,
            "Extremism-Social Disruption": 0.6,
            "Extremism-Extremist Ideological Trends": 0.6,
            # Inappropriate Suggestions
            "Inappropriate Suggestions-Finance": 0.75,
            "Inappropriate Suggestions-Medicine": 0.75,
            "Inappropriate Suggestions-Law": 0.75,
            # Risks Involving Minors
            "Risks Involving Minors-Corruption of Minors": 0.6,
            "Risks Involving Minors-Minor Abuse and Exploitation": 0.6,
            "Risks Involving Minors-Minor Delinquency": 0.6,
        },
        description="Fine-grained category thresholds. Keys are model output category names."
    )
    
    def get_threshold(self, category: str) -> float:
        """Get threshold for a specific fine-grained category.
        
        Args:
            category: Model output category name (e.g., "Cybersecurity-Hacker Attack")
            
        Returns:
            Threshold value for the category, or 0.7 as default fallback
        """
        return self.thresholds.get(category, 0.7)
    
    def set_threshold(self, category: str, value: float) -> None:
        """Set threshold for a specific category.
        
        Args:
            category: Model output category name
            value: Threshold value (0.0 to 1.0)
        """
        self.thresholds[category] = value


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


class PolicyActionResult(BaseModel):
    """Result of policy evaluation, including action and triggered risk categories."""
    
    action: Action = Field(..., description="Action to take")
    triggered_categories: List[DetectionCategory] = Field(
        default_factory=list,
        description="Categories that exceeded thresholds and triggered the action",
    )
    
    @property
    def risk_summary(self) -> str:
        """Human-readable summary of triggered risk categories."""
        if not self.triggered_categories:
            return ""
        parts = []
        for cat in self.triggered_categories:
            parts.append(f"{cat.category} (score={cat.score:.2f}, level={cat.level.value})")
        return "; ".join(parts)


class XGuardSafetyError(Exception):
    """Exception raised when content is blocked by XGuard safety policy.
    
    Attributes:
        action: The policy action that was triggered (BLOCK, MASK, etc.)
        detection_result: The full detection result from the model
        triggered_categories: The risk categories that exceeded thresholds
        message: Human-readable error message with risk type details
    """
    
    def __init__(
        self,
        action: Action,
        detection_result: DetectionResult,
        triggered_categories: List[DetectionCategory],
        fallback_message: str = "",
    ):
        self.action = action
        self.detection_result = detection_result
        self.triggered_categories = triggered_categories
        
        # Build informative message
        risk_types = []
        for cat in triggered_categories:
            risk_types.append(f"[{cat.category}] score={cat.score:.2f}, level={cat.level.value}")
        
        risk_detail = "\n  ".join(risk_types) if risk_types else "Unknown"
        self.message = (
            f"{fallback_message or 'Content blocked by safety policy.'}\n"
            f"Risk type(s) detected:\n  {risk_detail}"
        )
        super().__init__(self.message)


class StreamChunk(BaseModel):
    """Represents a chunk in streaming mode."""
    
    content: str = Field(..., description="Chunk content")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
