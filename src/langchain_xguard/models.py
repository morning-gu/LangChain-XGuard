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
    
    # Mapping from model output category names (YuFeng-XGuard-Reason ID2RISK values)
    # to policy threshold field names. This bridges the gap between the 25 fine-grained
    # model categories and the 4 coarse-grained policy threshold fields.
    CATEGORY_TO_THRESHOLD: Dict[str, str] = {
        # Safe
        "Safe-Safe": "jailbreak",  # safe content uses lowest sensitivity threshold
        # Crimes and Illegal Activities -> jailbreak
        "Crimes and Illegal Activities-Pornographic Contraband": "jailbreak",
        "Crimes and Illegal Activities-Drug Crimes": "jailbreak",
        "Crimes and Illegal Activities-Dangerous Weapons": "jailbreak",
        "Crimes and Illegal Activities-Property Infringement": "jailbreak",
        "Crimes and Illegal Activities-Economic Crimes": "jailbreak",
        # Hate Speech -> toxicity
        "Hate Speech-Abusive Curses": "toxicity",
        "Hate Speech-Defamation": "toxicity",
        "Hate Speech-Threats and Intimidation": "toxicity",
        "Hate Speech-Cyberbullying": "toxicity",
        # Physical and Mental Health -> compliance
        "Physical and Mental Health-Physical Health": "compliance",
        "Physical and Mental Health-Mental Health": "compliance",
        # Ethics and Morality -> compliance
        "Ethics and Morality-Social Ethics": "compliance",
        "Ethics and Morality-Science Ethics": "compliance",
        # Data Privacy -> pii
        "Data Privacy-Personal Privacy": "pii",
        "Data Privacy-Commercial Secret": "pii",
        # Cybersecurity -> jailbreak
        "Cybersecurity-Access Control": "jailbreak",
        "Cybersecurity-Malicious Code": "jailbreak",
        "Cybersecurity-Hacker Attack": "jailbreak",
        "Cybersecurity-Physical Security": "jailbreak",
        # Extremism -> toxicity
        "Extremism-Violent Terrorist Activities": "toxicity",
        "Extremism-Social Disruption": "toxicity",
        "Extremism-Extremist Ideological Trends": "toxicity",
        # Inappropriate Suggestions -> compliance
        "Inappropriate Suggestions-Finance": "compliance",
        "Inappropriate Suggestions-Medicine": "compliance",
        "Inappropriate Suggestions-Law": "compliance",
        # Risks Involving Minors -> toxicity
        "Risks Involving Minors-Corruption of Minors": "toxicity",
        "Risks Involving Minors-Minor Abuse and Exploitation": "toxicity",
        "Risks Involving Minors-Minor Delinquency": "toxicity",
    }
    
    def get_threshold(self, category: str) -> float:
        """Get threshold for a specific category.
        
        Maps model output category names (e.g., "Cybersecurity-Hacker Attack")
        to the corresponding policy threshold field (e.g., "jailbreak").
        """
        # First, try direct attribute match (for simple names like "jailbreak")
        if hasattr(self, category):
            return getattr(self, category)
        # Then, try mapping from model category name to threshold field
        threshold_field = self.CATEGORY_TO_THRESHOLD.get(category)
        if threshold_field and hasattr(self, threshold_field):
            return getattr(self, threshold_field)
        # Finally, check custom thresholds or use default
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
