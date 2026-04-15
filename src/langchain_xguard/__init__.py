"""
langchain-xguard: LangChain/LlamaIndex native security middleware.

Provides streaming safety interception, policy-as-code, and seamless observability.
"""

from langchain_xguard.middleware import XGuardInputMiddleware, XGuardOutputMiddleware
from langchain_xguard.client import XGuardClient
from langchain_xguard.policy import PolicyConfig, PolicyEngine
from langchain_xguard.models import RiskLevel, DetectionResult, Action

__version__ = "0.1.0"
__author__ = "XGuard Team"
__all__ = [
    "XGuardInputMiddleware",
    "XGuardOutputMiddleware",
    "XGuardClient",
    "PolicyConfig",
    "PolicyEngine",
    "RiskLevel",
    "DetectionResult",
    "Action",
]
