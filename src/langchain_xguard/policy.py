"""Policy engine for loading and managing security policies."""

import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import yaml

from langchain_xguard.models import (
    PolicyConfig,
    Action,
    DetectionResult,
    PolicyActionResult,
)


class PolicyEngine:
    """
    Policy engine for managing security policies.
    
    Supports YAML/JSON policy configuration, hot reloading,
    A/B testing routes, and version rollback.
    """
    
    def __init__(
        self,
        policy_path: Optional[str] = None,
        auto_reload: bool = False,
        reload_interval: int = 60,
    ):
        """
        Initialize policy engine.
        
        Args:
            policy_path: Path to policy YAML/JSON file
            auto_reload: Enable automatic policy reloading
            reload_interval: Reload interval in seconds
        """
        self.policy_path = policy_path
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        self._policies: Dict[str, PolicyConfig] = {}
        self._default_policy: str = "default"
        self._version: str = "v0"
        self._last_loaded: Optional[datetime] = None
        self._reload_task: Optional[asyncio.Task] = None
        self._policy_history: List[tuple] = []  # (version, policies, timestamp)
        
        # Load initial policies if path provided
        if policy_path:
            self.load_policies(policy_path)
        
        # Start auto-reload if enabled
        if auto_reload and policy_path:
            self._start_auto_reload()
    
    def _start_auto_reload(self) -> None:
        """Start background task for auto-reloading policies."""
        async def reload_loop():
            while True:
                await asyncio.sleep(self.reload_interval)
                try:
                    if self.policy_path and os.path.exists(self.policy_path):
                        mtime = os.path.getmtime(self.policy_path)
                        if self._last_loaded is None or mtime > self._last_loaded.timestamp():
                            self.load_policies(self.policy_path)
                except Exception as e:
                    # Log error but don't crash
                    print(f"Policy reload failed: {e}")
        
        self._reload_task = asyncio.create_task(reload_loop())
    
    def load_policies(self, path: str) -> None:
        """
        Load policies from YAML or JSON file.
        
        Args:
            path: Path to policy file
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")
        
        with open(path_obj, "r", encoding="utf-8") as f:
            if path_obj.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif path_obj.suffix == ".json":
                import json
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported policy file format: {path_obj.suffix}")
        
        # Parse policies
        policies_data = data.get("policies", {})
        new_policies: Dict[str, PolicyConfig] = {}
        
        for name, config in policies_data.items():
            policy = self._parse_policy_config(name, config)
            new_policies[name] = policy
        
        # Save old policies for rollback
        if self._policies:
            self._policy_history.append((
                self._version,
                dict(self._policies),
                datetime.now(),
            ))
        
        # Update policies
        self._policies = new_policies
        self._version = f"v{len(self._policy_history) + 1}"
        self._last_loaded = datetime.now()
        
        # Set default if exists
        if "default" not in self._policies and self._policies:
            self._default_policy = next(iter(self._policies.keys()))
    
    def _parse_policy_config(self, name: str, config: Dict[str, Any]) -> PolicyConfig:
        """Parse policy configuration dictionary into PolicyConfig."""
        input_config = config.get("input", {})
        output_config = config.get("output", {})
        
        # Parse thresholds
        input_thresholds_data = input_config.get("thresholds", {})
        output_thresholds_data = output_config.get("thresholds", {})
        
        from langchain_xguard.models import PolicyThresholds
        
        input_thresholds = PolicyThresholds(
            jailbreak=input_thresholds_data.get("jailbreak", 0.7),
            pii=input_thresholds_data.get("pii", 0.8),
            toxicity=input_thresholds_data.get("toxicity", 0.6),
            compliance=input_thresholds_data.get("compliance", 0.75),
            custom=input_thresholds_data.get("custom", {}),
        )
        
        output_thresholds = PolicyThresholds(
            jailbreak=output_thresholds_data.get("jailbreak", 0.7),
            pii=output_thresholds_data.get("pii", 0.8),
            toxicity=output_thresholds_data.get("toxicity", 0.6),
            compliance=output_thresholds_data.get("compliance", 0.75),
            custom=output_thresholds_data.get("custom", {}),
        )
        
        # Parse action
        input_action_str = input_config.get("action", "block")
        output_action_str = output_config.get("action", "mask")
        
        input_action = Action(input_action_str)
        output_action = Action(output_action_str)
        
        return PolicyConfig(
            name=name,
            input_action=input_action,
            output_action=output_action,
            input_thresholds=input_thresholds,
            output_thresholds=output_thresholds,
            fallback_message=input_config.get("fallback_message", 
                "Your request cannot be processed due to safety concerns."),
            fallback_llm=input_config.get("fallback_llm"),
            enable_context=config.get("enable_context", True),
            context_window=config.get("context_window", 5),
            cache_enabled=config.get("cache_enabled", True),
            cache_ttl=config.get("cache_ttl", 300),
        )
    
    def get_policy(self, name: Optional[str] = None) -> PolicyConfig:
        """
        Get policy by name.
        
        Args:
            name: Policy name (uses default if None)
            
        Returns:
            PolicyConfig object
        """
        policy_name = name or self._default_policy
        if policy_name not in self._policies:
            # Return a safe default policy
            return PolicyConfig(name="fallback_default")
        return self._policies[policy_name]
    
    def list_policies(self) -> List[str]:
        """List all available policy names."""
        return list(self._policies.keys())
    
    def set_default_policy(self, name: str) -> None:
        """Set default policy."""
        if name not in self._policies:
            raise ValueError(f"Policy '{name}' not found")
        self._default_policy = name
    
    def rollback(self, version: str) -> None:
        """
        Rollback to a previous policy version.
        
        Args:
            version: Version to rollback to
        """
        for v, policies, _ in self._policy_history:
            if v == version:
                self._policies = policies
                self._version = version
                self._last_loaded = datetime.now()
                return
        raise ValueError(f"Version '{version}' not found in history")
    
    def get_version(self) -> str:
        """Get current policy version."""
        return self._version
    
    def get_last_loaded(self) -> Optional[datetime]:
        """Get last loaded timestamp."""
        return self._last_loaded
    
    def evaluate_action(
        self,
        result: DetectionResult,
        policy: Optional[PolicyConfig] = None,
        is_input: bool = True,
    ) -> PolicyActionResult:
        """
        Evaluate what action to take based on detection result and policy.
        
        Args:
            result: Detection result
            policy: Policy to use (uses default if None)
            is_input: Whether this is input detection
            
        Returns:
            PolicyActionResult with action and triggered risk categories
        """
        if policy is None:
            policy = self.get_policy()
        
        # Get thresholds and action for input/output
        if is_input:
            thresholds = policy.input_thresholds
            base_action = policy.input_action
        else:
            thresholds = policy.output_thresholds
            base_action = policy.output_action
        
        # Check if any category exceeds threshold, collecting all triggered categories
        triggered_categories = []
        for category in result.categories:
            # Skip safe category
            if category.category == "Safe-Safe":
                continue
            threshold = thresholds.get_threshold(category.category)
            if category.score >= threshold:
                triggered_categories.append(category)
        
        if triggered_categories:
            return PolicyActionResult(
                action=base_action,
                triggered_categories=triggered_categories,
            )
        
        # If no threshold exceeded, allow
        return PolicyActionResult(action=Action.ALLOW)
    
    def create_inline_policy(
        self,
        name: str,
        input_action: Action = Action.BLOCK,
        output_action: Action = Action.MASK,
        input_threshold: float = 0.7,
        output_threshold: float = 0.75,
    ) -> PolicyConfig:
        """
        Create an inline policy without loading from file.
        
        Args:
            name: Policy name
            input_action: Action for input detection
            output_action: Action for output detection
            input_threshold: Default threshold for input
            output_threshold: Default threshold for output
            
        Returns:
            PolicyConfig object
        """
        from langchain_xguard.models import PolicyThresholds
        
        policy = PolicyConfig(
            name=name,
            input_action=input_action,
            output_action=output_action,
            input_thresholds=PolicyThresholds(
                jailbreak=input_threshold,
                pii=input_threshold,
                toxicity=input_threshold,
                compliance=input_threshold,
            ),
            output_thresholds=PolicyThresholds(
                jailbreak=output_threshold,
                pii=output_threshold,
                toxicity=output_threshold,
                compliance=output_threshold,
            ),
        )
        
        self._policies[name] = policy
        return policy
    
    async def close(self) -> None:
        """Cleanup resources."""
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
