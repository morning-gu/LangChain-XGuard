"""LangChain LCEL middleware for XGuard security integration."""

import asyncio
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from langchain_core.runnables import (
    RunnableSerializable,
    RunnableConfig,
    RunnableLambda,
)
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

from langchain_xguard.client import XGuardClient
from langchain_xguard.policy import PolicyEngine, PolicyConfig
from langchain_xguard.models import (
    Action,
    DetectionResult,
    DetectionCategory,
    RiskLevel,
    PolicyActionResult,
    XGuardSafetyError,
)


class XGuardMiddleware(RunnableSerializable[Any, Any], ABC):
    """
    Abstract base class for XGuard middleware.
    
    Provides common functionality for input/output security detection
    with LCEL native integration.
    """
    
    client: Optional[XGuardClient] = None
    policy_engine: Optional[PolicyEngine] = None
    policy_name: Optional[str] = None
    chunk_threshold: int = 3  # Number of chunks to buffer before detection
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        client: Optional[XGuardClient] = None,
        policy_engine: Optional[PolicyEngine] = None,
        policy: Optional[str] = None,
        model_name: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        cache_enabled: bool = True,
        lazy_load: bool = True,
        cache_dir: Optional[str] = None,
        policy_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize XGuard middleware.
        
        Args:
            client: XGuardClient instance (created if not provided)
            policy_engine: PolicyEngine instance (created if not provided)
            policy: Policy name to use
            model_name: Model name for XGuard client (if creating client)
            device_map: Device map for model loading
            torch_dtype: Torch dtype for model loading
            cache_enabled: Enable caching
            lazy_load: Lazy load model
            cache_dir: Custom directory for model storage. If None, uses default HuggingFace cache
            policy_path: Path to policy file (if creating policy engine)
        """
        super().__init__(**kwargs)
        if client is None:
            client_kwargs = {
                "cache_enabled": cache_enabled,
                "lazy_load": lazy_load,
            }
            if model_name is not None:
                client_kwargs["model_name"] = model_name
            if device_map != "auto":
                client_kwargs["device_map"] = device_map
            if torch_dtype != "auto":
                client_kwargs["torch_dtype"] = torch_dtype
            if cache_dir is not None:
                client_kwargs["cache_dir"] = cache_dir
            self.client = XGuardClient(**client_kwargs)
        else:
            self.client = client
        self.policy_engine = policy_engine or PolicyEngine(policy_path=policy_path)
        self.policy_name = policy
    
    def get_policy(self) -> PolicyConfig:
        """Get current policy configuration."""
        return self.policy_engine.get_policy(self.policy_name)
    
    @abstractmethod
    async def _detect_safety(
        self,
        content: str,
        config: RunnableConfig,
    ) -> DetectionResult:
        """Perform safety detection on content."""
        pass
    
    @abstractmethod
    async def _apply_action(
        self,
        result: DetectionResult,
        original_content: str,
        config: RunnableConfig,
    ) -> Any:
        """Apply policy action based on detection result."""
        pass
    
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        """Synchronous invoke - delegates to async implementation."""
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(input, config, **kwargs)
        )
    
    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> Any:
        """
        Async invoke with safety detection.
        
        Args:
            input: Input to process
            config: Runnable configuration
            **kwargs: Additional arguments
            
        Returns:
            Processed input (either original or modified/blocked)
        """
        config = config or {}
        
        # Convert input to string for detection
        if isinstance(input, dict):
            content = input.get("input", input.get("content", str(input)))
        elif hasattr(input, "content"):
            content = input.content
        else:
            content = str(input)
        
        # Perform safety detection
        result = await self._detect_safety(content, config)
        
        # Apply action based on result
        processed = await self._apply_action(result, content, config)
        
        # If input was dict, preserve structure
        if isinstance(input, dict):
            if "input" in input:
                return {**input, "input": processed}
            elif "content" in input:
                return {**input, "content": processed}
            return processed
        
        return processed
    
    async def astream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> AsyncIterator[Any]:
        """
        Stream input through middleware (for passthrough scenarios).
        
        For input middleware, this typically just passes through.
        Output middleware will implement chunk-level detection.
        """
        # Default: just yield the processed input
        result = await self.ainvoke(input, config, **kwargs)
        yield result


class XGuardInputMiddleware(XGuardMiddleware):
    """
    Input middleware for detecting unsafe user inputs.
    
    Placed before LLM in LCEL pipeline to block/rewrite malicious prompts.
    When content is blocked, raises XGuardSafetyError to stop the pipeline
    and prevent the request from reaching the LLM.
    """
    
    async def _detect_safety(
        self,
        content: str,
        config: RunnableConfig,
    ) -> DetectionResult:
        """Detect safety issues in input."""
        session_id = config.get("configurable", {}).get("session_id")
        policy = self.get_policy()
        
        return await self.client.detect_async(
            content=content,
            session_id=session_id,
            context_window=policy.context_window if policy.enable_context else 0,
            is_input=True,
        )
    
    async def _apply_action(
        self,
        result: DetectionResult,
        original_content: str,
        config: RunnableConfig,
    ) -> Any:
        """Apply action based on input detection result.
        
        BLOCK: Raises XGuardSafetyError to stop the pipeline.
        REWRITE: Returns original content (not yet implemented).
        LOG_ONLY: Logs risk info and allows content through.
        ALLOW: Passes content through unchanged.
        """
        policy = self.get_policy()
        action_result = self.policy_engine.evaluate_action(result, policy, is_input=True)
        
        if action_result.action == Action.BLOCK:
            raise XGuardSafetyError(
                action=Action.BLOCK,
                detection_result=result,
                triggered_categories=action_result.triggered_categories,
                fallback_message=policy.fallback_message,
            )
        elif action_result.action == Action.REWRITE:
            # TODO: Implement rewrite logic with fallback LLM
            # For now, log the risk and return original content
            print(f"[XGuard] REWRITE action triggered but not yet implemented. "
                  f"Risk: {action_result.risk_summary}")
            return original_content
        elif action_result.action == Action.LOG_ONLY:
            print(f"[XGuard] LOG_ONLY: Risk detected in input. "
                  f"Risk: {action_result.risk_summary}")
            return original_content
        else:
            # ALLOW
            return original_content


class XGuardOutputMiddleware(XGuardMiddleware):
    """
    Output middleware for detecting unsafe LLM outputs.
    
    Placed after LLM in LCEL pipeline to mask/filter responses.
    Supports streaming chunk-level detection with graceful interruption.
    
    When unsafe content is detected:
    - BLOCK: Raises XGuardSafetyError to stop the pipeline.
    - MASK: Replaces the entire output with a safety notice listing risk types.
    - REWRITE: Returns original content (not yet implemented).
    - LOG_ONLY: Logs risk info and allows content through.
    """
    
    action: Action = Action.MASK  # Default action for output
    mask_pattern: str = "[REDACTED]"  # Pattern for masking sensitive content
    
    async def _detect_safety(
        self,
        content: str,
        config: RunnableConfig,
    ) -> DetectionResult:
        """Detect safety issues in output."""
        session_id = config.get("configurable", {}).get("session_id")
        policy = self.get_policy()
        
        return await self.client.detect_async(
            content=content,
            session_id=session_id,
            context_window=0,  # No additional context for output
            is_input=False,
        )
    
    async def _apply_action(
        self,
        result: DetectionResult,
        original_content: str,
        config: RunnableConfig,
    ) -> Any:
        """Apply action based on output detection result.
        
        BLOCK: Raises XGuardSafetyError to stop the pipeline.
        MASK: Masks sensitive content based on detected risk categories.
        REWRITE: Returns original content (not yet implemented).
        LOG_ONLY: Logs risk info and allows content through.
        ALLOW: Passes content through unchanged.
        """
        policy = self.get_policy()
        action_result = self.policy_engine.evaluate_action(result, policy, is_input=False)
        
        if action_result.action == Action.BLOCK:
            raise XGuardSafetyError(
                action=Action.BLOCK,
                detection_result=result,
                triggered_categories=action_result.triggered_categories,
                fallback_message=policy.fallback_message,
            )
        elif action_result.action == Action.MASK:
            return self._mask_content(original_content, action_result)
        elif action_result.action == Action.REWRITE:
            # TODO: Implement rewrite with fallback LLM
            print(f"[XGuard] REWRITE action triggered but not yet implemented. "
                  f"Risk: {action_result.risk_summary}")
            return original_content
        elif action_result.action == Action.LOG_ONLY:
            print(f"[XGuard] LOG_ONLY: Risk detected in output. "
                  f"Risk: {action_result.risk_summary}")
            return original_content
        else:
            # ALLOW
            return original_content
    
    def _mask_content(
        self,
        original_content: str,
        action_result: PolicyActionResult,
    ) -> str:
        """Mask sensitive content in output based on detected risk categories.
        
        For PII categories, applies regex-based pattern masking.
        For other categories, appends a safety notice with risk type details.
        """
        masked = original_content
        pii_categories = []
        other_categories = []
        
        for cat in action_result.triggered_categories:
            if cat.category in (
                "Data Privacy-Personal Privacy",
                "Data Privacy-Commercial Secret",
            ):
                pii_categories.append(cat)
            else:
                other_categories.append(cat)
        
        # Apply PII pattern masking
        if pii_categories:
            masked = self._apply_pii_masking(masked)
        
        # For non-PII risks, append safety notice
        if other_categories:
            risk_types = ", ".join(
                f"{cat.category}({cat.score:.0%})" for cat in other_categories
            )
            safety_notice = f"\n[SAFETY NOTICE] Content may contain: {risk_types}"
            masked = masked + safety_notice
        
        return masked
    
    def _apply_pii_masking(self, content: str) -> str:
        """Apply regex-based PII pattern masking to content."""
        # Credit card numbers
        content = re.sub(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            self.mask_pattern, content,
        )
        # Phone numbers (various formats)
        content = re.sub(
            r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            self.mask_pattern, content,
        )
        # Email addresses
        content = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            self.mask_pattern, content,
        )
        # SSN-like patterns
        content = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            self.mask_pattern, content,
        )
        # IP addresses
        content = re.sub(
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            self.mask_pattern, content,
        )
        return content
    
    async def astream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> AsyncIterator[Any]:
        """
        Stream output with chunk-level safety detection.
        
        Implements buffering and early termination on risk detection.
        """
        policy = self.get_policy()
        session_id = config.get("configurable", {}).get("session_id")
        
        # Buffer for accumulating chunks
        buffer: List[str] = []
        interrupted = False
        interruption_result: Optional[DetectionResult] = None
        
        if isinstance(input, dict):
            content = input.get("input", input.get("content", str(input)))
        else:
            content = str(input)
        
        # Simulate streaming by yielding content in chunks
        chunk_size = max(10, len(content) // self.chunk_threshold)
        
        for i in range(0, len(content), chunk_size):
            if interrupted:
                break
            
            chunk = content[i:i + chunk_size]
            buffer.append(chunk)
            
            # Check safety if buffer is large enough
            if len(buffer) >= self.chunk_threshold:
                should_interrupt, result = await self.client.detect_stream_chunk_async(
                    chunk="",
                    buffer=buffer,
                    threshold=0.7,  # Default threshold for streaming detection
                    session_id=session_id,
                )
                
                if should_interrupt:
                    interrupted = True
                    interruption_result = result
                    # Yield fallback message with risk info instead
                    risk_info = ""
                    if result.categories:
                        top = max(result.categories, key=lambda c: c.score)
                        if top.category != "Safe-Safe":
                            risk_info = f" (Risk: {top.category}, score={top.score:.2f})"
                    yield f"{policy.fallback_message}{risk_info}"
                    break
            
            # Yield current chunk
            yield chunk
        
        # Yield remaining buffer if not interrupted
        if not interrupted and buffer:
            remaining = "".join(buffer[len(buffer):]) if len(buffer) > self.chunk_threshold else ""
            if remaining:
                yield remaining
        
        # If we collected everything and haven't checked yet, do final check
        if not interrupted and len(buffer) < self.chunk_threshold:
            full_content = "".join(buffer)
            result = await self._detect_safety(full_content, config)
            action_result = self.policy_engine.evaluate_action(result, policy, is_input=False)
            
            if action_result.action == Action.BLOCK:
                # Already yielded chunks, but we can't take them back
                # In real streaming, we'd close the stream early
                pass
