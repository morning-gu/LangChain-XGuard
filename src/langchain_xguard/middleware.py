"""LangChain LCEL middleware for XGuard security integration."""

import asyncio
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
    RiskLevel,
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
        
        # Extract session ID from config if available
        session_id = config.get("configurable", {}).get("session_id")
        
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
        """Apply action based on input detection result."""
        policy = self.get_policy()
        action = self.policy_engine.evaluate_action(result, policy, is_input=True)
        
        if action == Action.BLOCK:
            # Return fallback message
            return policy.fallback_message
        elif action == Action.REWRITE:
            # TODO: Implement rewrite logic with fallback LLM
            # For now, return original content
            return original_content
        elif action == Action.LOG_ONLY:
            # Log and allow
            # TODO: Integrate with logging/Observability
            return original_content
        else:
            # ALLOW or default
            return original_content


class XGuardOutputMiddleware(XGuardMiddleware):
    """
    Output middleware for detecting unsafe LLM outputs.
    
    Placed after LLM in LCEL pipeline to mask/filter responses.
    Supports streaming chunk-level detection with graceful interruption.
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
        """Apply action based on output detection result."""
        policy = self.get_policy()
        action = self.policy_engine.evaluate_action(result, policy, is_input=False)
        
        if action == Action.BLOCK:
            return policy.fallback_message
        elif action == Action.MASK:
            # Simple masking - in production, would use entity recognition
            return original_content.replace(
                self._get_sensitive_content(result),
                self.mask_pattern,
            )
        elif action == Action.REWRITE:
            # TODO: Implement rewrite with fallback LLM
            return original_content
        elif action == Action.LOG_ONLY:
            # Log and allow
            return original_content
        else:
            return original_content
    
    def _get_sensitive_content(self, result: DetectionResult) -> str:
        """Extract sensitive content from detection result."""
        # In production, this would use NER or pattern matching
        # For now, return empty string (no masking)
        return ""
    
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
        
        # Get the next runnable in the chain
        # This is called when middleware is part of a chain
        
        # Buffer for accumulating chunks
        buffer: List[str] = []
        interrupted = False
        interruption_result: Optional[DetectionResult] = None
        
        # Process input through the chain
        # Note: In actual LCEL usage, this wraps another runnable
        # For standalone usage, we expect input to be the content
        
        if isinstance(input, dict):
            content = input.get("input", input.get("content", str(input)))
        else:
            content = str(input)
        
        # Simulate streaming by yielding content in chunks
        # In real usage, this would wrap an LLM's stream
        chunk_size = max(10, len(content) // self.chunk_threshold)
        
        for i in range(0, len(content), chunk_size):
            if interrupted:
                break
            
            chunk = content[i:i + chunk_size]
            buffer.append(chunk)
            
            # Check safety if buffer is large enough
            if len(buffer) >= self.chunk_threshold:
                aggregated = "".join(buffer)
                should_interrupt, result = await self.client.detect_stream_chunk_async(
                    chunk="",
                    buffer=buffer,
                    threshold=policy.output_thresholds.compliance,
                    session_id=session_id,
                )
                
                if should_interrupt:
                    interrupted = True
                    interruption_result = result
                    # Yield fallback message instead
                    yield policy.fallback_message
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
            action = self.policy_engine.evaluate_action(result, policy, is_input=False)
            
            if action == Action.BLOCK:
                # Already yielded chunks, but we can't take them back
                # In real streaming, we'd close the stream early
                pass
