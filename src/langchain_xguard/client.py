"""Async XGuard client for security detection API."""

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import httpx
from langchain_core.runnables import RunnableConfig

from langchain_xguard.models import (
    DetectionResult,
    DetectionCategory,
    RiskLevel,
)


class XGuardClient:
    """
    Async client for XGuard security detection API.
    
    Supports both synchronous and streaming detection modes,
    with built-in caching and session management.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.xguard.ai/v1",
        timeout: float = 30.0,
        max_retries: int = 3,
        cache_enabled: bool = True,
        cache_ttl: int = 300,
    ):
        """
        Initialize XGuard client.
        
        Args:
            api_key: API key for authentication (optional for local mode)
            base_url: Base URL for XGuard API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            cache_enabled: Enable local caching for repeated requests
            cache_ttl: Cache TTL in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}  # hash -> (result, timestamp)
        self._session_states: Dict[str, List[Dict]] = {}  # session_id -> conversation history
        
    def _get_cache_key(self, content: str, session_id: Optional[str] = None) -> str:
        """Generate cache key from content and session."""
        key_data = f"{content}:{session_id}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[DetectionResult]:
        """Retrieve result from cache if valid."""
        if not self.cache_enabled:
            return None
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return DetectionResult.model_validate(result)
            else:
                del self._cache[key]
        return None
    
    def _save_to_cache(self, key: str, result: DetectionResult) -> None:
        """Save result to cache."""
        if self.cache_enabled:
            self._cache[key] = (result.model_dump(), time.time())
    
    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.time()
        expired_keys = [
            k for k, (_, ts) in self._cache.items()
            if now - ts >= self.cache_ttl
        ]
        for key in expired_keys:
            del self._cache[key]
    
    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make HTTP request to XGuard API with retry logic."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["Content-Type"] = "application/json"
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise
                except httpx.RequestError as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise RuntimeError("Unexpected error in request loop")
    
    def _update_session_state(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Update conversation history for context-aware detection."""
        if session_id not in self._session_states:
            self._session_states[session_id] = []
        
        self._session_states[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
    
    def _get_session_context(
        self,
        session_id: str,
        window_size: int = 5,
    ) -> List[Dict[str, str]]:
        """Get recent conversation history for context."""
        if session_id not in self._session_states:
            return []
        
        history = self._session_states[session_id][-window_size:]
        return [{"role": h["role"], "content": h["content"]} for h in history]
    
    async def detect_async(
        self,
        content: str,
        session_id: Optional[str] = None,
        context_window: int = 5,
        is_input: bool = True,
        custom_categories: Optional[List[str]] = None,
    ) -> DetectionResult:
        """
        Perform async security detection on content.
        
        Args:
            content: Text content to analyze
            session_id: Optional session ID for context-aware detection
            context_window: Number of previous turns to include
            is_input: Whether this is input (True) or output (False) detection
            custom_categories: Optional list of custom categories to check
            
        Returns:
            DetectionResult with risk assessment
        """
        # Check cache first
        cache_key = self._get_cache_key(content, session_id)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Build payload
        payload: Dict[str, Any] = {
            "content": content,
            "is_input": is_input,
            "timestamp": time.time(),
        }
        
        # Add context if available
        if session_id and context_window > 0:
            context = self._get_session_context(session_id, context_window)
            if context:
                payload["context"] = context
        
        # Add custom categories
        if custom_categories:
            payload["categories"] = custom_categories
        
        # Make API request
        try:
            response_data = await self._make_request("detect", payload)
            
            # Parse response
            categories = []
            for cat_data in response_data.get("categories", []):
                categories.append(DetectionCategory(
                    category=cat_data["name"],
                    score=cat_data["score"],
                    level=RiskLevel.from_score(cat_data["score"]),
                    details=cat_data.get("details"),
                    matched_patterns=cat_data.get("matched_patterns"),
                ))
            
            overall_score = response_data.get("overall_score", 0.0)
            result = DetectionResult(
                is_safe=response_data.get("is_safe", True),
                overall_level=RiskLevel.from_score(overall_score),
                overall_score=overall_score,
                categories=categories,
                metadata=response_data.get("metadata", {}),
                session_id=session_id,
                timestamp=time.time(),
            )
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            # Update session state
            if session_id:
                role = "user" if is_input else "assistant"
                self._update_session_state(session_id, role, content)
            
            return result
            
        except Exception as e:
            # Return safe default on error (fail-open or fail-closed based on config)
            # For now, fail-open to avoid blocking legitimate requests
            return DetectionResult(
                is_safe=True,
                overall_level=RiskLevel.SAFE,
                overall_score=0.0,
                categories=[],
                metadata={"error": str(e), "fallback": True},
                session_id=session_id,
                timestamp=time.time(),
            )
    
    async def detect_stream_chunk_async(
        self,
        chunk: str,
        buffer: List[str],
        threshold: float = 0.7,
        session_id: Optional[str] = None,
    ) -> tuple[bool, DetectionResult]:
        """
        Detect risk in streaming chunk with buffered context.
        
        Args:
            chunk: Current chunk content
            buffer: List of previous chunks in current stream
            threshold: Risk threshold for interruption
            session_id: Optional session ID
            
        Returns:
            Tuple of (should_interrupt, detection_result)
        """
        # Aggregate buffer + current chunk
        aggregated = "".join(buffer) + chunk
        
        # Only detect if we have enough content
        if len(aggregated) < 50:  # Minimum 50 chars for meaningful detection
            return False, DetectionResult(
                is_safe=True,
                overall_level=RiskLevel.SAFE,
                overall_score=0.0,
                categories=[],
            )
        
        result = await self.detect_async(
            content=aggregated,
            session_id=session_id,
            context_window=0,  # No additional context in streaming
            is_input=False,  # Output detection
        )
        
        should_interrupt = result.overall_score >= threshold
        return should_interrupt, result
    
    def clear_session(self, session_id: str) -> None:
        """Clear session state."""
        if session_id in self._session_states:
            del self._session_states[session_id]
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
    
    async def health_check(self) -> bool:
        """Check if XGuard API is healthy."""
        try:
            response_data = await self._make_request("health", {})
            return response_data.get("status") == "healthy"
        except Exception:
            return False
