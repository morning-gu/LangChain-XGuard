"""Async XGuard client for YuFeng-XGuard-Reason model inference."""

import asyncio
import hashlib
import time
import re
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_xguard.models import (
    DetectionResult,
    DetectionCategory,
    RiskLevel,
)


# Built-in risk category mapping from YuFeng-XGuard documentation
ID2RISK = {
    "sec": "Safe-Safe",
    "pc": "Crimes and Illegal Activities-Pornographic Contraband",
    "dc": "Crimes and Illegal Activities-Drug Crimes",
    "dw": "Crimes and Illegal Activities-Dangerous Weapons",
    "pi": "Crimes and Illegal Activities-Property Infringement",
    "ec": "Crimes and Illegal Activities-Economic Crimes",
    "ac": "Hate Speech-Abusive Curses",
    "def": "Hate Speech-Defamation",
    "ti": "Hate Speech-Threats and Intimidation",
    "cy": "Hate Speech-Cyberbullying",
    "ph": "Physical and Mental Health-Physical Health",
    "mh": "Physical and Mental Health-Mental Health",
    "se": "Ethics and Morality-Social Ethics",
    "sci": "Ethics and Morality-Science Ethics",
    "pp": "Data Privacy-Personal Privacy",
    "cs": "Data Privacy-Commercial Secret",
    "acc": "Cybersecurity-Access Control",
    "mc": "Cybersecurity-Malicious Code",
    "ha": "Cybersecurity-Hacker Attack",
    "ps": "Cybersecurity-Physical Security",
    "ter": "Extremism-Violent Terrorist Activities",
    "sd": "Extremism-Social Disruption",
    "ext": "Extremism-Extremist Ideological Trends",
    "fin": "Inappropriate Suggestions-Finance",
    "med": "Inappropriate Suggestions-Medicine",
    "law": "Inappropriate Suggestions-Law",
    "cm": "Risks Involving Minors-Corruption of Minors",
    "ma": "Risks Involving Minors-Minor Abuse and Exploitation",
    "md": "Risks Involving Minors-Minor Delinquency",
}

RISK2ID = {v: k for k, v in ID2RISK.items()}


class XGuardClient:
    """
    Client for YuFeng-XGuard-Reason local model inference.
    
    Based on Qwen3 architecture, optimized for real-time content safety evaluation.
    Supports two-stage output: risk classification followed by optional explanation.
    
    Features:
        - Multi-size models: 0.6B (fast) and 8B (accurate)
        - Two-stage output: risk label first, then explanation
        - Dynamic policy configuration (8B model only)
        - Comprehensive safety taxonomy
    """
    
    def __init__(
        self,
        model_name: str = "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        cache_enabled: bool = True,
        cache_ttl: int = 300,
        lazy_load: bool = True,
    ):
        """
        Initialize XGuard client with YuFeng-XGuard-Reason model.
        
        Args:
            model_name: Model identifier on ModelScope/HuggingFace
            device_map: Device mapping strategy for model loading
            torch_dtype: Data type for model weights
            cache_enabled: Enable local caching for repeated requests
            cache_ttl: Cache TTL in seconds
            lazy_load: Defer model loading until first inference
        """
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}  # hash -> (result, timestamp)
        self._session_states: Dict[str, List[Dict]] = {}  # session_id -> conversation history
        
        # Lazy loading
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._loaded = False
        
        if not lazy_load:
            self.load_model()
    
    @property
    def model(self) -> Any:
        """Get model instance, loading if necessary."""
        if not self._loaded:
            self.load_model()
        return self._model
    
    @property
    def tokenizer(self) -> Any:
        """Get tokenizer instance, loading if necessary."""
        if not self._loaded:
            self.load_model()
        return self._tokenizer
    
    def load_model(self) -> None:
        """Load model and tokenizer into memory."""
        if self._loaded:
            return
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        ).eval()
        self._loaded = True
    
    def _get_cache_key(self, content: str, session_id: Optional[str] = None, 
                       policy: Optional[str] = None, reason_first: bool = False) -> str:
        """Generate cache key from content and parameters."""
        key_data = f"{content}:{session_id}:{policy}:{reason_first}"
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
    
    def _build_messages(
        self,
        content: str,
        is_input: bool = True,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build message list for model inference.
        
        Args:
            content: Current content to evaluate
            is_input: Whether this is user input (True) or assistant output (False)
            context: Optional conversation history
            
        Returns:
            Formatted messages list
        """
        messages = []
        
        # Add context if provided
        if context:
            messages.extend(context)
        
        # Add current message
        role = "user" if is_input else "assistant"
        messages.append({"role": role, "content": content})
        
        return messages
    
    def _parse_inference_result(
        self,
        outputs: Any,
        input_length: int,
        reason_first: bool = False,
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """
        Parse model inference output into structured result.
        
        Args:
            outputs: Raw model generation outputs
            input_length: Length of input tokens
            reason_first: Whether model was configured for reason-first output
            
        Returns:
            Tuple of (response_text, token_scores, risk_scores)
        """
        batch_idx = 0
        
        # Decode response text
        output_ids = outputs["sequences"].tolist()[batch_idx][input_length:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Parse token-level scores
        generated_tokens_with_probs = []
        generated_tokens = outputs.sequences[:, input_length:]
        
        scores = torch.stack(outputs.scores, 1)
        scores = scores.softmax(-1)
        scores_topk_value, scores_topk_index = scores.topk(k=10, dim=-1)
        
        for generated_token, score_topk_value, score_topk_index in zip(
            generated_tokens, scores_topk_value, scores_topk_index
        ):
            generated_tokens_with_prob = []
            for token, topk_value, topk_index in zip(
                generated_token, score_topk_value, score_topk_index
            ):
                token = int(token.cpu())
                if token == self.tokenizer.pad_token_id:
                    continue
                
                res_topk_score = {}
                for ii, (value, index) in enumerate(zip(topk_value, topk_index)):
                    if ii == 0 or value.cpu().numpy() > 1e-4:
                        text = self.tokenizer.decode(index.cpu().numpy())
                        res_topk_score[text] = {
                            "id": str(int(index.cpu().numpy())),
                            "prob": round(float(value.cpu().numpy()), 4),
                        }
                
                generated_tokens_with_prob.append(res_topk_score)
            
            generated_tokens_with_probs.append(generated_tokens_with_prob)
        
        # Extract risk scores from appropriate token position
        score_idx = max(len(generated_tokens_with_probs[batch_idx]) - 2, 0) if reason_first else 0
        id2risk = self.tokenizer.init_kwargs.get('id2risk', ID2RISK)
        
        token_score = {
            k: v['prob'] 
            for k, v in generated_tokens_with_probs[batch_idx][score_idx].items()
        }
        risk_score = {
            id2risk.get(k, k): v['prob'] 
            for k, v in generated_tokens_with_probs[batch_idx][score_idx].items() 
            if k in id2risk
        }
        
        return response, token_score, risk_score
    
    def _infer(
        self,
        messages: List[Dict[str, str]],
        policy: Optional[str] = None,
        max_new_tokens: int = 1,
        reason_first: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform model inference on messages.
        
        Args:
            messages: Formatted message list
            policy: Optional dynamic policy string (8B model only)
            max_new_tokens: Maximum tokens to generate
            reason_first: Whether to generate explanation before label
            
        Returns:
            Dictionary with response, token_score, and risk_score
        """
        # Apply chat template
        rendered_query = self.tokenizer.apply_chat_template(
            messages, 
            policy=policy, 
            reason_first=reason_first, 
            tokenize=False
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([rendered_query], return_tensors="pt").to(self.model.device)
        
        # Generate output
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        # Parse results
        response, token_score, risk_score = self._parse_inference_result(
            outputs, 
            model_inputs['input_ids'].shape[1],
            reason_first,
        )
        
        return {
            'response': response,
            'token_score': token_score,
            'risk_score': risk_score,
        }
    
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
        policy: Optional[str] = None,
        reason_first: bool = False,
        max_new_tokens: int = 1,
    ) -> DetectionResult:
        """
        Perform async security detection using YuFeng-XGuard-Reason model.
        
        Args:
            content: Text content to analyze
            session_id: Optional session ID for context-aware detection
            context_window: Number of previous turns to include
            is_input: Whether this is input (True) or output (False) detection
            policy: Optional dynamic policy string (8B model only)
            reason_first: Whether to generate explanation before label
            max_new_tokens: Maximum tokens to generate (1 for label only, more for explanation)
            
        Returns:
            DetectionResult with risk assessment
        """
        # Check cache first
        cache_key = self._get_cache_key(content, session_id, policy, reason_first)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Get context if available
        context = None
        if session_id and context_window > 0:
            context = self._get_session_context(session_id, context_window)
        
        # Build messages
        messages = self._build_messages(content, is_input, context)
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result_dict = await loop.run_in_executor(
            None,
            lambda: self._infer(messages, policy, max_new_tokens, reason_first),
        )
        
        # Parse risk scores into categories
        categories = []
        for category_name, score in result_dict['risk_score'].items():
            if score > 1e-4:  # Filter very low scores
                categories.append(DetectionCategory(
                    category=category_name,
                    score=score,
                    level=RiskLevel.from_score(score),
                    details={"explanation": result_dict['response']} if max_new_tokens > 1 else None,
                ))
        
        # Calculate overall score (max of all category scores)
        overall_score = max([c.score for c in categories], default=0.0)
        
        # Determine if safe (highest score is 'Safe-Safe')
        is_safe = False
        if categories:
            top_category = max(categories, key=lambda c: c.score)
            is_safe = top_category.category == "Safe-Safe"
        
        result = DetectionResult(
            is_safe=is_safe,
            overall_level=RiskLevel.from_score(overall_score),
            overall_score=overall_score,
            categories=categories,
            metadata={
                "response": result_dict['response'],
                "token_score": result_dict['token_score'],
                "policy_used": policy is not None,
                "reason_first": reason_first,
            },
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
    
    async def detect_stream_chunk_async(
        self,
        chunk: str,
        buffer: List[str],
        threshold: float = 0.7,
        session_id: Optional[str] = None,
    ) -> Tuple[bool, DetectionResult]:
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
        if len(aggregated) < 50:
            return False, DetectionResult(
                is_safe=True,
                overall_level=RiskLevel.SAFE,
                overall_score=0.0,
                categories=[],
            )
        
        result = await self.detect_async(
            content=aggregated,
            session_id=session_id,
            context_window=0,
            is_input=False,
            max_new_tokens=1,
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
    
    def get_risk_taxonomy(self) -> Dict[str, str]:
        """Get the built-in risk category taxonomy."""
        return dict(ID2RISK)
