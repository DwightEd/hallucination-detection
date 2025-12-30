"""LLM-as-Judge for hallucination evaluation.

Supports:
- Qwen API (dashscope)
- OpenAI-compatible APIs
- Custom prompts for different evaluation criteria
- Batch processing with rate limiting
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import logging
import time
import json
import re
from abc import ABC, abstractmethod

from src.core import Sample, JudgeResult, LLMAPIConfig, Progress, APIError

logger = logging.getLogger(__name__)


class JudgeMode(str, Enum):
    """Evaluation mode for LLM judge."""
    BINARY = "binary"              # Simple yes/no hallucination
    SCORE = "score"                # 1-5 score
    DETAILED = "detailed"          # Detailed analysis with reasoning
    COMPARISON = "comparison"      # Compare response to reference


# ==============================================================================
# Prompt Templates
# ==============================================================================

BINARY_PROMPT = """You are an expert evaluator for detecting hallucinations in AI-generated responses.

A hallucination is when the AI response contains information that:
1. Is factually incorrect
2. Is not supported by the provided context/source
3. Makes claims that cannot be verified from the given information
4. Contradicts the source material

## Task
Evaluate if the following response contains any hallucinations.

## Context/Source
{context}

## Question/Prompt
{prompt}

## Response to Evaluate
{response}

## Instructions
Analyze the response carefully and determine if it contains hallucinations.
Respond with ONLY a JSON object in this exact format:
{{"hallucinated": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}
"""

SCORE_PROMPT = """You are an expert evaluator for assessing the factual accuracy of AI-generated responses.

## Task
Rate the factual accuracy of the following response on a scale of 1-5:
- 1: Completely hallucinated (no factual basis)
- 2: Mostly hallucinated (major factual errors)
- 3: Partially accurate (some errors or unsupported claims)
- 4: Mostly accurate (minor issues only)
- 5: Fully accurate (all claims supported)

## Context/Source
{context}

## Question/Prompt
{prompt}

## Response to Evaluate
{response}

## Instructions
Provide your evaluation as a JSON object:
{{"score": 1-5, "confidence": 0.0-1.0, "issues": ["list of issues if any"], "reason": "explanation"}}
"""

DETAILED_PROMPT = """You are an expert evaluator for detecting and analyzing hallucinations in AI-generated responses.

## Definitions
- **Intrinsic Hallucination**: Information that contradicts the source
- **Extrinsic Hallucination**: Information that cannot be verified from the source
- **Fabrication**: Made-up facts, statistics, or quotes
- **Misattribution**: Attributing information to wrong sources

## Context/Source
{context}

## Question/Prompt
{prompt}

## Response to Evaluate
{response}

## Instructions
Analyze the response thoroughly and provide:
1. Overall hallucination assessment
2. Specific hallucinated spans (if any)
3. Type of hallucination for each span
4. Confidence level

Respond with a JSON object:
{{
    "hallucinated": true/false,
    "score": 1-5,
    "confidence": 0.0-1.0,
    "hallucination_spans": [
        {{"text": "hallucinated text", "type": "intrinsic/extrinsic/fabrication", "explanation": "why"}}
    ],
    "reasoning": "detailed analysis"
}}
"""


# ==============================================================================
# Base LLM Client
# ==============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""
    
    def __init__(self, config: LLMAPIConfig):
        self.config = config
        self.request_count = 0
        self.last_request_time = 0.0
    
    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """Make API call and return response text."""
        pass
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self.config.rate_limit > 0:
            min_interval = 60.0 / self.config.rate_limit
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1


class QwenClient(BaseLLMClient):
    """Client for Qwen API via dashscope."""
    
    def __init__(self, config: LLMAPIConfig):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Lazy load dashscope client."""
        if self._client is None:
            try:
                import dashscope
                dashscope.api_key = self.config.api_key
                self._client = dashscope
            except ImportError:
                raise APIError("dashscope not installed. Run: pip install dashscope")
        return self._client
    
    def call(self, prompt: str, **kwargs) -> str:
        """Call Qwen API.
        
        Args:
            prompt: User prompt
            **kwargs: Additional parameters
            
        Returns:
            Response text
        """
        self._rate_limit()
        
        client = self._get_client()
        
        messages = [{"role": "user", "content": prompt}]
        
        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})
        
        try:
            from dashscope import Generation
            
            response = Generation.call(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                result_format="message",
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise APIError(f"Qwen API error: {response.code} - {response.message}")
                
        except Exception as e:
            raise APIError(f"Qwen API call failed: {e}")


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs."""
    
    def __init__(self, config: LLMAPIConfig):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                raise APIError("openai not installed. Run: pip install openai")
        return self._client
    
    def call(self, prompt: str, **kwargs) -> str:
        """Call OpenAI-compatible API.
        
        Args:
            prompt: User prompt
            **kwargs: Additional parameters
            
        Returns:
            Response text
        """
        self._rate_limit()
        
        client = self._get_client()
        
        messages = [{"role": "user", "content": prompt}]
        
        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})
        
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise APIError(f"OpenAI API call failed: {e}")


def create_llm_client(config: LLMAPIConfig) -> BaseLLMClient:
    """Create LLM client based on config.
    
    Args:
        config: API configuration
        
    Returns:
        LLM client instance
    """
    provider = config.provider.lower()
    
    if provider in ["qwen", "dashscope", "alibaba"]:
        return QwenClient(config)
    elif provider in ["openai", "azure", "openai_compatible"]:
        return OpenAICompatibleClient(config)
    else:
        raise APIError(f"Unknown provider: {provider}")


# ==============================================================================
# LLM Judge
# ==============================================================================

class LLMJudge:
    """LLM-based hallucination judge.
    
    Uses LLM to evaluate whether responses contain hallucinations.
    """
    
    def __init__(
        self,
        config: LLMAPIConfig,
        mode: JudgeMode = JudgeMode.BINARY,
        custom_prompt: Optional[str] = None,
    ):
        """Initialize LLM judge.
        
        Args:
            config: LLM API configuration
            mode: Evaluation mode
            custom_prompt: Custom prompt template (optional)
        """
        self.config = config
        self.mode = mode
        self.client = create_llm_client(config)
        
        # Select prompt template
        if custom_prompt:
            self.prompt_template = custom_prompt
        elif mode == JudgeMode.BINARY:
            self.prompt_template = BINARY_PROMPT
        elif mode == JudgeMode.SCORE:
            self.prompt_template = SCORE_PROMPT
        elif mode == JudgeMode.DETAILED:
            self.prompt_template = DETAILED_PROMPT
        else:
            self.prompt_template = BINARY_PROMPT
        
        logger.info(f"LLM Judge initialized: {config.provider}/{config.model}, mode={mode}")
    
    def _build_prompt(self, sample: Sample) -> str:
        """Build evaluation prompt from sample.
        
        Args:
            sample: Sample to evaluate
            
        Returns:
            Formatted prompt string
        """
        context = sample.reference or sample.metadata.get("context", "No context provided")
        
        return self.prompt_template.format(
            context=context,
            prompt=sample.prompt,
            response=sample.response,
        )
    
    def _parse_response(self, response: str, sample_id: str) -> JudgeResult:
        """Parse LLM response into JudgeResult.
        
        Args:
            response: Raw LLM response
            sample_id: Sample identifier
            
        Returns:
            JudgeResult instance
        """
        # Try to extract JSON from response
        try:
            # Look for JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Try parsing entire response as JSON
                data = json.loads(response)
            
            # Extract fields based on mode
            if self.mode == JudgeMode.BINARY:
                return JudgeResult(
                    sample_id=sample_id,
                    hallucinated=data.get("hallucinated", False),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reason", ""),
                )
            
            elif self.mode == JudgeMode.SCORE:
                score = data.get("score", 3)
                return JudgeResult(
                    sample_id=sample_id,
                    hallucinated=score <= 2,
                    score=score,
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reason", ""),
                    metadata={"issues": data.get("issues", [])},
                )
            
            elif self.mode == JudgeMode.DETAILED:
                return JudgeResult(
                    sample_id=sample_id,
                    hallucinated=data.get("hallucinated", False),
                    score=data.get("score", 3),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    hallucination_spans=data.get("hallucination_spans", []),
                )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response for {sample_id}: {e}")
            
            # Fallback: try to detect keywords
            response_lower = response.lower()
            hallucinated = any(word in response_lower for word in 
                            ["hallucinated", "incorrect", "false", "fabricated", "not supported"])
            
            return JudgeResult(
                sample_id=sample_id,
                hallucinated=hallucinated,
                confidence=0.3,  # Low confidence for fallback
                reasoning=response[:500],  # Keep first 500 chars as reasoning
            )
    
    def judge(self, sample: Sample, retries: int = 3) -> JudgeResult:
        """Judge a single sample.
        
        Args:
            sample: Sample to evaluate
            retries: Number of retries on failure
            
        Returns:
            JudgeResult instance
        """
        prompt = self._build_prompt(sample)
        
        for attempt in range(retries):
            try:
                response = self.client.call(prompt)
                return self._parse_response(response, sample.id)
            
            except APIError as e:
                logger.warning(f"API error on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return JudgeResult(
                        sample_id=sample.id,
                        hallucinated=False,
                        confidence=0.0,
                        reasoning=f"API error: {e}",
                    )
        
        return JudgeResult(sample_id=sample.id, hallucinated=False, confidence=0.0)
    
    def judge_batch(
        self,
        samples: List[Sample],
        show_progress: bool = True,
    ) -> List[JudgeResult]:
        """Judge multiple samples.
        
        Args:
            samples: List of samples
            show_progress: Whether to show progress bar
            
        Returns:
            List of JudgeResults
        """
        results = []
        
        if show_progress:
            with Progress(len(samples), desc="LLM Judging") as pbar:
                for sample in samples:
                    result = self.judge(sample)
                    results.append(result)
                    pbar.update()
        else:
            for sample in samples:
                result = self.judge(sample)
                results.append(result)
        
        logger.info(f"Judged {len(results)} samples, "
                   f"{sum(r.hallucinated for r in results)} hallucinated")
        
        return results


def create_judge(
    config: LLMAPIConfig,
    mode: str = "binary",
    custom_prompt: Optional[str] = None,
) -> LLMJudge:
    """Create LLM judge.
    
    Args:
        config: API configuration
        mode: Evaluation mode (binary, score, detailed)
        custom_prompt: Custom prompt template
        
    Returns:
        LLMJudge instance
    """
    judge_mode = JudgeMode(mode) if isinstance(mode, str) else mode
    return LLMJudge(config, judge_mode, custom_prompt)


# ==============================================================================
# Evaluation with Judge
# ==============================================================================

def evaluate_with_judge(
    samples: List[Sample],
    judge: LLMJudge,
    ground_truth: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Evaluate samples using LLM judge and compute metrics.
    
    Args:
        samples: Samples to evaluate
        judge: LLM judge instance
        ground_truth: Ground truth labels (optional)
        
    Returns:
        Dictionary with results and metrics
    """
    from .metrics import compute_metrics, Prediction
    
    # Get judge results
    judge_results = judge.judge_batch(samples)
    
    # Convert to predictions for metrics
    predictions = [
        Prediction(
            sample_id=r.sample_id,
            score=r.confidence if r.hallucinated else 1 - r.confidence,
            label=1 if r.hallucinated else 0,
            confidence=r.confidence,
        )
        for r in judge_results
    ]
    
    result = {
        "judge_results": judge_results,
        "predictions": predictions,
        "n_hallucinated": sum(r.hallucinated for r in judge_results),
        "n_clean": sum(not r.hallucinated for r in judge_results),
    }
    
    # Compute metrics if ground truth available
    if ground_truth is not None:
        metrics = compute_metrics(predictions, ground_truth)
        result["metrics"] = metrics
        logger.info(f"Judge evaluation metrics: {metrics}")
    
    return result
