"""Feature extraction for hallucination detection.

Key features:
- Teacher forcing mode: Concatenate prompt+response, forward pass
- Generation mode: Generate response, extract during generation
- Diagonal extraction for attention (lapeigvals style)
- Laplacian diagonal computation
- Hidden states pooling
- Token probability extraction
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import logging
import torch
import torch.nn.functional as F

from src.core import (
    Sample, ExtractedFeatures, FeaturesConfig, ExtractionMode,
    parse_layers, Progress, FeatureError,
)
from src.models import LoadedModel

logger = logging.getLogger(__name__)


# ==============================================================================
# Attention Feature Extraction
# ==============================================================================

def extract_attention_diagonal(attention: torch.Tensor) -> torch.Tensor:
    """Extract diagonal from attention matrix.
    
    Args:
        attention: Attention weights [batch, heads, seq, seq]
        
    Returns:
        Diagonal values [batch, heads, seq]
    """
    return torch.diagonal(attention, dim1=-2, dim2=-1)


def compute_laplacian_diagonal(attention: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian diagonal from attention matrix.
    
    Laplacian L = D - A, where D is degree matrix.
    For row-normalized attention, D_ii = sum_j A_ij = 1 (approximately).
    Laplacian diagonal L_ii = D_ii - A_ii = 1 - A_ii (approximately).
    
    Following lapeigvals: L = D - A where D is out-degree.
    
    Args:
        attention: Attention weights [batch, heads, seq, seq]
        
    Returns:
        Laplacian diagonal [batch, heads, seq]
    """
    # Out-degree: sum over columns (what each position attends to)
    degree = attention.sum(dim=-1)  # [batch, heads, seq]
    
    # Diagonal of attention
    attn_diag = extract_attention_diagonal(attention)  # [batch, heads, seq]
    
    # Laplacian diagonal: L_ii = D_ii - A_ii
    laplacian_diag = degree - attn_diag
    
    return laplacian_diag


def compute_attention_entropy(attention: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute row-wise entropy of attention distribution.
    
    Args:
        attention: Attention weights [batch, heads, seq, seq]
        eps: Small value for numerical stability
        
    Returns:
        Entropy for each position [batch, heads, seq]
    """
    attention = attention.clamp(min=eps)
    entropy = -torch.sum(attention * torch.log(attention), dim=-1)
    return entropy


def stack_layer_attentions(
    attentions: Tuple[torch.Tensor, ...],
    layer_indices: List[int],
) -> torch.Tensor:
    """Stack attention tensors from selected layers.
    
    Args:
        attentions: Tuple of attention tensors, one per layer
        layer_indices: Which layers to select
        
    Returns:
        Stacked attention [n_layers, batch, heads, seq, seq]
    """
    selected = [attentions[i] for i in layer_indices if i < len(attentions)]
    return torch.stack(selected, dim=0)


# ==============================================================================
# Hidden State Feature Extraction
# ==============================================================================

def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    method: str = "last_token",
) -> torch.Tensor:
    """Pool hidden states to single vector.
    
    Args:
        hidden_states: Hidden states [batch, seq, hidden]
        attention_mask: Attention mask [batch, seq]
        method: Pooling method (last_token, mean, max, first_token)
        
    Returns:
        Pooled hidden states [batch, hidden]
    """
    if method == "last_token":
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=-1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, seq_lens]
        return hidden_states[:, -1, :]
    
    elif method == "first_token":
        return hidden_states[:, 0, :]
    
    elif method == "mean":
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hidden_states.mean(dim=1)
    
    elif method == "max":
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states.masked_fill(mask == 0, float('-inf'))
        return hidden_states.max(dim=1)[0]
    
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def stack_layer_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_indices: List[int],
    attention_mask: Optional[torch.Tensor] = None,
    pooling: str = "last_token",
) -> torch.Tensor:
    """Stack and pool hidden states from selected layers.
    
    Args:
        hidden_states: Tuple of hidden state tensors, one per layer
        layer_indices: Which layers to select
        attention_mask: Attention mask for pooling
        pooling: Pooling method
        
    Returns:
        Stacked pooled hidden states [n_layers, batch, hidden]
    """
    pooled = []
    for i in layer_indices:
        if i < len(hidden_states):
            p = pool_hidden_states(hidden_states[i], attention_mask, pooling)
            pooled.append(p)
    
    return torch.stack(pooled, dim=0)


# ==============================================================================
# Token Probability Feature Extraction
# ==============================================================================

def compute_token_probs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute probability of target tokens.
    
    Args:
        logits: Model logits [batch, seq, vocab]
        target_ids: Target token ids [batch, seq]
        
    Returns:
        Token probabilities [batch, seq]
    """
    # Shift: logits[i] predicts target[i+1]
    shift_logits = logits[:, :-1, :]
    shift_targets = target_ids[:, 1:]
    
    # Compute probabilities
    probs = F.softmax(shift_logits, dim=-1)
    
    # Get probability of target tokens
    target_probs = torch.gather(probs, dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
    
    return target_probs


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of probability distribution at each position.
    
    Args:
        logits: Model logits [batch, seq, vocab]
        
    Returns:
        Entropy [batch, seq]
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def compute_top_k_probs(
    logits: torch.Tensor,
    k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get top-k probabilities and indices.
    
    Args:
        logits: Model logits [batch, seq, vocab]
        k: Number of top tokens
        
    Returns:
        (top_k_probs, top_k_indices) each [batch, seq, k]
    """
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
    return top_k_probs, top_k_indices


# ==============================================================================
# Main Feature Extractor
# ==============================================================================

class FeatureExtractor:
    """Extract features from model outputs.
    
    Supports both teacher forcing and generation modes.
    """
    
    def __init__(self, model: LoadedModel, config: FeaturesConfig):
        """Initialize feature extractor.
        
        Args:
            model: Loaded model instance
            config: Feature extraction configuration
        """
        self.model = model
        self.config = config
        
        # Resolve layer indices
        n_layers = model.num_layers
        self.attn_layers = parse_layers(config.attention_layers, n_layers) if config.attention_enabled else []
        self.hidden_layers = parse_layers(config.hidden_states_layers, n_layers) if config.hidden_states_enabled else []
        
        logger.info(f"Extractor initialized: attn_layers={len(self.attn_layers)}, hidden_layers={len(self.hidden_layers)}")
    
    def extract(self, sample: Sample) -> ExtractedFeatures:
        """Extract features from a single sample.
        
        Args:
            sample: Sample with prompt and response
            
        Returns:
            ExtractedFeatures instance
        """
        if self.config.mode == "teacher_forcing":
            return self._extract_teacher_forcing(sample)
        elif self.config.mode == "generation":
            return self._extract_generation(sample)
        else:
            raise FeatureError(f"Unknown extraction mode: {self.config.mode}")
    
    def _extract_teacher_forcing(self, sample: Sample) -> ExtractedFeatures:
        """Extract features using teacher forcing.
        
        Concatenate prompt + response, run forward pass, extract features.
        """
        # Tokenize
        prompt_ids = self.model.encode(sample.prompt, add_special_tokens=True)
        response_ids = self.model.encode(sample.response, add_special_tokens=False)
        
        prompt_len = prompt_ids.size(1)
        response_len = response_ids.size(1)
        
        # Concatenate
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # Truncate if needed
        if input_ids.size(1) > self.config.max_length:
            input_ids = input_ids[:, :self.config.max_length]
            response_len = max(0, self.config.max_length - prompt_len)
        
        # Forward pass
        outputs = self.model.forward(
            input_ids=input_ids,
            output_attentions=self.config.attention_enabled,
            output_hidden_states=self.config.hidden_states_enabled,
        )
        
        # Initialize feature containers
        features = ExtractedFeatures(
            sample_id=sample.id,
            prompt_len=prompt_len,
            response_len=response_len,
            label=sample.label,
            layers=self.attn_layers or self.hidden_layers,
            model_name=self.model.config.name,
            mode=ExtractionMode.TEACHER_FORCING,
        )
        
        # Extract attention features
        if self.config.attention_enabled and "attentions" in outputs:
            attentions = outputs["attentions"]
            
            # Stack selected layers: [n_layers, batch, heads, seq, seq]
            stacked = stack_layer_attentions(attentions, self.attn_layers)
            
            # Extract diagonal (lapeigvals style): [n_layers, batch, heads, seq]
            features.attn_diags = extract_attention_diagonal(stacked).squeeze(1)
            
            # Compute Laplacian diagonal
            features.laplacian_diags = compute_laplacian_diagonal(stacked).squeeze(1)
            
            # Compute attention entropy
            features.attn_entropy = compute_attention_entropy(stacked).squeeze(1)
        
        # Extract hidden state features
        if self.config.hidden_states_enabled and "hidden_states" in outputs:
            hidden_states = outputs["hidden_states"]
            
            # Stack and pool: [n_layers, hidden]
            features.hidden_states = stack_layer_hidden_states(
                hidden_states,
                self.hidden_layers,
                pooling=self.config.hidden_states_pooling,
            ).squeeze(1)
        
        # Extract token probability features
        if self.config.token_probs_enabled:
            logits = outputs["logits"]
            
            # Only compute for response portion
            if response_len > 0:
                response_start = prompt_len
                
                if logits.size(1) > response_start:
                    features.token_probs = compute_token_probs(
                        logits[:, response_start-1:, :],
                        input_ids[:, response_start-1:],
                    ).squeeze(0)
                    
                    features.token_entropy = compute_token_entropy(
                        logits[:, response_start:, :]
                    ).squeeze(0)
                    
                    if self.config.token_probs_top_k > 0:
                        top_k_probs, top_k_indices = compute_top_k_probs(
                            logits[:, response_start:, :],
                            k=self.config.token_probs_top_k,
                        )
                        features.top_k_probs = top_k_probs.squeeze(0)
                        features.top_k_indices = top_k_indices.squeeze(0)
                    
                    # Compute perplexity
                    if features.token_probs is not None and features.token_probs.numel() > 0:
                        features.perplexity = torch.exp(
                            -torch.log(features.token_probs.clamp(min=1e-10)).mean()
                        ).item()
        
        return features
    
    def _extract_generation(self, sample: Sample) -> ExtractedFeatures:
        """Extract features during generation.
        
        Generate response from prompt, then extract features with teacher forcing.
        """
        # Tokenize prompt
        prompt_ids = self.model.encode(sample.prompt, add_special_tokens=True)
        prompt_len = prompt_ids.size(1)
        
        # Generate
        gen_outputs = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=self.config.max_length - prompt_len,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        generated_ids = gen_outputs["generated_ids"]
        response_ids = generated_ids[:, prompt_len:]
        
        # Decode generated response
        generated_response = self.model.decode(response_ids[0])
        
        # Now do teacher forcing on the generated sequence
        sample_with_gen = Sample(
            id=sample.id,
            prompt=sample.prompt,
            response=generated_response,
            reference=sample.reference,
            label=sample.label,
            task_type=sample.task_type,
            metadata={**sample.metadata, "generated": True},
        )
        
        features = self._extract_teacher_forcing(sample_with_gen)
        features.mode = ExtractionMode.GENERATION
        features.metadata["generated_response"] = generated_response
        
        return features
    
    def extract_batch(
        self,
        samples: List[Sample],
        show_progress: bool = True,
    ) -> List[ExtractedFeatures]:
        """Extract features from multiple samples.
        
        Args:
            samples: List of samples
            show_progress: Whether to show progress
            
        Returns:
            List of ExtractedFeatures
        """
        features_list = []
        
        if show_progress:
            with Progress(len(samples), desc="Extracting features") as pbar:
                for sample in samples:
                    try:
                        features = self.extract(sample)
                        features_list.append(features)
                    except Exception as e:
                        logger.warning(f"Failed to extract features for {sample.id}: {e}")
                    pbar.update()
        else:
            for sample in samples:
                try:
                    features = self.extract(sample)
                    features_list.append(features)
                except Exception as e:
                    logger.warning(f"Failed to extract features for {sample.id}: {e}")
        
        return features_list


def create_extractor(model: LoadedModel, config: FeaturesConfig) -> FeatureExtractor:
    """Create feature extractor.
    
    Args:
        model: Loaded model
        config: Feature extraction config
        
    Returns:
        FeatureExtractor instance
    """
    return FeatureExtractor(model, config)
