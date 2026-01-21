"""LapEigvals: Laplacian Eigenvalue-based Hallucination Detection.

Based on the EMNLP 2025 paper.
Uses eigenvalues of attention Laplacian matrix to detect hallucinations.

Key insight: Hallucinated responses show different attention patterns,
which manifest as different Laplacian eigenvalue distributions.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
import logging
import numpy as np
import torch

from src.core import ExtractedFeatures, MethodConfig, METHODS
from .base import BaseMethod

logger = logging.getLogger(__name__)


def compute_laplacian_eigenvalues(
    attention: torch.Tensor,
    top_k: int = 10,
) -> torch.Tensor:
    """Compute top-k eigenvalues of attention Laplacian.
    
    For attention matrix A:
    - Degree matrix D_ii = sum_j A_ij
    - Laplacian L = D - A
    - Compute eigenvalues of L
    
    Args:
        attention: Attention matrix [heads, seq, seq] or [seq, seq]
        top_k: Number of top eigenvalues to return
        
    Returns:
        Top-k eigenvalues (sorted descending) [k] or [heads, k]
    """
    if attention.dim() == 2:
        attention = attention.unsqueeze(0)  # Add head dim
    
    n_heads, seq_len, _ = attention.shape
    eigenvalues_list = []
    
    for h in range(n_heads):
        A = attention[h].float()
        
        # Compute degree matrix (out-degree)
        D = torch.diag(A.sum(dim=1))
        
        # Laplacian
        L = D - A
        
        # Compute eigenvalues
        try:
            eigvals = torch.linalg.eigvalsh(L)
            # Sort descending and take top-k
            eigvals_sorted = torch.sort(eigvals, descending=True)[0]
            top_eigvals = eigvals_sorted[:top_k]
            
            # Pad if needed
            if len(top_eigvals) < top_k:
                top_eigvals = torch.cat([
                    top_eigvals,
                    torch.zeros(top_k - len(top_eigvals), device=eigvals.device)
                ])
            
            eigenvalues_list.append(top_eigvals)
        except Exception:
            eigenvalues_list.append(torch.zeros(top_k, device=attention.device))
    
    return torch.stack(eigenvalues_list)  # [heads, k]


def compute_laplacian_features_from_diagonal(
    laplacian_diag: torch.Tensor,
    response_start: int,
    response_len: int,
) -> np.ndarray:
    """Compute features from Laplacian diagonal (efficient version).
    
    When we only have diagonal, we use statistics of the diagonal values.
    
    Args:
        laplacian_diag: Laplacian diagonal [n_layers, n_heads, seq_len]
        response_start: Start index of response tokens
        response_len: Length of response
        
    Returns:
        Feature vector
    """
    # Focus on response portion
    if response_len > 0 and response_start + response_len <= laplacian_diag.shape[-1]:
        response_diag = laplacian_diag[:, :, response_start:response_start + response_len]
    else:
        response_diag = laplacian_diag
    
    # Compute statistics per layer and head - vectorized
    n_layers, n_heads = response_diag.shape[:2]
    response_diag = response_diag.float()
    
    if response_diag.shape[-1] == 0:
        return np.zeros(n_layers * n_heads * 7, dtype=np.float32)
    
    # Convert to numpy for percentile computation
    response_np = response_diag.cpu().numpy()
    
    # Vectorized statistics [n_layers, n_heads]
    diag_mean = np.mean(response_np, axis=-1)
    diag_std = np.std(response_np, axis=-1)
    diag_max = np.max(response_np, axis=-1)
    diag_min = np.min(response_np, axis=-1)
    diag_median = np.median(response_np, axis=-1)
    diag_q1 = np.percentile(response_np, 25, axis=-1)
    diag_q3 = np.percentile(response_np, 75, axis=-1)
    
    # Stack and flatten [n_layers, n_heads, 7] -> [n_layers * n_heads * 7]
    features = np.stack([diag_mean, diag_std, diag_max, diag_min, diag_median, diag_q1, diag_q3], axis=-1)
    
    return features.flatten().astype(np.float32)


@METHODS.register("lapeigvals", aliases=["lap_eigvals", "laplacian_eigenvalues"])
class LapEigvalsMethod(BaseMethod):
    """Laplacian Eigenvalue-based hallucination detection.
    
    Features:
    - Eigenvalues of attention Laplacian (if full attention available)
    - Statistics of Laplacian diagonal (efficient mode)
    - Per-layer, per-head analysis
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        # Method-specific params
        params = self.config.params or {}
        self.top_k_eigenvalues = params.get("top_k_eigenvalues", 10)
        self.use_diagonal_only = params.get("use_diagonal_only", True)
        self.aggregate_layers = params.get("aggregate_layers", "all")  # all, last, mean
        self.aggregate_heads = params.get("aggregate_heads", "all")    # all, mean
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract Laplacian-based features.
        
        Args:
            features: Extracted features with laplacian_diags
            
        Returns:
            Feature vector
        """
        if features.laplacian_diags is None:
            raise ValueError("LapEigvals requires laplacian_diags")
        
        laplacian_diag = features.laplacian_diags
        
        # Convert to tensor if needed
        if isinstance(laplacian_diag, np.ndarray):
            laplacian_diag = torch.from_numpy(laplacian_diag)
        
        # Compute features from diagonal
        feat_vec = compute_laplacian_features_from_diagonal(
            laplacian_diag,
            response_start=features.prompt_len,
            response_len=features.response_len,
        )
        
        # Add attention diagonal features if available
        if features.attn_diags is not None:
            attn_diag = features.attn_diags
            if isinstance(attn_diag, np.ndarray):
                attn_diag = torch.from_numpy(attn_diag)
            
            # Same statistics for attention diagonal
            attn_features = self._compute_diagonal_stats(
                attn_diag,
                features.prompt_len,
                features.response_len,
            )
            feat_vec = np.concatenate([feat_vec, attn_features])
        
        # Add entropy features if available
        if features.attn_entropy is not None:
            entropy = features.attn_entropy
            if isinstance(entropy, np.ndarray):
                entropy = torch.from_numpy(entropy)
            
            entropy_features = self._compute_diagonal_stats(
                entropy,
                features.prompt_len,
                features.response_len,
            )
            feat_vec = np.concatenate([feat_vec, entropy_features])
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(feat_vec)):
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat_vec
    
    def _compute_diagonal_stats(
        self,
        diag: torch.Tensor,
        response_start: int,
        response_len: int,
    ) -> np.ndarray:
        """Compute statistics from diagonal tensor - vectorized."""
        # Focus on response portion
        if response_len > 0 and response_start + response_len <= diag.shape[-1]:
            response_diag = diag[:, :, response_start:response_start + response_len]
        else:
            response_diag = diag
        
        n_layers, n_heads = response_diag.shape[:2]
        
        # Aggregate based on config
        if self.aggregate_layers == "last":
            response_diag = response_diag[-1:, :, :]
        elif self.aggregate_layers == "mean":
            response_diag = response_diag.mean(dim=0, keepdim=True)
        
        if self.aggregate_heads == "mean":
            response_diag = response_diag.mean(dim=1, keepdim=True)
        
        # Check for empty response
        if response_diag.shape[-1] == 0:
            n_out_layers = response_diag.shape[0]
            n_out_heads = response_diag.shape[1]
            return np.zeros(n_out_layers * n_out_heads * 4, dtype=np.float32)
        
        # Convert to numpy for vectorized computation
        response_np = response_diag.float().cpu().numpy()
        
        # Vectorized statistics [n_layers, n_heads]
        diag_mean = np.mean(response_np, axis=-1)
        diag_std = np.std(response_np, axis=-1)
        diag_max = np.max(response_np, axis=-1)
        diag_min = np.min(response_np, axis=-1)
        
        # Stack and flatten
        features = np.stack([diag_mean, diag_std, diag_max, diag_min], axis=-1)
        
        return features.flatten().astype(np.float32)


@METHODS.register("lapeigvals_full", aliases=["lap_eigvals_full"])
class LapEigvalsFullMethod(LapEigvalsMethod):
    """LapEigvals with full eigenvalue computation.
    
    Requires full attention matrices (not just diagonals).
    More accurate but more expensive.
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        self.use_diagonal_only = False
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract features using full eigenvalue computation.
        
        Note: This requires storing full attention matrices,
        which is memory-intensive.
        """
        # If we have full attention, compute eigenvalues
        if hasattr(features, 'full_attention') and features.full_attention is not None:
            return self._extract_from_full_attention(features)
        
        # Fall back to diagonal-based features
        return super().extract_method_features(features)
    
    def _extract_from_full_attention(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract eigenvalue features from full attention."""
        full_attn = features.full_attention  # [n_layers, n_heads, seq, seq]
        
        all_eigvals = []
        
        for layer in range(full_attn.shape[0]):
            layer_attn = full_attn[layer]  # [n_heads, seq, seq]
            eigvals = compute_laplacian_eigenvalues(layer_attn, top_k=self.top_k_eigenvalues)
            all_eigvals.append(eigvals.flatten().float().cpu().numpy())
        
        feat_vec = np.concatenate(all_eigvals)
        
        # Add statistics
        feat_vec = np.concatenate([
            feat_vec,
            np.array([np.mean(feat_vec), np.std(feat_vec), np.max(feat_vec), np.min(feat_vec)])
        ])
        
        return feat_vec.astype(np.float32)
