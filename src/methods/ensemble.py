"""Ensemble methods for hallucination detection.

Combines multiple detection methods for improved performance:
- Weighted voting
- Stacking
- Feature concatenation
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
import pickle
import numpy as np

from src.core import ExtractedFeatures, Prediction, MethodConfig, MethodError, METHODS
from .base import BaseMethod, create_method

logger = logging.getLogger(__name__)


@METHODS.register("ensemble", aliases=["combined"])
class EnsembleMethod(BaseMethod):
    """Ensemble of multiple detection methods.
    
    Modes:
    - voting: Weighted average of predictions
    - stacking: Train meta-classifier on base method outputs
    - concat: Concatenate features from all methods
    """
    
    def __init__(
        self,
        config: Optional[MethodConfig] = None,
        methods: Optional[List[BaseMethod]] = None,
    ):
        super().__init__(config)
        
        params = self.config.params or {}
        
        # Ensemble configuration
        self.ensemble_mode = params.get("mode", "voting")  # voting, stacking, concat
        self.method_names = params.get("methods", ["lapeigvals", "lookback_lens", "entropy"])
        self.weights = params.get("weights", None)  # For voting mode
        
        # Initialize base methods
        if methods is not None:
            self.methods = methods
        else:
            self.methods = []
            for name in self.method_names:
                try:
                    method = create_method(name)
                    self.methods.append(method)
                except Exception as e:
                    logger.warning(f"Failed to create method {name}: {e}")
        
        if len(self.methods) == 0:
            raise MethodError("No valid methods for ensemble")
        
        # Set default weights
        if self.weights is None:
            self.weights = [1.0 / len(self.methods)] * len(self.methods)
        
        logger.info(f"Ensemble with {len(self.methods)} methods: {[m.__class__.__name__ for m in self.methods]}")
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract features based on ensemble mode.
        
        For concat mode: concatenate features from all methods
        For other modes: return empty (predictions are combined differently)
        """
        if self.ensemble_mode == "concat":
            all_features = []
            for method in self.methods:
                try:
                    feat = method.extract_method_features(features)
                    all_features.append(feat)
                except Exception as e:
                    logger.debug(f"Method {method.__class__.__name__} failed: {e}")
            
            if len(all_features) == 0:
                raise MethodError("No features extracted from any method")
            
            return np.concatenate(all_features)
        
        # For voting/stacking, we don't use this
        return np.array([0.0])
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """Train ensemble.
        
        For concat mode: Train single classifier on concatenated features
        For voting mode: Train each method independently
        For stacking mode: Train base methods, then meta-classifier
        """
        if labels is None:
            labels = [f.label for f in features_list]
        
        metrics = {}
        
        if self.ensemble_mode == "concat":
            # Standard training with concatenated features
            return super().fit(features_list, labels, cv)
        
        elif self.ensemble_mode == "voting":
            # Train each method independently
            for i, method in enumerate(self.methods):
                try:
                    method_metrics = method.fit(features_list, labels, cv=False)
                    metrics[f"method_{i}"] = method_metrics
                    logger.info(f"Trained {method.__class__.__name__}")
                except Exception as e:
                    logger.warning(f"Failed to train {method.__class__.__name__}: {e}")
            
            self.is_fitted = True
        
        elif self.ensemble_mode == "stacking":
            # Train base methods
            for method in self.methods:
                try:
                    method.fit(features_list, labels, cv=False)
                except Exception as e:
                    logger.warning(f"Failed to train {method.__class__.__name__}: {e}")
            
            # Get base predictions
            X_meta = []
            y_meta = []
            
            for i, feat in enumerate(features_list):
                preds = []
                for method in self.methods:
                    try:
                        pred = method.predict(feat)
                        preds.append(pred.score)
                    except Exception:
                        preds.append(0.5)
                
                if len(preds) > 0:
                    X_meta.append(preds)
                    y_meta.append(labels[i] if labels[i] is not None else feat.label)
            
            # Train meta-classifier
            X_meta = np.array(X_meta)
            y_meta = np.array(y_meta)
            
            X_scaled = self.scaler.fit_transform(X_meta)
            self.classifier = self._create_classifier()
            self.classifier.fit(X_scaled, y_meta)
            self.is_fitted = True
            
            metrics["n_base_methods"] = len(self.methods)
        
        metrics["ensemble_mode"] = self.ensemble_mode
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        """Predict using ensemble.
        
        Args:
            features: Extracted features
            
        Returns:
            Combined prediction
        """
        if not self.is_fitted:
            raise MethodError("Ensemble not fitted")
        
        if self.ensemble_mode == "concat":
            return super().predict(features)
        
        elif self.ensemble_mode == "voting":
            # Weighted average of predictions
            scores = []
            weights_used = []
            
            for method, weight in zip(self.methods, self.weights):
                try:
                    pred = method.predict(features)
                    scores.append(pred.score)
                    weights_used.append(weight)
                except Exception as e:
                    logger.debug(f"Method {method.__class__.__name__} failed: {e}")
            
            if len(scores) == 0:
                raise MethodError("All methods failed to predict")
            
            # Normalize weights
            weights_used = np.array(weights_used)
            weights_used = weights_used / weights_used.sum()
            
            # Weighted average
            final_score = float(np.average(scores, weights=weights_used))
            
            return Prediction(
                sample_id=features.sample_id,
                score=final_score,
                label=1 if final_score > 0.5 else 0,
                confidence=abs(final_score - 0.5) * 2,
            )
        
        elif self.ensemble_mode == "stacking":
            # Get base predictions
            base_preds = []
            for method in self.methods:
                try:
                    pred = method.predict(features)
                    base_preds.append(pred.score)
                except Exception:
                    base_preds.append(0.5)
            
            # Meta-classifier prediction
            X = np.array(base_preds).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            proba = self.classifier.predict_proba(X_scaled)[0]
            score = float(proba[1]) if len(proba) > 1 else float(proba[0])
            
            return Prediction(
                sample_id=features.sample_id,
                score=score,
                label=1 if score > 0.5 else 0,
                confidence=abs(score - 0.5) * 2,
            )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save ensemble to file."""
        if not self.is_fitted:
            raise MethodError("Cannot save unfitted ensemble")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "ensemble_mode": self.ensemble_mode,
            "method_names": self.method_names,
            "weights": self.weights,
            "methods": self.methods,
            "classifier": self.classifier,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved ensemble to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load ensemble from file."""
        path = Path(path)
        if not path.exists():
            raise MethodError(f"Ensemble file not found: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.ensemble_mode = state["ensemble_mode"]
        self.method_names = state["method_names"]
        self.weights = state["weights"]
        self.methods = state["methods"]
        self.classifier = state["classifier"]
        self.scaler = state["scaler"]
        self.is_fitted = state["is_fitted"]
        
        logger.info(f"Loaded ensemble from {path}")


@METHODS.register("auto_ensemble")
class AutoEnsembleMethod(EnsembleMethod):
    """Automatically select and weight methods based on validation performance."""
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """Train with automatic method selection and weighting."""
        if labels is None:
            labels = [f.label for f in features_list]
        
        # Train each method and get CV scores
        method_scores = []
        
        for i, method in enumerate(self.methods):
            try:
                metrics = method.fit(features_list, labels, cv=True)
                cv_score = metrics.get("cv_auroc_mean", 0.5)
                method_scores.append(cv_score)
                logger.info(f"{method.__class__.__name__} CV AUROC: {cv_score:.4f}")
            except Exception as e:
                logger.warning(f"Failed to train {method.__class__.__name__}: {e}")
                method_scores.append(0.0)
        
        # Weight by performance (softmax of scores)
        scores = np.array(method_scores)
        scores = np.maximum(scores - 0.5, 0)  # Only positive contribution above 0.5
        
        if scores.sum() > 0:
            self.weights = list(scores / scores.sum())
        else:
            self.weights = [1.0 / len(self.methods)] * len(self.methods)
        
        logger.info(f"Auto weights: {self.weights}")
        
        self.is_fitted = True
        
        return {
            "method_scores": dict(zip(self.method_names, method_scores)),
            "weights": dict(zip(self.method_names, self.weights)),
        }
