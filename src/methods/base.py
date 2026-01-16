"""Base class for hallucination detection methods.

All detection methods inherit from BaseMethod and implement:
- fit(): Train the method on labeled data
- predict(): Predict hallucination scores
- save()/load(): Persistence
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.core import ExtractedFeatures, Prediction, EvalMetrics, MethodConfig, MethodError, METHODS

logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    """Abstract base class for detection methods.
    
    All methods must implement:
    - extract_method_features(): Convert ExtractedFeatures to method-specific features
    - fit(): Train classifier on features
    - predict(): Predict hallucination probability
    
    Token-level training:
    - Override supports_token_level to return True if the method can use token labels
    - The fit() method receives token_labels when level is 'token' or 'both'
    """
    
    # Class attribute: whether this method supports token-level training
    supports_token_level: bool = False
    
    def __init__(self, config: Optional[MethodConfig] = None):
        """Initialize method.
        
        Args:
            config: Method configuration
        """
        self.config = config or MethodConfig()
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.is_token_fitted = False
        self._feature_dim = None
    
    @abstractmethod
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract method-specific features from ExtractedFeatures.
        
        Args:
            features: Extracted features from model
            
        Returns:
            Feature vector for this method [feature_dim]
        """
        pass
    
    def _create_classifier(self):
        """Create classifier based on config."""
        clf_type = self.config.classifier.lower()
        params = self.config.params or {}
        
        if clf_type == "logistic":
            return LogisticRegression(
                max_iter=params.get("max_iter", 1000),
                C=params.get("C", 1.0),
                class_weight=params.get("class_weight", "balanced"),
                random_state=self.config.random_seed,
            )
        elif clf_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                class_weight=params.get("class_weight", "balanced"),
                random_state=self.config.random_seed,
            )
        elif clf_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=params.get("hidden_layer_sizes", (64, 32)),
                max_iter=params.get("max_iter", 500),
                random_state=self.config.random_seed,
            )
        else:
            raise MethodError(f"Unknown classifier type: {clf_type}")
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """Train the method on labeled data.
        
        Args:
            features_list: List of extracted features
            labels: Labels (if not in features). 0=correct, 1=hallucinated
            cv: Whether to run cross-validation
            
        Returns:
            Training metrics (cv_score if cv=True)
        """
        # Extract method-specific features
        X = []
        y = []
        
        n_total = len(features_list)
        log_interval = max(1, n_total // 10)  # Log every 10%
        
        for i, feat in enumerate(features_list):
            if i % log_interval == 0:
                logger.info(f"Extracting features: {i}/{n_total} ({100*i/n_total:.0f}%)")
            try:
                x = self.extract_method_features(feat)
                if x is not None and not np.any(np.isnan(x)):
                    X.append(x)
                    label = labels[i] if labels else feat.label
                    if label is not None:
                        y.append(label)
                    else:
                        logger.warning(f"No label for sample {feat.sample_id}")
            except Exception as e:
                logger.warning(f"Failed to extract features for {feat.sample_id}: {e}")
        
        if len(X) == 0:
            raise MethodError("No valid features extracted")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training on {len(X)} samples, feature dim={X.shape[1]}")
        self._feature_dim = X.shape[1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and train classifier
        self.classifier = self._create_classifier()
        
        metrics = {}
        
        # Cross-validation
        if cv and self.config.cv_folds > 1:
            cv_scores = cross_val_score(
                self.classifier, X_scaled, y,
                cv=min(self.config.cv_folds, len(X)),
                scoring="roc_auc",
            )
            metrics["cv_auroc_mean"] = float(cv_scores.mean())
            metrics["cv_auroc_std"] = float(cv_scores.std())
            logger.info(f"CV AUROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Final fit on all data
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        
        metrics["n_samples"] = len(X)
        metrics["n_positive"] = int(y.sum())
        metrics["n_negative"] = int(len(y) - y.sum())
        
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        """Predict hallucination probability for a single sample.
        
        Args:
            features: Extracted features
            
        Returns:
            Prediction with score and label
        """
        if not self.is_fitted:
            raise MethodError("Method not fitted. Call fit() first.")
        
        x = self.extract_method_features(features)
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        
        # Get probability of hallucination (class 1)
        proba = self.classifier.predict_proba(x_scaled)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return Prediction(
            sample_id=features.sample_id,
            score=score,
            label=1 if score > 0.5 else 0,
            confidence=abs(score - 0.5) * 2,
        )
    
    def predict_batch(self, features_list: List[ExtractedFeatures]) -> List[Prediction]:
        """Predict for multiple samples.
        
        Args:
            features_list: List of extracted features
            
        Returns:
            List of predictions
        """
        predictions = []
        for feat in features_list:
            try:
                pred = self.predict(feat)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict for {feat.sample_id}: {e}")
        return predictions
    
    def save(self, path: Union[str, Path]) -> None:
        """Save method to file.
        
        Args:
            path: Save path
        """
        if not self.is_fitted:
            raise MethodError("Cannot save unfitted method")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "classifier": self.classifier,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "is_token_fitted": getattr(self, 'is_token_fitted', False),
            "feature_dim": self._feature_dim,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved method to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load method from file.
        
        Args:
            path: Load path
        """
        path = Path(path)
        if not path.exists():
            raise MethodError(f"Method file not found: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.classifier = state["classifier"]
        self.scaler = state["scaler"]
        self.is_fitted = state["is_fitted"]
        self.is_token_fitted = state.get("is_token_fitted", False)
        self._feature_dim = state["feature_dim"]
        
        logger.info(f"Loaded method from {path}")


def create_method(name: str, config: Optional[MethodConfig] = None) -> BaseMethod:
    """Create method by name.
    
    Args:
        name: Method name
        config: Method configuration
        
    Returns:
        Method instance
    """
    if config is None:
        config = MethodConfig(name=name)
    return METHODS.create(name, config=config)
