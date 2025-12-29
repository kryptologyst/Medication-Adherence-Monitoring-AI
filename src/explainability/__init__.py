"""
Explainability and uncertainty quantification for medication adherence monitoring.

This module provides SHAP explanations, uncertainty quantification,
and safety measures for adherence prediction models.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class SHAPExplainer:
    """SHAP-based explainability for adherence models."""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain.
            feature_names: List of feature names.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
    
    def fit_explainer(
        self, 
        X_background: np.ndarray, 
        X_explain: Optional[np.ndarray] = None,
        explainer_type: str = 'auto'
    ) -> None:
        """Fit SHAP explainer to background data.
        
        Args:
            X_background: Background data for explainer.
            X_explain: Data to explain (optional).
            explainer_type: Type of SHAP explainer ('auto', 'tree', 'linear', 'kernel').
        """
        if explainer_type == 'auto':
            # Auto-detect explainer type based on model
            if hasattr(self.model, 'predict_proba'):
                if hasattr(self.model, 'feature_importances_'):
                    explainer_type = 'tree'
                else:
                    explainer_type = 'kernel'
            else:
                explainer_type = 'kernel'
        
        if explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_background)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        # Calculate SHAP values
        if X_explain is not None:
            self.shap_values = self.explainer.shap_values(X_explain)
    
    def explain_instance(
        self, 
        instance: np.ndarray, 
        class_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Explain a single instance.
        
        Args:
            instance: Instance to explain.
            class_idx: Class index to explain (optional).
            
        Returns:
            Dict[str, Any]: Explanation results.
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        # Calculate SHAP values for this instance
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        
        if isinstance(shap_values, list):
            # Multi-class case
            if class_idx is None:
                class_idx = 0  # Default to first class
            shap_values = shap_values[class_idx]
        
        # Create explanation dictionary
        explanation = {
            'shap_values': shap_values[0],  # Remove batch dimension
            'feature_names': self.feature_names,
            'feature_importance': np.abs(shap_values[0]),
            'prediction': self.model.predict_proba(instance.reshape(1, -1))[0]
        }
        
        return explanation
    
    def explain_dataset(
        self, 
        X: np.ndarray, 
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Explain a dataset.
        
        Args:
            X: Dataset to explain.
            max_samples: Maximum number of samples to explain.
            
        Returns:
            Dict[str, Any]: Dataset explanation results.
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        # Limit samples for efficiency
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            # Multi-class case - use first class
            shap_values = shap_values[0]
        
        # Calculate summary statistics
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        std_shap_values = np.std(shap_values, axis=0)
        
        explanation = {
            'shap_values': shap_values,
            'mean_importance': mean_shap_values,
            'std_importance': std_shap_values,
            'feature_names': self.feature_names,
            'n_samples': len(X_sample)
        }
        
        return explanation
    
    def plot_feature_importance(
        self, 
        explanation: Dict[str, Any], 
        top_k: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance from SHAP explanation.
        
        Args:
            explanation: SHAP explanation results.
            top_k: Number of top features to show.
            save_path: Optional path to save the plot.
        """
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        
        if shap_values.ndim > 1:
            # Multiple samples - use mean absolute values
            importance = np.mean(np.abs(shap_values), axis=0)
        else:
            # Single sample
            importance = np.abs(shap_values)
        
        # Get top features
        top_indices = np.argsort(importance)[::-1][:top_k]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importance, y=top_features)
        plt.title(f'Top {top_k} Feature Importance (SHAP)')
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_waterfall(
        self, 
        explanation: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Plot SHAP waterfall plot for single instance.
        
        Args:
            explanation: SHAP explanation results.
            save_path: Optional path to save the plot.
        """
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        
        if shap_values.ndim > 1:
            shap_values = shap_values[0]  # Take first sample
        
        # Sort features by SHAP value
        sorted_indices = np.argsort(shap_values)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_values = shap_values[sorted_indices]
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        
        # Calculate cumulative values
        cumulative = np.cumsum(sorted_values)
        
        # Plot bars
        colors = ['red' if v < 0 else 'blue' for v in sorted_values]
        plt.barh(range(len(sorted_features)), sorted_values, color=colors, alpha=0.7)
        
        # Add cumulative line
        plt.plot(cumulative, range(len(sorted_features)), 'k-', linewidth=2, alpha=0.5)
        
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('SHAP Value')
        plt.title('SHAP Waterfall Plot')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class UncertaintyQuantifier:
    """Quantify model uncertainty for adherence prediction."""
    
    def __init__(self, model: Any, device: torch.device):
        """Initialize uncertainty quantifier.
        
        Args:
            model: Trained model.
            device: Device for computation.
        """
        self.model = model
        self.device = device
        self.uncertainty_metrics = {}
    
    def mc_dropout_prediction(
        self, 
        X: torch.Tensor, 
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout prediction for uncertainty estimation.
        
        Args:
            X: Input features.
            n_samples: Number of Monte Carlo samples.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean predictions and uncertainties.
        """
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Enable dropout during inference
                for module in self.model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                
                pred = self.model(X)
                predictions.append(pred)
        
        # Disable dropout
        self.model.eval()
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [n_samples, batch_size, n_classes]
        
        # Calculate mean and variance
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        return mean_pred, var_pred
    
    def ensemble_prediction(
        self, 
        models: List[Any], 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble prediction for uncertainty estimation.
        
        Args:
            models: List of trained models.
            X: Input features.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean predictions and uncertainties.
        """
        predictions = []
        
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
                # Convert to probabilities if needed
                pred = np.column_stack([1 - pred, pred])
            
            predictions.append(pred)
        
        predictions = np.array(predictions)  # [n_models, batch_size, n_classes]
        
        # Calculate mean and variance
        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)
        
        return mean_pred, var_pred
    
    def calculate_uncertainty_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred_mean: np.ndarray, 
        y_pred_var: np.ndarray
    ) -> Dict[str, float]:
        """Calculate uncertainty metrics.
        
        Args:
            y_true: True labels.
            y_pred_mean: Mean predictions.
            y_pred_var: Prediction variances.
            
        Returns:
            Dict[str, float]: Uncertainty metrics.
        """
        metrics = {}
        
        # Prediction uncertainty (variance)
        metrics['mean_variance'] = np.mean(y_pred_var)
        metrics['max_variance'] = np.max(y_pred_var)
        
        # Confidence intervals
        confidence_levels = [0.5, 0.8, 0.9, 0.95]
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            z_score = 1.96 if conf_level == 0.95 else 1.28  # Approximate
            
            # Calculate confidence interval width
            ci_width = 2 * z_score * np.sqrt(y_pred_var)
            metrics[f'ci_width_{conf_level}'] = np.mean(ci_width)
            
            # Calculate empirical coverage
            lower_bound = y_pred_mean - z_score * np.sqrt(y_pred_var)
            upper_bound = y_pred_mean + z_score * np.sqrt(y_pred_var)
            
            coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
            metrics[f'coverage_{conf_level}'] = coverage
        
        # Calibration of uncertainty
        metrics.update(self._calculate_uncertainty_calibration(y_true, y_pred_mean, y_pred_var))
        
        return metrics
    
    def _calculate_uncertainty_calibration(
        self, 
        y_true: np.ndarray, 
        y_pred_mean: np.ndarray, 
        y_pred_var: np.ndarray
    ) -> Dict[str, float]:
        """Calculate uncertainty calibration metrics.
        
        Args:
            y_true: True labels.
            y_pred_mean: Mean predictions.
            y_pred_var: Prediction variances.
            
        Returns:
            Dict[str, float]: Uncertainty calibration metrics.
        """
        # Calculate prediction errors
        errors = np.abs(y_true - y_pred_mean)
        
        # Calculate uncertainty-accuracy correlation
        uncertainty = np.sqrt(y_pred_var)
        correlation = np.corrcoef(errors, uncertainty)[0, 1]
        
        # Calculate reliability diagram
        n_bins = 10
        uncertainty_bins = np.linspace(0, np.max(uncertainty), n_bins + 1)
        
        calibration_metrics = {}
        
        for i in range(n_bins):
            bin_mask = (uncertainty >= uncertainty_bins[i]) & (uncertainty < uncertainty_bins[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_errors = errors[bin_mask]
                bin_uncertainty = uncertainty[bin_mask]
                
                # Calculate calibration metrics for this bin
                calibration_metrics[f'bin_{i}_mean_error'] = np.mean(bin_errors)
                calibration_metrics[f'bin_{i}_mean_uncertainty'] = np.mean(bin_uncertainty)
        
        calibration_metrics['uncertainty_accuracy_correlation'] = correlation
        
        return calibration_metrics
    
    def plot_uncertainty_analysis(
        self, 
        y_true: np.ndarray, 
        y_pred_mean: np.ndarray, 
        y_pred_var: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot uncertainty analysis.
        
        Args:
            y_true: True labels.
            y_pred_mean: Mean predictions.
            y_pred_var: Prediction variances.
            save_path: Optional path to save the plot.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prediction vs uncertainty
        axes[0, 0].scatter(y_pred_mean, np.sqrt(y_pred_var), alpha=0.6)
        axes[0, 0].set_xlabel('Predicted Probability')
        axes[0, 0].set_ylabel('Uncertainty (Std)')
        axes[0, 0].set_title('Prediction vs Uncertainty')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error vs uncertainty
        errors = np.abs(y_true - y_pred_mean)
        uncertainty = np.sqrt(y_pred_var)
        axes[0, 1].scatter(uncertainty, errors, alpha=0.6)
        axes[0, 1].set_xlabel('Uncertainty (Std)')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Error vs Uncertainty')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Uncertainty distribution
        axes[1, 0].hist(uncertainty, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Uncertainty (Std)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Uncertainty Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calibration plot
        n_bins = 10
        uncertainty_bins = np.linspace(0, np.max(uncertainty), n_bins + 1)
        bin_centers = []
        bin_errors = []
        
        for i in range(n_bins):
            bin_mask = (uncertainty >= uncertainty_bins[i]) & (uncertainty < uncertainty_bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_centers.append(np.mean(uncertainty[bin_mask]))
                bin_errors.append(np.mean(errors[bin_mask]))
        
        axes[1, 1].plot(bin_centers, bin_errors, 'o-', label='Empirical Error')
        axes[1, 1].plot([0, np.max(uncertainty)], [0, np.max(uncertainty)], 'k--', label='Perfect Calibration')
        axes[1, 1].set_xlabel('Predicted Uncertainty')
        axes[1, 1].set_ylabel('Empirical Error')
        axes[1, 1].set_title('Uncertainty Calibration')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class SafetyChecker:
    """Safety checks for adherence prediction models."""
    
    def __init__(self):
        """Initialize safety checker."""
        self.safety_metrics = {}
    
    def check_prediction_safety(
        self, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray,
        thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Check prediction safety.
        
        Args:
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            thresholds: Safety thresholds.
            
        Returns:
            Dict[str, Any]: Safety check results.
        """
        if thresholds is None:
            thresholds = {
                'min_confidence': 0.7,
                'max_uncertainty': 0.3,
                'min_samples_per_class': 10
            }
        
        safety_results = {}
        
        # Confidence check
        if y_proba.ndim > 1:
            confidence = np.max(y_proba, axis=1)
        else:
            confidence = np.maximum(y_proba, 1 - y_proba)
        
        low_confidence_mask = confidence < thresholds['min_confidence']
        safety_results['low_confidence_count'] = np.sum(low_confidence_mask)
        safety_results['low_confidence_rate'] = np.mean(low_confidence_mask)
        
        # Uncertainty check
        uncertainty = 1 - confidence
        high_uncertainty_mask = uncertainty > thresholds['max_uncertainty']
        safety_results['high_uncertainty_count'] = np.sum(high_uncertainty_mask)
        safety_results['high_uncertainty_rate'] = np.mean(high_uncertainty_mask)
        
        # Class balance check
        unique_classes, counts = np.unique(y_pred, return_counts=True)
        min_class_count = np.min(counts)
        safety_results['min_class_count'] = min_class_count
        safety_results['balanced_classes'] = min_class_count >= thresholds['min_samples_per_class']
        
        # Overall safety score
        safety_score = 1.0
        safety_score -= safety_results['low_confidence_rate'] * 0.5
        safety_score -= safety_results['high_uncertainty_rate'] * 0.3
        safety_score -= (1 - safety_results['balanced_classes']) * 0.2
        
        safety_results['safety_score'] = max(0, safety_score)
        
        self.safety_metrics = safety_results
        return safety_results
    
    def generate_safety_report(self) -> str:
        """Generate safety report.
        
        Returns:
            str: Safety report.
        """
        if not self.safety_metrics:
            return "No safety metrics available. Run check_prediction_safety first."
        
        report = "=== SAFETY REPORT ===\n"
        report += f"Safety Score: {self.safety_metrics['safety_score']:.3f}\n"
        report += f"Low Confidence Rate: {self.safety_metrics['low_confidence_rate']:.3f}\n"
        report += f"High Uncertainty Rate: {self.safety_metrics['high_uncertainty_rate']:.3f}\n"
        report += f"Balanced Classes: {self.safety_metrics['balanced_classes']}\n"
        
        if self.safety_metrics['safety_score'] < 0.7:
            report += "\n⚠️  WARNING: Low safety score detected!\n"
            report += "Consider:\n"
            report += "- Increasing training data\n"
            report += "- Improving model calibration\n"
            report += "- Adding uncertainty quantification\n"
        
        return report


def create_explainability_report(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create comprehensive explainability report.
    
    Args:
        model: Trained model.
        X: Features.
        y_true: True labels.
        feature_names: Feature names.
        save_path: Optional path to save the report.
        
    Returns:
        Dict[str, Any]: Explainability report.
    """
    report = {}
    
    # SHAP explanations
    if SHAP_AVAILABLE:
        try:
            shap_explainer = SHAPExplainer(model, feature_names)
            shap_explainer.fit_explainer(X[:100], X[:50])  # Use subset for efficiency
            
            # Explain dataset
            dataset_explanation = shap_explainer.explain_dataset(X[:50])
            report['shap_dataset'] = dataset_explanation
            
            # Explain single instance
            instance_explanation = shap_explainer.explain_instance(X[0])
            report['shap_instance'] = instance_explanation
            
            # Create plots
            if save_path:
                shap_explainer.plot_feature_importance(dataset_explanation, save_path=f"{save_path}_shap_importance.png")
                shap_explainer.plot_waterfall(instance_explanation, save_path=f"{save_path}_shap_waterfall.png")
            
        except Exception as e:
            report['shap_error'] = str(e)
    
    # Safety checks
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    safety_checker = SafetyChecker()
    safety_results = safety_checker.check_prediction_safety(y_pred, y_proba)
    report['safety'] = safety_results
    report['safety_report'] = safety_checker.generate_safety_report()
    
    return report
