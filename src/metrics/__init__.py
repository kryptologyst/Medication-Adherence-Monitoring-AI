"""
Comprehensive evaluation metrics for medication adherence monitoring.

This module provides clinical metrics, calibration analysis, fairness evaluation,
and uncertainty quantification for adherence prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings


class ClinicalMetrics:
    """Clinical evaluation metrics for adherence prediction."""
    
    def __init__(self):
        """Initialize clinical metrics calculator."""
        self.metrics = {}
    
    def calculate_all_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate all clinical metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Clinical metrics
        metrics.update(self._calculate_clinical_metrics(y_true, y_pred))
        
        # Probability-based metrics
        if y_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_proba))
        
        # Confusion matrix metrics
        metrics.update(self._calculate_confusion_metrics(y_true, y_pred))
        
        self.metrics = metrics
        return metrics
    
    def _calculate_clinical_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clinical-specific metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Dict[str, float]: Clinical metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Sensitivity (Recall) - True Positive Rate
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity - True Negative Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Positive Predictive Value (PPV) - Precision
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Negative Predictive Value (NPV)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Likelihood Ratios
        lr_positive = sensitivity / (1 - specificity) if specificity < 1 else np.inf
        lr_negative = (1 - sensitivity) / specificity if specificity > 0 else np.inf
        
        # Diagnostic Odds Ratio
        dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.inf
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'lr_positive': lr_positive,
            'lr_negative': lr_negative,
            'diagnostic_odds_ratio': dor
        }
    
    def _calculate_probability_metrics(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate probability-based metrics.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Dict[str, float]: Probability metrics.
        """
        # Ensure probabilities are for positive class
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        # AUC-ROC
        auroc = roc_auc_score(y_true, y_proba)
        
        # AUC-PR (Average Precision)
        auprc = average_precision_score(y_true, y_proba)
        
        # Brier Score
        brier_score = np.mean((y_proba - y_true) ** 2)
        
        # Log Loss
        epsilon = 1e-15
        y_proba = np.clip(y_proba, epsilon, 1 - epsilon)
        log_loss = -np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))
        
        return {
            'auroc': auroc,
            'auprc': auprc,
            'brier_score': brier_score,
            'log_loss': log_loss
        }
    
    def _calculate_confusion_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate confusion matrix derived metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Dict[str, float]: Confusion matrix metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Prevalence
        prevalence = (tp + fn) / (tp + tn + fp + fn)
        
        # Balanced Accuracy
        balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
        
        # Matthews Correlation Coefficient
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        return {
            'prevalence': prevalence,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc
        }


class CalibrationAnalyzer:
    """Analyze model calibration for adherence prediction."""
    
    def __init__(self, n_bins: int = 10):
        """Initialize calibration analyzer.
        
        Args:
            n_bins: Number of bins for calibration analysis.
        """
        self.n_bins = n_bins
        self.calibration_data = {}
    
    def analyze_calibration(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        method: str = 'isotonic'
    ) -> Dict[str, Any]:
        """Analyze model calibration.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            method: Calibration method ('isotonic' or 'sigmoid').
            
        Returns:
            Dict[str, Any]: Calibration analysis results.
        """
        # Ensure probabilities are for positive class
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=self.n_bins
        )
        
        # Calculate Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, y_proba)
        
        # Calculate Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y_true, y_proba)
        
        # Calculate reliability diagram data
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece_per_bin = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece_per_bin.append(np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin)
            else:
                ece_per_bin.append(0)
        
        self.calibration_data = {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'ece': ece,
            'mce': mce,
            'ece_per_bin': ece_per_bin,
            'bin_boundaries': bin_boundaries
        }
        
        return self.calibration_data
    
    def _calculate_ece(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Expected Calibration Error.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            float: ECE value.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Maximum Calibration Error.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            float: MCE value.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def plot_calibration_curve(self, save_path: Optional[str] = None) -> None:
        """Plot calibration curve.
        
        Args:
            save_path: Optional path to save the plot.
        """
        if not self.calibration_data:
            raise ValueError("Must run analyze_calibration first")
        
        plt.figure(figsize=(10, 8))
        
        # Calibration curve
        plt.subplot(2, 2, 1)
        plt.plot(
            self.calibration_data['mean_predicted_value'],
            self.calibration_data['fraction_of_positives'],
            'o-', label=f'Model (ECE={self.calibration_data["ece"]:.3f})'
        )
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Reliability diagram
        plt.subplot(2, 2, 2)
        bin_centers = (self.calibration_data['bin_boundaries'][:-1] + 
                      self.calibration_data['bin_boundaries'][1:]) / 2
        plt.bar(bin_centers, self.calibration_data['ece_per_bin'], 
                width=0.8/self.n_bins, alpha=0.7)
        plt.xlabel('Predicted Probability')
        plt.ylabel('ECE per Bin')
        plt.title('Reliability Diagram')
        plt.grid(True, alpha=0.3)
        
        # Histogram of predicted probabilities
        plt.subplot(2, 2, 3)
        plt.hist(self.calibration_data['mean_predicted_value'], 
                bins=self.n_bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Probabilities')
        plt.grid(True, alpha=0.3)
        
        # ECE per bin
        plt.subplot(2, 2, 4)
        plt.bar(range(self.n_bins), self.calibration_data['ece_per_bin'], 
                alpha=0.7, edgecolor='black')
        plt.xlabel('Bin')
        plt.ylabel('ECE')
        plt.title('ECE per Bin')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class FairnessAnalyzer:
    """Analyze fairness across demographic groups."""
    
    def __init__(self):
        """Initialize fairness analyzer."""
        self.fairness_metrics = {}
    
    def analyze_fairness(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray,
        groups: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze fairness across demographic groups.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            groups: Dictionary mapping group names to group assignments.
            
        Returns:
            Dict[str, Dict[str, float]]: Fairness metrics by group.
        """
        fairness_results = {}
        
        for group_name, group_assignments in groups.items():
            group_metrics = {}
            unique_groups = np.unique(group_assignments)
            
            for group_value in unique_groups:
                mask = group_assignments == group_value
                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]
                group_y_proba = y_proba[mask]
                
                if len(group_y_true) > 0:
                    # Calculate metrics for this group
                    clinical_metrics = ClinicalMetrics()
                    group_metrics[f'{group_name}_{group_value}'] = clinical_metrics.calculate_all_metrics(
                        group_y_true, group_y_pred, group_y_proba
                    )
            
            fairness_results[group_name] = group_metrics
        
        # Calculate fairness gaps
        fairness_results = self._calculate_fairness_gaps(fairness_results)
        
        self.fairness_metrics = fairness_results
        return fairness_results
    
    def _calculate_fairness_gaps(
        self, 
        fairness_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate fairness gaps between groups.
        
        Args:
            fairness_results: Fairness results by group.
            
        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: Results with fairness gaps.
        """
        for group_name, group_metrics in fairness_results.items():
            if len(group_metrics) < 2:
                continue
            
            # Calculate gaps for each metric
            metric_names = list(list(group_metrics.values())[0].keys())
            
            for metric_name in metric_names:
                values = [metrics[metric_name] for metrics in group_metrics.values()]
                if len(values) > 1:
                    gap = max(values) - min(values)
                    fairness_results[group_name][f'{metric_name}_gap'] = gap
        
        return fairness_results
    
    def plot_fairness_analysis(self, save_path: Optional[str] = None) -> None:
        """Plot fairness analysis results.
        
        Args:
            save_path: Optional path to save the plot.
        """
        if not self.fairness_metrics:
            raise ValueError("Must run analyze_fairness first")
        
        n_groups = len(self.fairness_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        for group_name, group_metrics in self.fairness_metrics.items():
            if plot_idx >= 4:
                break
                
            ax = axes[plot_idx]
            
            # Extract metrics for plotting
            metrics_to_plot = ['sensitivity', 'specificity', 'ppv', 'npv']
            group_names = list(group_metrics.keys())
            
            x = np.arange(len(metrics_to_plot))
            width = 0.35
            
            for i, group_key in enumerate(group_names):
                if not group_key.endswith('_gap'):
                    values = [group_metrics[group_key].get(metric, 0) for metric in metrics_to_plot]
                    ax.bar(x + i * width, values, width, label=group_key, alpha=0.7)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Value')
            ax.set_title(f'Fairness Analysis: {group_name}')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(metrics_to_plot)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class UncertaintyQuantifier:
    """Quantify model uncertainty for adherence prediction."""
    
    def __init__(self):
        """Initialize uncertainty quantifier."""
        self.uncertainty_metrics = {}
    
    def calculate_uncertainty_metrics(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        method: str = 'entropy'
    ) -> Dict[str, float]:
        """Calculate uncertainty metrics.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            method: Uncertainty method ('entropy', 'variance', 'confidence').
            
        Returns:
            Dict[str, float]: Uncertainty metrics.
        """
        # Ensure probabilities are for positive class
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        metrics = {}
        
        if method == 'entropy':
            # Entropy-based uncertainty
            epsilon = 1e-15
            y_proba_clipped = np.clip(y_proba, epsilon, 1 - epsilon)
            entropy = -np.sum(y_proba_clipped * np.log(y_proba_clipped) + 
                            (1 - y_proba_clipped) * np.log(1 - y_proba_clipped), axis=0)
            metrics['entropy'] = entropy
        
        elif method == 'variance':
            # Variance-based uncertainty (for ensemble predictions)
            variance = np.var(y_proba)
            metrics['variance'] = variance
        
        elif method == 'confidence':
            # Confidence-based uncertainty
            confidence = np.maximum(y_proba, 1 - y_proba)
            metrics['confidence'] = np.mean(confidence)
            metrics['uncertainty'] = 1 - np.mean(confidence)
        
        # Calibration of uncertainty
        metrics.update(self._calculate_uncertainty_calibration(y_true, y_proba))
        
        self.uncertainty_metrics = metrics
        return metrics
    
    def _calculate_uncertainty_calibration(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate uncertainty calibration metrics.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Dict[str, float]: Uncertainty calibration metrics.
        """
        # Confidence intervals
        confidence_levels = [0.5, 0.8, 0.9, 0.95]
        calibration_metrics = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_bound = alpha / 2
            upper_bound = 1 - alpha / 2
            
            # Calculate empirical coverage
            coverage = np.mean((y_proba >= lower_bound) & (y_proba <= upper_bound))
            calibration_metrics[f'coverage_{conf_level}'] = coverage
        
        return calibration_metrics


def create_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    groups: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create comprehensive evaluation report.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        groups: Optional demographic groups for fairness analysis.
        save_path: Optional path to save the report.
        
    Returns:
        Dict[str, Any]: Comprehensive evaluation report.
    """
    report = {}
    
    # Clinical metrics
    clinical_metrics = ClinicalMetrics()
    report['clinical_metrics'] = clinical_metrics.calculate_all_metrics(y_true, y_pred, y_proba)
    
    # Calibration analysis
    calibration_analyzer = CalibrationAnalyzer()
    report['calibration'] = calibration_analyzer.analyze_calibration(y_true, y_proba)
    
    # Uncertainty quantification
    uncertainty_quantifier = UncertaintyQuantifier()
    report['uncertainty'] = uncertainty_quantifier.calculate_uncertainty_metrics(y_true, y_proba)
    
    # Fairness analysis
    if groups is not None:
        fairness_analyzer = FairnessAnalyzer()
        report['fairness'] = fairness_analyzer.analyze_fairness(y_true, y_pred, y_proba, groups)
    
    # Create visualizations
    if save_path:
        calibration_analyzer.plot_calibration_curve(f"{save_path}_calibration.png")
        
        if groups is not None:
            fairness_analyzer.plot_fairness_analysis(f"{save_path}_fairness.png")
    
    return report
