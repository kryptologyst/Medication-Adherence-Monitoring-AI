#!/usr/bin/env python3
"""
Evaluation script for medication adherence monitoring models.

This script evaluates trained models and generates comprehensive reports
including clinical metrics, calibration analysis, and fairness evaluation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import set_seed, get_device, setup_logging, load_config
from data import AdherenceDataGenerator, AdherenceDataProcessor
from models import create_model, create_deep_model
from metrics import ClinicalMetrics, CalibrationAnalyzer, FairnessAnalyzer, create_evaluation_report
from explainability import SHAPExplainer, UncertaintyQuantifier, SafetyChecker, create_explainability_report


class ModelEvaluator:
    """Evaluator for medication adherence models."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """Initialize evaluator.
        
        Args:
            config: Configuration dictionary.
            device: Device for evaluation.
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger('adherence_monitoring')
        
        # Initialize components
        self.data_generator = AdherenceDataGenerator(seed=config['data']['synthetic']['seed'])
        self.data_processor = AdherenceDataProcessor()
        
        # Initialize metrics
        self.clinical_metrics = ClinicalMetrics()
        self.calibration_analyzer = CalibrationAnalyzer()
        self.fairness_analyzer = FairnessAnalyzer()
        self.uncertainty_quantifier = UncertaintyQuantifier(None, device)
        self.safety_checker = SafetyChecker()
    
    def load_models(self, model_path: str) -> Dict[str, Any]:
        """Load trained models.
        
        Args:
            model_path: Path to model directory.
            
        Returns:
            Dict[str, Any]: Loaded models.
        """
        models = {}
        
        # Load feature names
        feature_names = joblib.load(f'{model_path}/feature_names.pkl')
        
        # Load traditional models
        traditional_models = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in traditional_models:
            model_file = f'{model_path}/{model_name}.pkl'
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                models[model_name] = model
                self.logger.info(f"Loaded {model_name} model")
        
        # Load deep models
        deep_models = ['tabnet', 'ft_transformer']
        
        for model_name in deep_models:
            model_file = f'{model_path}/{model_name}.pth'
            if os.path.exists(model_file):
                # Create model architecture
                model = create_deep_model(model_name, input_dim=len(feature_names), output_dim=2)
                model.load_state_dict(torch.load(model_file, map_location=self.device))
                model.eval()
                models[model_name] = model
                self.logger.info(f"Loaded {model_name} model")
        
        # Load ensemble if exists
        ensemble_file = f'{model_path}/ensemble.pkl'
        if os.path.exists(ensemble_file):
            models['ensemble'] = joblib.load(ensemble_file)
            self.logger.info("Loaded ensemble model")
        
        return models, feature_names
    
    def generate_test_data(self) -> tuple:
        """Generate test data for evaluation.
        
        Returns:
            tuple: Test data and groups.
        """
        self.logger.info("Generating test data...")
        
        # Generate fresh data for testing
        df = self.data_generator.generate_patient_data(
            n_patients=500,  # Smaller test set
            n_days_per_patient=self.config['data']['synthetic']['n_days_per_patient']
        )
        
        # Add temporal features
        df = self.data_generator.add_temporal_features(df)
        
        # Create demographic groups
        groups = {
            'age_group': pd.cut(df['age'], bins=[0, 50, 65, 80, 100], 
                              labels=['young', 'middle', 'senior', 'elderly']).astype(str),
            'gender': df['gender'].values,
            'insurance_type': df['insurance_type'].values
        }
        
        return df, groups
    
    def prepare_test_data(self, df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        """Prepare test data.
        
        Args:
            df: Test dataframe.
            feature_names: Feature names.
            
        Returns:
            np.ndarray: Prepared test features.
        """
        # Get feature columns
        static_features = self.config['data']['features']['static']
        temporal_features = self.config['data']['features']['temporal']
        
        # Add temporal features
        temporal_features.extend([
            'adherence_rate_3d', 'adherence_rate_7d', 'adherence_rate_14d',
            'on_time_rate_3d', 'on_time_rate_7d', 'on_time_rate_14d',
            'adherence_trend', 'day_of_week', 'is_weekend'
        ])
        
        # Prepare features
        X, y = self.data_processor.prepare_features(df, static_features, temporal_features)
        
        # Scale features (using the same scaler as training)
        X_scaled = self.data_processor.scale_features(X, X)[0]
        
        return X_scaled, y
    
    def evaluate_model_comprehensive(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        groups: Dict[str, np.ndarray],
        feature_names: List[str],
        model_name: str
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test targets.
            groups: Demographic groups.
            feature_names: Feature names.
            model_name: Name of the model.
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results.
        """
        self.logger.info(f"Comprehensive evaluation of {model_name}...")
        
        # Make predictions
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                outputs = model(X_test_tensor)
                y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Clinical evaluation
        clinical_results = self.clinical_metrics.calculate_all_metrics(y_test, y_pred, y_pred_proba)
        
        # Calibration analysis
        calibration_results = self.calibration_analyzer.analyze_calibration(y_test, y_pred_proba)
        
        # Fairness analysis
        fairness_results = self.fairness_analyzer.analyze_fairness(y_test, y_pred, y_pred_proba, groups)
        
        # Uncertainty quantification
        uncertainty_results = self.uncertainty_quantifier.calculate_uncertainty_metrics(
            y_test, y_pred_proba
        )
        
        # Safety checks
        safety_results = self.safety_checker.check_prediction_safety(y_pred, y_pred_proba)
        
        # SHAP explanations
        explainability_results = {}
        try:
            explainability_results = create_explainability_report(
                model, X_test[:100], y_test[:100], feature_names  # Use subset for efficiency
            )
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed: {e}")
            explainability_results = {'error': str(e)}
        
        # Combine all results
        comprehensive_results = {
            'model_name': model_name,
            'clinical_metrics': clinical_results,
            'calibration': calibration_results,
            'fairness': fairness_results,
            'uncertainty': uncertainty_results,
            'safety': safety_results,
            'explainability': explainability_results
        }
        
        return comprehensive_results
    
    def create_evaluation_plots(
        self, 
        results: Dict[str, Any], 
        save_dir: str = 'assets'
    ) -> None:
        """Create evaluation plots.
        
        Args:
            results: Evaluation results.
            save_dir: Directory to save plots.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Model comparison plot
        self._plot_model_comparison(results, save_dir)
        
        # Calibration plots
        self._plot_calibration_curves(results, save_dir)
        
        # Fairness plots
        self._plot_fairness_analysis(results, save_dir)
        
        # Feature importance plots
        self._plot_feature_importance(results, save_dir)
    
    def _plot_model_comparison(self, results: Dict[str, Any], save_dir: str) -> None:
        """Plot model comparison.
        
        Args:
            results: Evaluation results.
            save_dir: Directory to save plots.
        """
        models = list(results.keys())
        metrics = ['auroc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'npv']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model]['clinical_metrics'][metric] for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_calibration_curves(self, results: Dict[str, Any], save_dir: str) -> None:
        """Plot calibration curves.
        
        Args:
            results: Evaluation results.
            save_dir: Directory to save plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, (model_name, result) in enumerate(results.items()):
            if i >= 4:
                break
                
            ax = axes[i//2, i%2]
            
            calibration_data = result['calibration']
            
            # Calibration curve
            ax.plot(
                calibration_data['mean_predicted_value'],
                calibration_data['fraction_of_positives'],
                'o-', label=f'{model_name} (ECE={calibration_data["ece"]:.3f})'
            )
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Curve: {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_fairness_analysis(self, results: Dict[str, Any], save_dir: str) -> None:
        """Plot fairness analysis.
        
        Args:
            results: Evaluation results.
            save_dir: Directory to save plots.
        """
        # Get first model's fairness results
        first_model = list(results.keys())[0]
        fairness_data = results[first_model]['fairness']
        
        if not fairness_data:
            return
        
        fig, axes = plt.subplots(1, len(fairness_data), figsize=(5*len(fairness_data), 6))
        if len(fairness_data) == 1:
            axes = [axes]
        
        for i, (group_name, group_metrics) in enumerate(fairness_data.items()):
            ax = axes[i]
            
            # Extract metrics for plotting
            metrics_to_plot = ['sensitivity', 'specificity', 'ppv', 'npv']
            group_names = list(group_metrics.keys())
            
            x = np.arange(len(metrics_to_plot))
            width = 0.35
            
            for j, group_key in enumerate(group_names):
                if not group_key.endswith('_gap'):
                    values = [group_metrics[group_key].get(metric, 0) for metric in metrics_to_plot]
                    ax.bar(x + j * width, values, width, label=group_key, alpha=0.7)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Value')
            ax.set_title(f'Fairness Analysis: {group_name}')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(metrics_to_plot)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/fairness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self, results: Dict[str, Any], save_dir: str) -> None:
        """Plot feature importance.
        
        Args:
            results: Evaluation results.
            save_dir: Directory to save plots.
        """
        # Get first model with feature importance
        for model_name, result in results.items():
            if 'explainability' in result and 'shap_dataset' in result['explainability']:
                shap_data = result['explainability']['shap_dataset']
                
                plt.figure(figsize=(12, 8))
                
                # Plot mean importance
                importance = shap_data['mean_importance']
                feature_names = shap_data['feature_names']
                
                # Sort by importance
                sorted_indices = np.argsort(importance)[::-1][:15]  # Top 15
                sorted_importance = importance[sorted_indices]
                sorted_features = [feature_names[i] for i in sorted_indices]
                
                sns.barplot(x=sorted_importance, y=sorted_features)
                plt.title(f'Feature Importance: {model_name}')
                plt.xlabel('Mean |SHAP Value|')
                plt.tight_layout()
                plt.savefig(f'{save_dir}/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.show()
                break
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results.
            
        Returns:
            str: Evaluation report.
        """
        report = []
        report.append("=== MEDICATION ADHERENCE MONITORING - EVALUATION REPORT ===\n")
        
        # Executive summary
        report.append("EXECUTIVE SUMMARY")
        report.append("=" * 50)
        
        best_model = max(results.keys(), key=lambda x: results[x]['clinical_metrics']['auroc'])
        best_auroc = results[best_model]['clinical_metrics']['auroc']
        
        report.append(f"Best performing model: {best_model}")
        report.append(f"Best AUROC: {best_auroc:.3f}")
        report.append("")
        
        # Model performance comparison
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("=" * 50)
        
        for model_name, result in results.items():
            metrics = result['clinical_metrics']
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  AUROC: {metrics['auroc']:.3f}")
            report.append(f"  AUPRC: {metrics['auprc']:.3f}")
            report.append(f"  Sensitivity: {metrics['sensitivity']:.3f}")
            report.append(f"  Specificity: {metrics['specificity']:.3f}")
            report.append(f"  PPV: {metrics['ppv']:.3f}")
            report.append(f"  NPV: {metrics['npv']:.3f}")
            report.append(f"  ECE: {result['calibration']['ece']:.3f}")
        
        # Safety analysis
        report.append("\nSAFETY ANALYSIS")
        report.append("=" * 50)
        
        for model_name, result in results.items():
            safety = result['safety']
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  Safety Score: {safety['safety_score']:.3f}")
            report.append(f"  Low Confidence Rate: {safety['low_confidence_rate']:.3f}")
            report.append(f"  High Uncertainty Rate: {safety['high_uncertainty_rate']:.3f}")
        
        # Fairness analysis
        report.append("\nFAIRNESS ANALYSIS")
        report.append("=" * 50)
        
        for model_name, result in results.items():
            fairness = result['fairness']
            report.append(f"\n{model_name.upper()}:")
            
            for group_name, group_metrics in fairness.items():
                report.append(f"  {group_name}:")
                for metric_name, metric_value in group_metrics.items():
                    if metric_name.endswith('_gap'):
                        report.append(f"    {metric_name}: {metric_value:.3f}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS")
        report.append("=" * 50)
        
        if best_auroc < 0.8:
            report.append("- Model performance could be improved with more training data")
        
        if any(results[model]['calibration']['ece'] > 0.1 for model in results):
            report.append("- Consider calibration techniques to improve probability estimates")
        
        if any(results[model]['safety']['safety_score'] < 0.7 for model in results):
            report.append("- Implement additional safety measures for low-confidence predictions")
        
        report.append("\n⚠️  IMPORTANT DISCLAIMER:")
        report.append("This is a research demonstration with synthetic data.")
        report.append("Not intended for clinical use or medical decision making.")
        report.append("Always consult healthcare professionals for medical decisions.")
        
        return "\n".join(report)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate medication adherence models')
    parser.add_argument('--model_path', type=str, default='models',
                       help='Path to trained models')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='assets',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_logging(
        log_level=config.get('logging', {}).get('level', 'INFO'),
        log_dir=config.get('logging', {}).get('log_dir', 'logs')
    )
    
    # Set seed
    set_seed(config['data']['synthetic']['seed'])
    
    # Get device
    device = get_device(args.device if args.device != 'auto' else None)
    
    # Create evaluator
    evaluator = ModelEvaluator(config, device)
    
    # Load models
    models, feature_names = evaluator.load_models(args.model_path)
    
    if not models:
        logger.error("No models found to evaluate")
        return
    
    logger.info(f"Loaded {len(models)} models for evaluation")
    
    # Generate test data
    df, groups = evaluator.generate_test_data()
    
    # Prepare test data
    X_test, y_test = evaluator.prepare_test_data(df, feature_names)
    
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Evaluate all models
    evaluation_results = {}
    
    for model_name, model in models.items():
        try:
            results = evaluator.evaluate_model_comprehensive(
                model, X_test, y_test, groups, feature_names, model_name
            )
            evaluation_results[model_name] = results
            
            logger.info(f"Completed evaluation of {model_name}")
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
    
    # Create plots
    evaluator.create_evaluation_plots(evaluation_results, args.output_dir)
    
    # Generate report
    report = evaluator.generate_evaluation_report(evaluation_results)
    
    # Save report
    with open(f'{args.output_dir}/evaluation_report.txt', 'w') as f:
        f.write(report)
    
    # Print report
    print(report)
    
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
