#!/usr/bin/env python3
"""
Training script for medication adherence monitoring models.

This script trains various models on synthetic adherence data and saves
the best performing model for evaluation and deployment.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
from tqdm import tqdm
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import set_seed, get_device, setup_logging, load_config, PrivacyProtector
from data import AdherenceDataGenerator, AdherenceDataProcessor
from models import (
    LogisticRegressionModel, RandomForestModel, XGBoostModel, 
    TabNetModel, FTTransformerModel, EnsembleModel, create_model, create_deep_model
)
from metrics import ClinicalMetrics, CalibrationAnalyzer, FairnessAnalyzer
from explainability import SHAPExplainer, UncertaintyQuantifier, SafetyChecker


class ModelTrainer:
    """Trainer for medication adherence models."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary.
            device: Device for training.
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger('adherence_monitoring')
        self.privacy_protector = PrivacyProtector(
            deid=config.get('privacy', {}).get('deid', True)
        )
        
        # Initialize data components
        self.data_generator = AdherenceDataGenerator(seed=config['data']['synthetic']['seed'])
        self.data_processor = AdherenceDataProcessor()
        
        # Initialize metrics
        self.clinical_metrics = ClinicalMetrics()
        self.calibration_analyzer = CalibrationAnalyzer()
        self.fairness_analyzer = FairnessAnalyzer()
        
        # Training history
        self.training_history = {}
        self.best_models = {}
    
    def generate_data(self) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Generate synthetic adherence data.
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, np.ndarray]]: Data and groups.
        """
        self.logger.info("Generating synthetic adherence data...")
        
        # Generate data
        df = self.data_generator.generate_patient_data(
            n_patients=self.config['data']['synthetic']['n_patients'],
            n_days_per_patient=self.config['data']['synthetic']['n_days_per_patient']
        )
        
        # Add temporal features
        df = self.data_generator.add_temporal_features(df)
        
        # Create demographic groups for fairness analysis
        groups = {
            'age_group': pd.cut(df['age'], bins=[0, 50, 65, 80, 100], labels=['young', 'middle', 'senior', 'elderly']).astype(str),
            'gender': df['gender'].values,
            'insurance_type': df['insurance_type'].values
        }
        
        self.logger.info(f"Generated data shape: {df.shape}")
        self.logger.info(f"Adherence risk rate: {df['adherence_risk'].mean():.3f}")
        
        return df, groups
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training.
        
        Args:
            df: Input dataframe.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, targets, feature names.
        """
        self.logger.info("Preparing data for training...")
        
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
        
        # Handle class imbalance
        if self.config['training'].get('handle_imbalance', True):
            X, y = self.data_processor.handle_class_imbalance(X, y, method='smote')
            self.logger.info(f"After resampling - X shape: {X.shape}, y shape: {y.shape}")
        
        # Scale features
        X_scaled = self.data_processor.scale_features(X, X)[0]  # Use same data for scaling
        
        feature_names = static_features + temporal_features
        
        self.logger.info(f"Final feature shape: {X_scaled.shape}")
        self.logger.info(f"Feature names: {feature_names}")
        
        return X_scaled, y, feature_names
    
    def train_traditional_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train traditional ML model.
        
        Args:
            model_name: Name of the model.
            X_train: Training features.
            y_train: Training targets.
            
        Returns:
            Any: Trained model.
        """
        self.logger.info(f"Training {model_name} model...")
        
        # Create model
        model = create_model(model_name)
        
        # Train model
        model.fit(X_train, y_train)
        
        self.logger.info(f"{model_name} training completed")
        return model
    
    def train_deep_model(
        self, 
        model_name: str, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> nn.Module:
        """Train deep learning model.
        
        Args:
            model_name: Name of the model.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            
        Returns:
            nn.Module: Trained model.
        """
        self.logger.info(f"Training {model_name} deep model...")
        
        # Create model
        model = create_deep_model(
            model_name, 
            input_dim=X_train.shape[1], 
            output_dim=2
        ).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=False
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['training']['learning_rate']
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}: "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'models/best_{model_name}.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(f'models/best_{model_name}.pth'))
        
        self.logger.info(f"{model_name} deep model training completed")
        return model
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        groups: Dict[str, np.ndarray],
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test targets.
            groups: Demographic groups.
            model_name: Name of the model.
            
        Returns:
            Dict[str, Any]: Evaluation results.
        """
        self.logger.info(f"Evaluating {model_name} model...")
        
        # Make predictions
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                outputs = model(X_test_tensor)
                y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate clinical metrics
        clinical_results = self.clinical_metrics.calculate_all_metrics(y_test, y_pred, y_pred_proba)
        
        # Calibration analysis
        calibration_results = self.calibration_analyzer.analyze_calibration(y_test, y_pred_proba)
        
        # Fairness analysis
        fairness_results = self.fairness_analyzer.analyze_fairness(y_test, y_pred, y_pred_proba, groups)
        
        # Combine results
        results = {
            'model_name': model_name,
            'clinical_metrics': clinical_results,
            'calibration': calibration_results,
            'fairness': fairness_results
        }
        
        self.logger.info(f"{model_name} evaluation completed")
        self.logger.info(f"AUROC: {clinical_results['auroc']:.3f}")
        self.logger.info(f"AUPRC: {clinical_results['auprc']:.3f}")
        
        return results
    
    def train_ensemble(self, models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray) -> EnsembleModel:
        """Train ensemble model.
        
        Args:
            models: Dictionary of trained models.
            X_train: Training features.
            y_train: Training targets.
            
        Returns:
            EnsembleModel: Trained ensemble.
        """
        self.logger.info("Training ensemble model...")
        
        # Create ensemble
        ensemble_models = list(models.values())
        ensemble_weights = self.config['model']['ensemble']['weights']
        
        ensemble = EnsembleModel(ensemble_models, ensemble_weights)
        ensemble.fit(X_train, y_train)
        
        self.logger.info("Ensemble training completed")
        return ensemble
    
    def train_all_models(self, df: pd.DataFrame, groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train all models.
        
        Args:
            df: Training data.
            groups: Demographic groups.
            
        Returns:
            Dict[str, Any]: Training results.
        """
        self.logger.info("Starting model training...")
        
        # Prepare data
        X, y, feature_names = self.prepare_data(df)
        
        # Patient-level split
        train_df, val_df, test_df = self.data_processor.patient_level_split(
            df, 
            test_size=self.config['training']['test_split'],
            val_size=self.config['training']['validation_split']
        )
        
        # Prepare splits
        X_train, y_train, _ = self.prepare_data(train_df)
        X_val, y_val, _ = self.prepare_data(val_df)
        X_test, y_test, _ = self.prepare_data(test_df)
        
        # Prepare groups for test set
        test_groups = {name: groups[name][test_df.index] for name in groups.keys()}
        
        self.logger.info(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Train models
        trained_models = {}
        evaluation_results = {}
        
        # Traditional models
        traditional_models = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in traditional_models:
            model = self.train_traditional_model(model_name, X_train, y_train)
            trained_models[model_name] = model
            
            # Evaluate
            results = self.evaluate_model(model, X_test, y_test, test_groups, model_name)
            evaluation_results[model_name] = results
        
        # Deep models
        deep_models = ['tabnet', 'ft_transformer']
        
        for model_name in deep_models:
            model = self.train_deep_model(model_name, X_train, y_train, X_val, y_val)
            trained_models[model_name] = model
            
            # Evaluate
            results = self.evaluate_model(model, X_test, y_test, test_groups, model_name)
            evaluation_results[model_name] = results
        
        # Ensemble model
        if self.config['model']['name'] == 'ensemble':
            ensemble = self.train_ensemble(trained_models, X_train, y_train)
            trained_models['ensemble'] = ensemble
            
            # Evaluate ensemble
            results = self.evaluate_model(ensemble, X_test, y_test, test_groups, 'ensemble')
            evaluation_results['ensemble'] = results
        
        # Save models
        self.save_models(trained_models, feature_names)
        
        # Save evaluation results
        self.save_evaluation_results(evaluation_results)
        
        return {
            'models': trained_models,
            'evaluation_results': evaluation_results,
            'feature_names': feature_names
        }
    
    def save_models(self, models: Dict[str, Any], feature_names: List[str]) -> None:
        """Save trained models.
        
        Args:
            models: Dictionary of trained models.
            feature_names: List of feature names.
        """
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in models.items():
            if isinstance(model, nn.Module):
                # Save PyTorch model
                torch.save(model.state_dict(), f'models/{model_name}.pth')
            else:
                # Save sklearn model
                joblib.dump(model, f'models/{model_name}.pkl')
        
        # Save feature names
        joblib.dump(feature_names, 'models/feature_names.pkl')
        
        self.logger.info("Models saved successfully")
    
    def save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results.
        
        Args:
            results: Evaluation results.
        """
        os.makedirs('assets', exist_ok=True)
        
        # Save results
        joblib.dump(results, 'assets/evaluation_results.pkl')
        
        # Create summary
        summary = {}
        for model_name, result in results.items():
            summary[model_name] = {
                'auroc': result['clinical_metrics']['auroc'],
                'auprc': result['clinical_metrics']['auprc'],
                'sensitivity': result['clinical_metrics']['sensitivity'],
                'specificity': result['clinical_metrics']['specificity'],
                'ece': result['calibration']['ece']
            }
        
        # Save summary
        joblib.dump(summary, 'assets/model_summary.pkl')
        
        self.logger.info("Evaluation results saved successfully")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train medication adherence models')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
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
    
    # Create trainer
    trainer = ModelTrainer(config, device)
    
    # Generate data
    df, groups = trainer.generate_data()
    
    # Train models
    results = trainer.train_all_models(df, groups)
    
    logger.info("Training completed successfully!")
    
    # Print summary
    logger.info("\n=== MODEL PERFORMANCE SUMMARY ===")
    for model_name, result in results['evaluation_results'].items():
        metrics = result['clinical_metrics']
        logger.info(f"{model_name}: AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}")


if __name__ == '__main__':
    main()
