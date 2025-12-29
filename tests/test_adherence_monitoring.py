"""
Unit tests for medication adherence monitoring.

This module contains comprehensive tests for all components of the
medication adherence monitoring system.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import set_seed, get_device, PrivacyProtector, validate_config
from data import AdherenceDataGenerator, AdherenceDataProcessor
from models import (
    LogisticRegressionModel, RandomForestModel, XGBoostModel,
    TabNetModel, FTTransformerModel, EnsembleModel
)
from metrics import ClinicalMetrics, CalibrationAnalyzer, FairnessAnalyzer
from explainability import SafetyChecker


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy seed
        np.random.seed(42)
        assert np.random.random() == np.random.random()
        
        # Test torch seed
        torch.manual_seed(42)
        assert torch.rand(1) == torch.rand(1)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_privacy_protector(self):
        """Test privacy protection."""
        protector = PrivacyProtector(deid=True)
        
        # Test de-identification
        text = "Patient 123-45-6789 has email john@example.com"
        deidentified = protector.deidentify_text(text)
        assert "[REDACTED]" in deidentified
        assert "123-45-6789" not in deidentified
        assert "john@example.com" not in deidentified
    
    def test_validate_config(self):
        """Test configuration validation."""
        config = {'model': {'name': 'test'}}
        validated = validate_config(config)
        
        assert 'data' in validated
        assert 'training' in validated
        assert 'evaluation' in validated


class TestDataGenerator:
    """Test data generation."""
    
    def test_data_generator_init(self):
        """Test data generator initialization."""
        generator = AdherenceDataGenerator(seed=42)
        assert generator.seed == 42
    
    def test_generate_patient_data(self):
        """Test patient data generation."""
        generator = AdherenceDataGenerator(seed=42)
        df = generator.generate_patient_data(n_patients=10, n_days_per_patient=5)
        
        assert len(df) == 50  # 10 patients * 5 days
        assert 'patient_id' in df.columns
        assert 'adherence_risk' in df.columns
        assert df['adherence_risk'].isin([0, 1]).all()
    
    def test_add_temporal_features(self):
        """Test temporal feature addition."""
        generator = AdherenceDataGenerator(seed=42)
        df = generator.generate_patient_data(n_patients=5, n_days_per_patient=10)
        df_with_temporal = generator.add_temporal_features(df)
        
        # Check temporal features
        temporal_features = [
            'adherence_rate_3d', 'adherence_rate_7d', 'adherence_rate_14d',
            'on_time_rate_3d', 'on_time_rate_7d', 'on_time_rate_14d',
            'adherence_trend', 'day_of_week', 'is_weekend'
        ]
        
        for feature in temporal_features:
            assert feature in df_with_temporal.columns


class TestDataProcessor:
    """Test data processing."""
    
    def test_data_processor_init(self):
        """Test data processor initialization."""
        processor = AdherenceDataProcessor()
        assert processor.scalers == {}
        assert processor.encoders == {}
    
    def test_prepare_features(self):
        """Test feature preparation."""
        processor = AdherenceDataProcessor()
        
        # Create sample data
        df = pd.DataFrame({
            'age': [65, 70, 75],
            'gender': ['M', 'F', 'M'],
            'dose_taken': [1, 0, 1],
            'on_time': [1, 1, 0],
            'adherence_risk': [0, 1, 0]
        })
        
        static_features = ['age', 'gender']
        temporal_features = ['dose_taken', 'on_time']
        
        X, y = processor.prepare_features(df, static_features, temporal_features)
        
        assert X.shape[0] == 3
        assert X.shape[1] == 4  # 2 static + 2 temporal
        assert len(y) == 3
        assert processor.feature_names == static_features + temporal_features
    
    def test_patient_level_split(self):
        """Test patient-level data splitting."""
        processor = AdherenceDataProcessor()
        
        # Create sample data with multiple patients
        df = pd.DataFrame({
            'patient_id': [1, 1, 2, 2, 3, 3],
            'feature1': [1, 2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0, 1]
        })
        
        train_df, val_df, test_df = processor.patient_level_split(df, test_size=0.33, val_size=0.33)
        
        # Check that patients are not split across sets
        train_patients = set(train_df['patient_id'].unique())
        val_patients = set(val_df['patient_id'].unique())
        test_patients = set(test_df['patient_id'].unique())
        
        assert len(train_patients.intersection(val_patients)) == 0
        assert len(train_patients.intersection(test_patients)) == 0
        assert len(val_patients.intersection(test_patients)) == 0


class TestModels:
    """Test model implementations."""
    
    def test_logistic_regression_model(self):
        """Test logistic regression model."""
        model = LogisticRegressionModel(random_state=42)
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train model
        model.fit(X, y)
        assert model.is_fitted
        
        # Make predictions
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_random_forest_model(self):
        """Test random forest model."""
        model = RandomForestModel(random_state=42)
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train model
        model.fit(X, y)
        
        # Test feature importance
        importance = model.get_feature_importances()
        assert len(importance) == 5
        assert np.all(importance >= 0)
    
    def test_xgboost_model(self):
        """Test XGBoost model."""
        model = XGBoostModel(random_state=42)
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train model
        model.fit(X, y)
        
        # Test feature importance
        importance = model.get_feature_importances()
        assert len(importance) == 5
        assert np.all(importance >= 0)
    
    def test_tabnet_model(self):
        """Test TabNet model."""
        model = TabNetModel(input_dim=5, output_dim=2)
        
        # Create sample data
        X = torch.randn(100, 5)
        
        # Forward pass
        output, attention_masks = model(X)
        
        assert output.shape == (100, 2)
        assert len(attention_masks) == model.n_steps
    
    def test_ft_transformer_model(self):
        """Test FT-Transformer model."""
        model = FTTransformerModel(input_dim=5, output_dim=2)
        
        # Create sample data
        X = torch.randn(100, 5)
        
        # Forward pass
        output = model(X)
        
        assert output.shape == (100, 2)
    
    def test_ensemble_model(self):
        """Test ensemble model."""
        # Create base models
        model1 = LogisticRegressionModel(random_state=42)
        model2 = RandomForestModel(random_state=42)
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train ensemble
        ensemble = EnsembleModel([model1, model2], weights=[0.6, 0.4])
        ensemble.fit(X, y)
        
        # Make predictions
        predictions = ensemble.predict(X[:10])
        probabilities = ensemble.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 2)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_clinical_metrics(self):
        """Test clinical metrics calculation."""
        metrics = ClinicalMetrics()
        
        # Create sample data
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 1, 0])
        y_proba = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.4, 0.6], [0.8, 0.2]])
        
        results = metrics.calculate_all_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in results
        assert 'auroc' in results
        assert 'sensitivity' in results
        assert 'specificity' in results
        assert 'ppv' in results
        assert 'npv' in results
    
    def test_calibration_analyzer(self):
        """Test calibration analysis."""
        analyzer = CalibrationAnalyzer(n_bins=5)
        
        # Create sample data
        y_true = np.random.randint(0, 2, 1000)
        y_proba = np.random.rand(1000)
        
        results = analyzer.analyze_calibration(y_true, y_proba)
        
        assert 'fraction_of_positives' in results
        assert 'mean_predicted_value' in results
        assert 'ece' in results
        assert 'mce' in results
    
    def test_fairness_analyzer(self):
        """Test fairness analysis."""
        analyzer = FairnessAnalyzer()
        
        # Create sample data
        y_true = np.random.randint(0, 2, 1000)
        y_pred = np.random.randint(0, 2, 1000)
        y_proba = np.random.rand(1000, 2)
        groups = {'gender': np.random.choice(['M', 'F'], 1000)}
        
        results = analyzer.analyze_fairness(y_true, y_pred, y_proba, groups)
        
        assert 'gender' in results
        assert len(results['gender']) > 0


class TestExplainability:
    """Test explainability components."""
    
    def test_safety_checker(self):
        """Test safety checker."""
        checker = SafetyChecker()
        
        # Create sample data
        y_pred = np.random.randint(0, 2, 1000)
        y_proba = np.random.rand(1000, 2)
        
        results = checker.check_prediction_safety(y_pred, y_proba)
        
        assert 'safety_score' in results
        assert 'low_confidence_rate' in results
        assert 'high_uncertainty_rate' in results
        assert 0 <= results['safety_score'] <= 1


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline."""
        # Generate data
        generator = AdherenceDataGenerator(seed=42)
        df = generator.generate_patient_data(n_patients=50, n_days_per_patient=10)
        df = generator.add_temporal_features(df)
        
        # Process data
        processor = AdherenceDataProcessor()
        static_features = ['age', 'gender', 'comorbidities_count']
        temporal_features = ['dose_taken', 'on_time', 'missed_previous_day']
        
        X, y = processor.prepare_features(df, static_features, temporal_features)
        
        # Train model
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        
        # Evaluate
        metrics = ClinicalMetrics()
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        results = metrics.calculate_all_metrics(y, y_pred, y_proba)
        
        assert 'auroc' in results
        assert 'auprc' in results
        assert results['auroc'] > 0.5  # Should be better than random


if __name__ == '__main__':
    pytest.main([__file__])
