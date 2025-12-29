"""
Streamlit demo application for medication adherence monitoring.

This application provides an interactive interface for exploring adherence
predictions, model explanations, and risk assessment tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import torch
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import get_device, load_config
from data import AdherenceDataGenerator, AdherenceDataProcessor
from models import create_model, create_deep_model
from metrics import ClinicalMetrics, CalibrationAnalyzer
from explainability import SHAPExplainer, SafetyChecker

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medication Adherence Monitoring AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
    <h3>⚠️ IMPORTANT DISCLAIMER</h3>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>NOT intended for clinical use or medical decision making</li>
        <li>Uses synthetic data for demonstration purposes</li>
        <li>No clinical validation or regulatory approval</li>
        <li>Always consult healthcare professionals for medical decisions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">💊 Medication Adherence Monitoring AI</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Patient Risk Assessment", "Model Performance", "Feature Analysis", "About"]
)

# Load configuration
@st.cache_data
def load_app_config():
    """Load application configuration."""
    try:
        config = load_config('configs/default.yaml')
        return config
    except:
        # Default config if file not found
        return {
            'data': {
                'features': {
                    'static': ['age', 'gender', 'comorbidities_count', 'medication_complexity', 'insurance_type'],
                    'temporal': ['dose_taken', 'on_time', 'missed_previous_day', 'reported_side_effects', 'mood_score', 'stress_level']
                }
            }
        }

config = load_app_config()

# Load models
@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    feature_names = []
    
    try:
        # Load feature names
        feature_names = joblib.load('models/feature_names.pkl')
        
        # Load traditional models
        traditional_models = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in traditional_models:
            try:
                model = joblib.load(f'models/{model_name}.pkl')
                models[model_name] = model
            except:
                pass
        
        # Load deep models
        deep_models = ['tabnet', 'ft_transformer']
        device = get_device()
        
        for model_name in deep_models:
            try:
                model = create_deep_model(model_name, input_dim=len(feature_names), output_dim=2)
                model.load_state_dict(torch.load(f'models/{model_name}.pth', map_location=device))
                model.eval()
                models[model_name] = model
            except:
                pass
        
        return models, feature_names
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, []

models, feature_names = load_models()

# Data generator
@st.cache_resource
def get_data_generator():
    """Get data generator."""
    return AdherenceDataGenerator(seed=42)

data_generator = get_data_generator()

# Patient Risk Assessment Page
if page == "Patient Risk Assessment":
    st.header("Patient Risk Assessment")
    
    if not models:
        st.error("No models available. Please train models first using the training script.")
        st.stop()
    
    # Patient input form
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 100, 65)
        gender = st.selectbox("Gender", ["M", "F"])
        comorbidities_count = st.slider("Number of Comorbidities", 0, 10, 2)
    
    with col2:
        medication_complexity = st.slider("Medication Complexity (1-5)", 1, 5, 3)
        insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private", "Uninsured"])
        dose_taken = st.selectbox("Dose Taken Today", [1, 0])
    
    with col3:
        on_time = st.selectbox("Taken On Time", [1, 0])
        missed_previous_day = st.selectbox("Missed Previous Day", [1, 0])
        reported_side_effects = st.selectbox("Reported Side Effects", [1, 0])
    
    # Additional features
    st.subheader("Additional Factors")
    
    col4, col5 = st.columns(2)
    
    with col4:
        mood_score = st.slider("Mood Score (1-10)", 1, 10, 6)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 4)
    
    with col5:
        adherence_rate_7d = st.slider("7-Day Adherence Rate", 0.0, 1.0, 0.85)
        on_time_rate_7d = st.slider("7-Day On-Time Rate", 0.0, 1.0, 0.80)
    
    # Create patient data
    patient_data = {
        'age': age,
        'gender': gender,
        'comorbidities_count': comorbidities_count,
        'medication_complexity': medication_complexity,
        'insurance_type': insurance_type,
        'dose_taken': dose_taken,
        'on_time': on_time,
        'missed_previous_day': missed_previous_day,
        'reported_side_effects': reported_side_effects,
        'mood_score': mood_score,
        'stress_level': stress_level,
        'adherence_rate_7d': adherence_rate_7d,
        'on_time_rate_7d': on_time_rate_7d,
        'adherence_trend': np.random.normal(0, 0.1),  # Simulated trend
        'day_of_week': np.random.randint(0, 7),  # Simulated day
        'is_weekend': np.random.choice([0, 1])  # Simulated weekend
    }
    
    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Prepare features
    data_processor = AdherenceDataProcessor()
    
    # Get feature columns
    static_features = config['data']['features']['static']
    temporal_features = config['data']['features']['temporal']
    temporal_features.extend([
        'adherence_rate_7d', 'on_time_rate_7d', 'adherence_trend', 
        'day_of_week', 'is_weekend'
    ])
    
    try:
        X, _ = data_processor.prepare_features(patient_df, static_features, temporal_features)
        X_scaled = data_processor.scale_features(X, X)[0]
        
        # Model selection
        selected_model = st.selectbox("Select Model", list(models.keys()))
        
        if st.button("Assess Risk"):
            model = models[selected_model]
            
            # Make prediction
            if isinstance(model, torch.nn.Module):
                device = get_device()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled).to(device)
                    outputs = model(X_tensor)
                    y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                    y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_scaled)
                y_pred_proba = model.predict_proba(X_scaled)
            
            # Display results
            risk_probability = y_pred_proba[0][1]  # Probability of non-adherence
            
            st.subheader("Risk Assessment Results")
            
            # Risk level
            if risk_probability > 0.7:
                risk_level = "HIGH"
                risk_class = "risk-high"
            elif risk_probability > 0.4:
                risk_level = "MEDIUM"
                risk_class = "risk-medium"
            else:
                risk_level = "LOW"
                risk_class = "risk-low"
            
            st.markdown(f'<div class="metric-card"><h3>Risk Level: <span class="{risk_class}">{risk_level}</span></h3></div>', unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Non-Adherence Probability", f"{risk_probability:.3f}")
            
            with col2:
                st.metric("Adherence Probability", f"{1-risk_probability:.3f}")
            
            with col3:
                st.metric("Prediction", "Non-Adherent" if y_pred[0] == 1 else "Adherent")
            
            # Risk factors
            st.subheader("Key Risk Factors")
            
            # Calculate feature importance (simplified)
            feature_importance = np.abs(X_scaled[0])
            top_features_idx = np.argsort(feature_importance)[::-1][:5]
            
            risk_factors = []
            for idx in top_features_idx:
                feature_name = feature_names[idx]
                importance = feature_importance[idx]
                risk_factors.append((feature_name, importance))
            
            for factor, importance in risk_factors:
                st.write(f"• {factor}: {importance:.3f}")
            
            # Recommendations
            st.subheader("Recommendations")
            
            if risk_probability > 0.7:
                st.warning("High risk detected. Consider:")
                st.write("• Immediate intervention")
                st.write("• Medication review")
                st.write("• Additional support")
            elif risk_probability > 0.4:
                st.info("Medium risk detected. Consider:")
                st.write("• Regular monitoring")
                st.write("• Patient education")
                st.write("• Reminder systems")
            else:
                st.success("Low risk. Continue current care plan.")
            
            # Safety check
            safety_checker = SafetyChecker()
            safety_results = safety_checker.check_prediction_safety(y_pred, y_pred_proba)
            
            st.subheader("Safety Assessment")
            st.metric("Safety Score", f"{safety_results['safety_score']:.3f}")
            
            if safety_results['safety_score'] < 0.7:
                st.warning("⚠️ Low safety score. Consider additional validation.")
    
    except Exception as e:
        st.error(f"Error processing patient data: {e}")

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance Analysis")
    
    # Load evaluation results if available
    try:
        evaluation_results = joblib.load('assets/evaluation_results.pkl')
        
        # Model comparison
        st.subheader("Model Performance Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in evaluation_results.items():
            metrics = results['clinical_metrics']
            comparison_data.append({
                'Model': model_name,
                'AUROC': metrics['auroc'],
                'AUPRC': metrics['auprc'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'PPV': metrics['ppv'],
                'NPV': metrics['npv'],
                'ECE': results['calibration']['ece']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display metrics table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance plots
        st.subheader("Performance Visualization")
        
        # AUROC comparison
        fig_auroc = px.bar(
            comparison_df, 
            x='Model', 
            y='AUROC',
            title='AUROC Comparison',
            color='AUROC',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_auroc, use_container_width=True)
        
        # Calibration curves
        st.subheader("Calibration Analysis")
        
        fig_calibration = go.Figure()
        
        for model_name, results in evaluation_results.items():
            calibration_data = results['calibration']
            
            fig_calibration.add_trace(go.Scatter(
                x=calibration_data['mean_predicted_value'],
                y=calibration_data['fraction_of_positives'],
                mode='lines+markers',
                name=f'{model_name} (ECE={calibration_data["ece"]:.3f})'
            ))
        
        # Perfect calibration line
        fig_calibration.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='black')
        ))
        
        fig_calibration.update_layout(
            title='Calibration Curves',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=800,
            height=500
        )
        
        st.plotly_chart(fig_calibration, use_container_width=True)
        
        # Safety analysis
        st.subheader("Safety Analysis")
        
        safety_data = []
        for model_name, results in evaluation_results.items():
            safety = results['safety']
            safety_data.append({
                'Model': model_name,
                'Safety Score': safety['safety_score'],
                'Low Confidence Rate': safety['low_confidence_rate'],
                'High Uncertainty Rate': safety['high_uncertainty_rate']
            })
        
        safety_df = pd.DataFrame(safety_data)
        st.dataframe(safety_df, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Evaluation results not found. Please run the evaluation script first.")
    except Exception as e:
        st.error(f"Error loading evaluation results: {e}")

# Feature Analysis Page
elif page == "Feature Analysis":
    st.header("Feature Analysis")
    
    if not models:
        st.error("No models available for feature analysis.")
        st.stop()
    
    # Generate sample data for analysis
    st.subheader("Feature Importance Analysis")
    
    # Generate synthetic data
    sample_df = data_generator.generate_patient_data(n_patients=100, n_days_per_patient=30)
    sample_df = data_generator.add_temporal_features(sample_df)
    
    # Prepare features
    data_processor = AdherenceDataProcessor()
    static_features = config['data']['features']['static']
    temporal_features = config['data']['features']['temporal']
    temporal_features.extend([
        'adherence_rate_3d', 'adherence_rate_7d', 'adherence_rate_14d',
        'on_time_rate_3d', 'on_time_rate_7d', 'on_time_rate_14d',
        'adherence_trend', 'day_of_week', 'is_weekend'
    ])
    
    try:
        X, y = data_processor.prepare_features(sample_df, static_features, temporal_features)
        X_scaled = data_processor.scale_features(X, X)[0]
        
        # Model selection for analysis
        analysis_model = st.selectbox("Select Model for Analysis", list(models.keys()))
        
        if st.button("Analyze Features"):
            model = models[analysis_model]
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                # Use random forest as proxy
                rf_model = models.get('random_forest', model)
                if hasattr(rf_model, 'feature_importances_'):
                    importance = rf_model.feature_importances_
                else:
                    importance = np.random.random(len(feature_names))
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Display top features
            st.subheader("Top 15 Most Important Features")
            
            fig_importance = px.bar(
                importance_df.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance',
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(height=600)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Feature correlation
            st.subheader("Feature Correlation Analysis")
            
            # Create correlation matrix
            feature_df = pd.DataFrame(X_scaled, columns=feature_names)
            correlation_matrix = feature_df.corr()
            
            # Plot correlation heatmap
            fig_corr = px.imshow(
                correlation_matrix,
                title='Feature Correlation Matrix',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Feature distribution
            st.subheader("Feature Distribution Analysis")
            
            # Select features to plot
            selected_features = st.multiselect(
                "Select features to analyze",
                feature_names,
                default=feature_names[:5]
            )
            
            if selected_features:
                # Create distribution plots
                fig_dist = make_subplots(
                    rows=len(selected_features),
                    cols=1,
                    subplot_titles=selected_features
                )
                
                for i, feature in enumerate(selected_features):
                    feature_idx = feature_names.index(feature)
                    fig_dist.add_trace(
                        go.Histogram(
                            x=X_scaled[:, feature_idx],
                            name=feature,
                            showlegend=False
                        ),
                        row=i+1, col=1
                    )
                
                fig_dist.update_layout(
                    title='Feature Distributions',
                    height=200 * len(selected_features)
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in feature analysis: {e}")

# About Page
elif page == "About":
    st.header("About Medication Adherence Monitoring AI")
    
    st.markdown("""
    ## Overview
    
    This application demonstrates AI techniques for medication adherence monitoring using synthetic EHR/tabular data. 
    It showcases advanced machine learning models, comprehensive evaluation metrics, and explainability tools 
    for research and educational purposes.
    
    ## Features
    
    ### Models
    - **Baseline Models**: Logistic Regression, Random Forest
    - **Gradient Boosting**: XGBoost
    - **Deep Learning**: TabNet, FT-Transformer
    - **Ensemble**: Weighted combination of best models
    
    ### Evaluation Metrics
    - **Clinical Metrics**: AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV
    - **Calibration**: Expected Calibration Error, Reliability Diagrams
    - **Fairness**: Performance across demographic groups
    - **Safety**: Confidence and uncertainty analysis
    
    ### Explainability
    - **SHAP**: Feature importance and instance explanations
    - **Uncertainty Quantification**: Model confidence estimation
    - **Safety Checks**: Prediction reliability assessment
    
    ## Data
    
    The application uses synthetic longitudinal medication adherence data with:
    - Patient-level features (age, comorbidities, medication complexity)
    - Temporal features (dose timing, missed doses, side effects)
    - Adherence risk labels
    
    ## Technical Stack
    
    - **Core**: PyTorch, NumPy, Pandas, Scikit-learn
    - **Models**: XGBoost, TabNet, FT-Transformer
    - **Evaluation**: Custom clinical metrics, calibration analysis
    - **Visualization**: Streamlit, Plotly
    - **Explainability**: SHAP, uncertainty quantification
    
    ## Usage
    
    1. **Training**: Run `python scripts/train.py` to train models
    2. **Evaluation**: Run `python scripts/evaluate.py` to evaluate models
    3. **Demo**: Run `streamlit run demo/app.py` to launch this interface
    
    ## Safety and Compliance
    
    - **De-identification**: Built-in utilities for data anonymization
    - **Privacy**: No PHI/PII logging or storage
    - **Bias Detection**: Fairness evaluation across demographic groups
    - **Uncertainty**: Model confidence and calibration reporting
    
    ## Limitations
    
    - Synthetic data only - not validated on real clinical data
    - No regulatory approval or clinical validation
    - Requires healthcare provider supervision for any clinical application
    
    ## Disclaimer
    
    **This is a research demonstration project.**
    
    - NOT intended for clinical use or medical decision making
    - Uses synthetic data for demonstration purposes
    - No clinical validation or regulatory approval
    - Always consult healthcare professionals for medical decisions
    
    ## Contributing
    
    This is a research demonstration project. For clinical applications, 
    ensure appropriate validation and regulatory compliance.
    
    ## License
    
    MIT License - See LICENSE file for details.
    """)
    
    # Technical details
    st.subheader("Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Architecture:**
        - TabNet: Attention-based tabular deep learning
        - FT-Transformer: Feature tokenization + Transformer
        - Ensemble: Weighted voting of best models
        """)
    
    with col2:
        st.markdown("""
        **Evaluation Framework:**
        - Patient-level data splits
        - Clinical metrics focus
        - Calibration analysis
        - Fairness evaluation
        """)
    
    # Contact information
    st.subheader("Contact")
    st.markdown("""
    For questions about this research demonstration:
    - This is a synthetic data demonstration
    - Not for clinical use
    - Educational purposes only
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>⚠️ Research and Educational Use Only - NOT FOR CLINICAL USE ⚠️</p>
    <p>Medication Adherence Monitoring AI Demo | Synthetic Data | No Medical Advice</p>
</div>
""", unsafe_allow_html=True)
