# Medication Adherence Monitoring AI

**⚠️ RESEARCH AND EDUCATIONAL USE ONLY - NOT FOR CLINICAL USE ⚠️**

This project demonstrates AI techniques for medication adherence monitoring using EHR/tabular data. It includes advanced machine learning models, comprehensive evaluation metrics, and explainability tools for research and educational purposes.

## Features

- **Advanced Models**: Gradient boosting baselines + deep tabular models (TabNet, FT-Transformer)
- **Comprehensive Evaluation**: Clinical metrics, calibration analysis, fairness evaluation
- **Explainability**: SHAP explanations, uncertainty quantification
- **Interactive Demo**: Streamlit interface for exploration and visualization
- **Production Ready**: Proper structure, configs, tests, and documentation

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training**:
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

3. **Launch Demo**:
   ```bash
   streamlit run demo/app.py
   ```

4. **Run Evaluation**:
   ```bash
   python scripts/evaluate.py --model_path models/best_model.pth
   ```

## Dataset

The project uses synthetic longitudinal medication adherence data with:
- Patient-level features (age, comorbidities, medication complexity)
- Temporal features (dose timing, missed doses, side effects)
- Adherence risk labels

## Models

- **Baseline**: Logistic Regression, Random Forest, XGBoost
- **Advanced**: TabNet, FT-Transformer
- **Ensemble**: Weighted combination of best models

## Evaluation Metrics

- **Classification**: AUROC, AUPRC, Sensitivity/Specificity, PPV/NPV
- **Calibration**: Brier Score, Expected Calibration Error
- **Fairness**: Performance by demographic groups
- **Clinical**: Decision curve analysis

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data processing
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── utils/             # Utilities
│   ├── train.py           # Training script
│   └── eval.py            # Evaluation script
├── configs/               # Configuration files
├── scripts/               # Training/evaluation scripts
├── demo/                  # Streamlit demo
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
└── data/                  # Data directory
```

## Safety and Compliance

- **De-identification**: Built-in utilities for data anonymization
- **Privacy**: No PHI/PII logging or storage
- **Bias Detection**: Fairness evaluation across demographic groups
- **Uncertainty**: Model confidence and calibration reporting

## Limitations

- Synthetic data only - not validated on real clinical data
- No regulatory approval or clinical validation
- Requires healthcare provider supervision for any clinical application

## Contributing

This is a research demonstration project. For clinical applications, ensure appropriate validation and regulatory compliance.

## License

MIT License - See LICENSE file for details.
# Medication-Adherence-Monitoring-AI
