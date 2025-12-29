# Quick Start Guide

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd medication-adherence-monitoring
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up pre-commit hooks** (optional):
   ```bash
   python setup_precommit.py
   ```

## Quick Start

### 1. Train Models

Train all models on synthetic data:

```bash
python scripts/train.py --config configs/default.yaml
```

This will:
- Generate synthetic longitudinal adherence data
- Train multiple models (Logistic Regression, Random Forest, XGBoost, TabNet, FT-Transformer)
- Save trained models to `models/` directory
- Generate evaluation results

### 2. Evaluate Models

Run comprehensive evaluation:

```bash
python scripts/evaluate.py --model_path models --output_dir assets
```

This will:
- Load trained models
- Generate test data
- Calculate clinical metrics, calibration, fairness, and safety
- Create visualization plots
- Generate evaluation report

### 3. Launch Interactive Demo

Start the Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo provides:
- Patient risk assessment interface
- Model performance comparison
- Feature importance analysis
- Interactive visualizations

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data processing
│   ├── metrics/           # Evaluation metrics
│   ├── explainability/    # SHAP, uncertainty, safety
│   └── utils/             # Utilities
├── configs/               # Configuration files
├── scripts/               # Training/evaluation scripts
├── demo/                  # Streamlit demo
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
└── models/                # Trained models
```

## Configuration

Edit `configs/default.yaml` to customize:
- Data generation parameters
- Model configurations
- Training settings
- Evaluation metrics
- Privacy settings

## Key Features

### Models
- **Baseline**: Logistic Regression, Random Forest
- **Advanced**: XGBoost, TabNet, FT-Transformer
- **Ensemble**: Weighted combination

### Evaluation
- **Clinical Metrics**: AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV
- **Calibration**: Expected Calibration Error, Reliability Diagrams
- **Fairness**: Performance across demographic groups
- **Safety**: Confidence and uncertainty analysis

### Explainability
- **SHAP**: Feature importance and instance explanations
- **Uncertainty**: Model confidence estimation
- **Safety Checks**: Prediction reliability assessment

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Important Notes

- **Research Only**: This is a demonstration project with synthetic data
- **Not Clinical**: Not intended for medical use or diagnosis
- **Privacy**: Built-in de-identification utilities
- **Safety**: Comprehensive safety checks and uncertainty quantification

## Troubleshooting

### Common Issues

1. **CUDA/MPS Issues**: The system auto-detects available devices (CUDA → MPS → CPU)
2. **Memory Issues**: Reduce batch size in config or use CPU
3. **Import Errors**: Ensure all dependencies are installed
4. **Model Loading**: Run training script first to generate models

### Getting Help

- Check the logs in `logs/` directory
- Review configuration in `configs/default.yaml`
- Run tests to verify installation
- Check the demo for interactive exploration

## Next Steps

1. **Customize Data**: Modify data generation in `src/data/`
2. **Add Models**: Implement new models in `src/models/`
3. **Extend Metrics**: Add evaluation metrics in `src/metrics/`
4. **Enhance Demo**: Customize the Streamlit interface
5. **Deploy**: Set up production deployment (not for clinical use)

Remember: This is a research demonstration. For clinical applications, ensure appropriate validation and regulatory compliance.
