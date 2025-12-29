"""
Core utilities for medication adherence monitoring.

This module provides essential utilities for device management, seeding,
logging, and privacy protection.
"""

import os
import random
import logging
import warnings
from typing import Optional, Union, Dict, Any
import torch
import numpy as np
import pandas as pd
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device with fallback.
    
    Args:
        device: Specific device to use. If None, auto-detect.
        
    Returns:
        torch.device: The selected device.
    """
    if device is not None:
        return torch.device(device)
    
    # Auto-detect device with fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    tensorboard: bool = True
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level.
        log_dir: Directory for log files.
        tensorboard: Whether to enable TensorBoard logging.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(log_dir) / 'adherence_monitoring.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('adherence_monitoring')
    
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging enabled: {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available")
    
    return logger


class PrivacyProtector:
    """Privacy protection utilities for handling sensitive data."""
    
    def __init__(self, deid: bool = True, redact_identifiers: bool = True):
        """Initialize privacy protector.
        
        Args:
            deid: Enable de-identification.
            redact_identifiers: Redact common identifiers.
        """
        self.deid = deid
        self.redact_identifiers = redact_identifiers
        
        # Common identifier patterns
        self.identifier_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
    
    def deidentify_text(self, text: str) -> str:
        """De-identify text by removing common identifiers.
        
        Args:
            text: Input text.
            
        Returns:
            str: De-identified text.
        """
        if not self.deid:
            return text
            
        import re
        
        deidentified = text
        for pattern in self.identifier_patterns:
            deidentified = re.sub(pattern, '[REDACTED]', deidentified)
        
        return deidentified
    
    def safe_log(self, data: Any, logger: logging.Logger) -> None:
        """Safely log data without PHI/PII.
        
        Args:
            data: Data to log.
            logger: Logger instance.
        """
        if isinstance(data, str):
            safe_data = self.deidentify_text(data)
        elif isinstance(data, dict):
            safe_data = {k: self.deidentify_text(str(v)) if isinstance(v, str) else v 
                        for k, v in data.items()}
        else:
            safe_data = data
            
        logger.info(f"Safe data: {safe_data}")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration parameters.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Dict[str, Any]: Validated configuration.
    """
    # Set defaults
    defaults = {
        'data': {
            'synthetic': {'n_patients': 1000, 'n_days_per_patient': 30, 'seed': 42},
            'features': {
                'static': ['age', 'gender', 'comorbidities_count', 'medication_complexity'],
                'temporal': ['dose_taken', 'on_time', 'missed_previous_day', 'reported_side_effects']
            }
        },
        'model': {'name': 'ensemble'},
        'training': {'batch_size': 64, 'learning_rate': 0.001, 'epochs': 100},
        'evaluation': {'metrics': ['auroc', 'auprc', 'sensitivity', 'specificity']},
        'device': {'auto': True, 'fallback': True},
        'privacy': {'deid': True, 'redact_identifiers': True, 'log_phi': False}
    }
    
    # Merge with defaults
    def deep_merge(d1: Dict, d2: Dict) -> Dict:
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                deep_merge(d1[key], value)
            else:
                d1[key] = value
        return d1
    
    validated_config = deep_merge(defaults.copy(), config)
    return validated_config


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Dict[str, Any]: Loaded configuration.
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return validate_config(config)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration.
    """
    import yaml
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
