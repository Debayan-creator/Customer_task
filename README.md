# Customer Support Ticket Auto-Triage System

## Overview
An advanced machine learning solution for automating customer support ticket classification using TensorFlow.

## Model Performance
- **Accuracy**: 21.00%
- **F1-Score**: 7.29%
- **Average Latency**: 87.82 ms

## Installation
```bash
pip install tensorflow pandas numpy scikit-learn flask
```

## Quick Start

### Training
```python
# Run the Jupyter notebook
jupyter notebook ticket_triage.ipynb
```

### API Deployment
```bash
python app.py
```

### API Usage
```python
import requests

response = requests.post(
    'http://localhost:5000/classify',
    json={
        'subject': 'Login issue',
        'description': 'Cannot access my account'
    }
)
print(response.json())
```

## Categories
- Bug Report
- Feature Request
- Technical Issue
- Billing Inquiry
- Account Management

## Files
- `ticket_triage_model/` - Trained TensorFlow model
- `label_encoder.pkl` - Label encoder for categories
- `model_config.json` - Model configuration and metrics
- `app.py` - Flask API server
- `requirements.txt` - Python dependencies
