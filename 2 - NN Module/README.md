# NN Module

This folder demonstrates **binary classification using PyTorch’s `nn.Module` API**, focusing on building neural networks, defining training pipelines, and understanding best practices for model training and evaluation.

The examples use a **simple neural network** implemented with `nn.Module` and are intended for learning and revision purposes.

<!-- --- -->

## Folder Structure
  ```
  NN Module/
  ├── .venv/ (ignored)
  ├── pytorch-nn-module.ipynb
  ├── training-pipeline-using-nn-module.ipynb
  ├── requirements.txt
  ├── requirements-lock.txt
  └── README.md
  ```

<!-- --- -->

## Contents

### 1. `pytorch-nn-module.ipynb`
- Introduction to PyTorch `nn.Module`
- Defining a custom neural network class
- Understanding:
  - `__init__`
  - `forward()`
  - Model parameters and weights
- Basic forward pass and output interpretation

### 2. `training-pipeline-using-nn-module.ipynb`
- End-to-end training pipeline using `nn.Module`
- Steps covered:
  - Data loading and preprocessing
  - Train–test split
  - Feature scaling
  - Label encoding
  - Model training loop
  - Loss computation (`BCEWithLogitsLoss`)
  - Backpropagation
  - Optimizer-based weight updates
  - Model evaluation and accuracy calculation

<!-- --- -->

## Environment Setup

### Create and activate virtual environment
  ```bash
  python -m venv .venv
  source .venv/bin/activate        # Linux / macOS
  .venv\Scripts\activate           # Windows
  ```

### Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

  For fully reproducible environments:
  ```bash
  pip install -r requirements-lock.txt
  ```

<!-- --- -->

## Learning Objectives
- Understand how `nn.Module` works internally
- Build neural networks in a clean, modular way
- Implement a proper PyTorch training loop
- Learn best practices for binary classification

<!-- --- -->

## Notes
- `.venv` is excluded from version control and can be safely deleted and recreated.
- These notebooks are designed for **learning and revision**, not production deployment.
- Code prioritizes clarity and correctness over abstraction.

<!-- --- -->

## Recommended Usage  
Start with:
1. `pytorch-nn-module.ipynb`
2. `training-pipeline-using-nn-module.ipynb`  

to progressively understand model definition → training pipeline.