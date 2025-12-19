# Training Pipeline

This module implements a **basic end-to-end training pipeline in PyTorch** from scratch, without using `torch.nn` or optimizers.  
The goal is to understand **how training actually works under the hood**.

<!-- --- -->

## Overview

The notebook walks through:
- Loading a real-world dataset
- Preprocessing using NumPy and scikit-learn
- Manual tensor conversion
- Implementing a simple neural network using raw tensors
- Explicit forward pass, loss computation, and backpropagation
- Manual weight updates using gradients
- Comparing CPU vs GPU training time

<!-- --- -->

## Dataset

- **Breast Cancer Wisconsin Dataset**
- Loaded directly from a public GitHub CSV
- Binary classification problem

<!-- --- -->

## Key Concepts Covered

- NumPy → PyTorch tensor conversion
- `requires_grad` and autograd mechanics
- Manual implementation of:
  - Linear layer
  - Sigmoid activation
  - Binary cross-entropy loss
- Gradient computation with `loss.backward()`
- Parameter updates using `torch.no_grad()`
- Device-aware training (`cpu` vs `cuda`)
- Measuring training time for CPU vs GPU

<!-- --- -->

## Project Structure
  ```
  Training-Pipeline/
  ├── notebook.ipynb
  ├── requirements.txt
  ├── README.md
  └── .venv/ (ignored)
  ```

<!-- --- -->

## Setup

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
  Select the `.venv` interpreter as the kernel in VS Code before running the notebook.