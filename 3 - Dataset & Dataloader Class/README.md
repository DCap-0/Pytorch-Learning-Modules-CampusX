# Dataset & DataLoader Class

This module introduces PyTorch’s `Dataset` and `DataLoader` abstractions and demonstrates how they are used in a real training pipeline.

The focus is on understanding **how data flows from raw arrays to mini-batches**, and how this integrates cleanly with PyTorch training loops.

<!-- --- -->

## Contents

- `dataset-and-dataloader-demo.ipynb`  
  A minimal, synthetic example using `sklearn.make_classification` to:
  - Build a custom `Dataset`
  - Understand `__len__` and `__getitem__`
  - Iterate over batches using `DataLoader`

- `training-pipeline-using-dataset-&-dataloader.ipynb`  
  A complete binary classification pipeline that:
  - Uses a real-world dataset (Breast Cancer dataset)
  - Applies preprocessing (scaling, encoding)
  - Wraps data using custom `Dataset` classes
  - Trains a neural network using mini-batch SGD via `DataLoader`

<!-- --- -->

## Key Concepts Covered

- Why `Dataset` exists and what problem it solves
- How `__getitem__` enables lazy loading
- Mini-batch training with `DataLoader`
- Shuffling and batching mechanics
- Clean separation of:
  - Data logic
  - Model definition
  - Training loop

<!-- --- -->

## Setup

Follow the same setup steps defined in the **main repository README**:

1. Create and activate a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

3. Select `.venv` as the kernel in VS Code.

<!-- --- -->

## Notes

- `.venv` is intentionally ignored via `.gitignore`
- No package installation commands are kept inside notebooks
- This module builds on concepts introduced in earlier NN modules

<!-- --- -->