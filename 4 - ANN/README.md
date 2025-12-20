# Artificial Neural Networks (ANN) – Fashion MNIST

This folder contains a progressive implementation of **Artificial Neural Networks (ANNs)** using **PyTorch**, trained on a subset of the **Fashion-MNIST** dataset.

The notebooks are structured to reflect an incremental learning approach — starting from a basic ANN and gradually introducing GPU acceleration, regularization techniques, and automated hyperparameter tuning.

<!-- --- -->

## Contents

### 1. Baseline ANN (CPU)
**Notebook:** `01_ann_fmnist_baseline_cpu.ipynb`  
- Manual data loading from CSV
- Custom `Dataset` and `DataLoader`
- Fully connected ANN using `nn.Sequential`
- Training and evaluation loops from scratch
- Accuracy evaluation on train and test sets

### 2. GPU-Accelerated Training
**Notebook:** `02_ann_fmnist_gpu.ipynb`  
- CUDA device handling (`.to(device)`)
- Optimized batch sizes
- `pin_memory` usage for faster GPU transfer
- Same architecture, faster execution

### 3. Regularized ANN
**Notebook:** `03_ann_fmnist_regularized.ipynb`  
- Added **Batch Normalization**
- Added **Dropout**
- **L2 Regularization** via `weight_decay`
- Reduced overfitting and improved generalization

### 4. Optuna – Architecture Search (Experimental)
**Notebook:** `04_ann_fmnist_optuna_architecture_search.ipynb`
- Dynamic ANN construction
- Hyperparameter tuning for:
  - Number of hidden layers
  - Neurons per layer
- Fixed optimizer and learning rate
- Experimental exploration of model capacity

### 5. Optuna – Full Training Pipeline
**Notebook:** `05_ann_fmnist_optuna_full_pipeline.ipynb`
- End-to-end hyperparameter tuning with Optuna
- Search space includes:
  - Network depth & width
  - Dropout rate
  - Optimizer choice (Adam, SGD, RMSprop)
  - Learning rate
  - Batch size
  - Weight decay
- Final model trained using best Optuna parameters
- Clean separation of:
  - Objective function
  - Final training
  - Evaluation

<!-- --- -->

## Dataset
- `fmnist_small.csv`
- Flattened 28×28 grayscale images
- 10-class classification problem

<!-- --- -->

## Key Concepts Covered
- ANN fundamentals in PyTorch
- Custom `Dataset` and `DataLoader`
- GPU acceleration
- Regularization techniques
- Modular training pipelines
- Hyperparameter optimization with Optuna

<!-- --- -->

## Requirements
Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

<!-- --- -->

> GPU support requires a CUDA-enabled PyTorch installation.