# PyTorch Learning Modules – CampusX

This repository contains my PyTorch learning notes and experiments while following the CampusX curriculum.

The repo is organized into independent modules (e.g. training pipeline, neural networks), each with its own notebook and environment.

<!-- --- -->

## General Setup (for any module)

1. **Navigate into the module folder**  
  ```bash
  cd <module-folder>
  ```

2. **Create and activate a virtual environment**  
  ```bash
  python -m venv .venv
  source .venv/bin/activate        # Linux / Mac
  # .venv\Scripts\activate 
  ```

3. **Install Dependencies**  
  ```bash
  pip install -r requirements.txt
  ```
4. **Select the environment as kernel**  
  In VS Code, select the `.venv` Python interpreter or the registered kernel.

> Each module manages its own environment. Do not reuse environments across folders.

<!-- --- -->

## Modules

### 1. Training Pipeline

Focuses on:
- Data preprocessing
- Train–test split
- Manual neural network implementation in PyTorch
- Training loops and evaluation
- CPU vs GPU experimentation

Folder: `1 - Training Pipeline/`

### 2. NN Module

Focuses on:
- Understanding how `nn.Module` works internally
- Building neural networks in a clean, modular structure
- Implementing a proper PyTorch training loop
- Applying best practices for binary classification

Folder: `2 - NN Module/`

### 3. Dataset & DataLoader Class

Focuses on:
- Understanding PyTorch’s `Dataset` and `DataLoader` abstractions
- Implementing custom datasets using `__len__` and `__getitem__`
- Mini-batch training and data shuffling
- Integrating `DataLoader` cleanly into a training pipeline

Folder: `3 - Dataset & DataLoader Class/`

### 4. Artificial Neural Networks (ANN)

Focuses on:
- Implementing ANN architectures using `nn.Module` and `nn.Sequential`
- Writing clean training and evaluation loops
- Using GPU acceleration for faster training
- Applying regularization techniques (Dropout, BatchNorm, L2)
- Designing dynamic networks programmatically
- Automating hyperparameter tuning with Optuna

Folder: `4 - ANN/`

### 5. Convolutional Neural Networks (CNN)

Focuses on:
- Transition from ANN to CNN architectures
- Learning convolutional feature extraction using `Conv2d`, `BatchNorm2d`, and pooling
- Working with image-shaped tensors in PyTorch
- Applying CNNs to Fashion-MNIST
- Hyperparameter optimization with Optuna

Folder: `5 - CNN/`

### 6. Transfer Learning

Focuses on:
- Using pretrained CNNs for new tasks
- Adapting ImageNet-trained models to Fashion-MNIST
- Freezing backbone layers and training custom classifiers
- Image preprocessing for transfer learning
- Efficient GPU-based fine-tuning

Model used:
- **VGG16 (pretrained on ImageNet)**

Folder: `6 - Transfer Learning/`

### 7. Recurrent Neural Networks (RNN)

Focuses on:
- Understanding sequence modeling fundamentals
- Working with text data in PyTorch
- Tokenization, vocabulary creation, and embeddings
- Implementing RNNs from scratch for NLP tasks
- Building a simple Question Answering system

Folder: `7 - RNN/`

### 8. Long Short-Term Memory Networks (LSTM)

Focuses on:
- Understanding LSTM internals (hidden state & cell state)
- Sequence modeling for language tasks
- Next-word prediction using autoregressive generation
- Handling variable-length text sequences
- Padding and sliding-window training strategies

Folder: `8 - LSTM/`

<!-- --- -->

## Notes
- `.venv` folders are intentionally ignored via `.gitignore`
- No package installation commands are kept inside notebooks
- Dependencies are defined only via `requirements.txt`

<!-- --- -->

## Requirements
- Python 3.10+
- NVIDIA GPU (optional, for CUDA experiments)