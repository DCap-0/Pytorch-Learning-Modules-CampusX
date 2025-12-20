# Convolutional Neural Networks (CNN) – Fashion MNIST

This folder contains implementations of **Convolutional Neural Networks (CNNs)** using **PyTorch**, applied to the **Fashion-MNIST** dataset.

The focus is on transitioning from fully connected ANNs to CNNs, understanding convolutional feature extraction, and training models efficiently using GPU acceleration.

<!-- --- -->

## Contents

### 1. CNN with GPU Acceleration
**Notebook:** `cnn-fashion-mnist-gpu.ipynb`

- Uses raw Fashion-MNIST CSV data
- Reshapes flat image vectors into `1×28×28` tensors
- Custom `Dataset` and `DataLoader`
- CNN architecture with:
  - Convolution layers
  - Batch Normalization
  - Max Pooling
  - Dropout for regularization
- Fully connected classifier head
- Trained using **SGD with weight decay**
- Evaluated on train and test sets

<!-- --- -->

## Dataset

This project uses the **Fashion-MNIST** dataset.

⚠️ The dataset file is **not included** in the repository due to GitHub file size limits.
- Please download: `fashion-mnist_train.csv`
- From Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist
- Place the CSV file in the same directory as the notebook before running.

<!-- --- -->

## Key Concepts Covered

- CNN fundamentals in PyTorch
- `Conv2d`, `BatchNorm2d`, `MaxPool2d`
- Image tensor reshaping for CNNs
- GPU-based training workflow
- Regularization using Dropout and L2 weight decay
- Modular CNN design (`features` + `classifier`)

<!-- --- -->

## Requirements

Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
> GPU support requires a CUDA-enabled PyTorch installation.
