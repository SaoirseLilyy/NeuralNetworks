# CNN Hyper-parameter Tuning for CIFAR-10 Classification

## Overview

This Google Colab notebook explores the design and optimization of a Convolutional Neural Network (CNN) for multi-class classification using the CIFAR-10 dataset. The experiment focuses on tuning various hyper-parameters, including convolutional channels, kernel sizes, learning rates, batch sizes, and regularization techniques to enhance model performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Base Model Architecture](#base-model-architecture)
3. [Hyper-parameter Tuning](#hyper-parameter-tuning)
   - [Network Setup](#network-setup)
   - [Learning Rate](#learning-rate)
   - [Batch Size](#batch-size)
   - [Regularization Procedures](#regularization-procedures)
      - [L2 Regularization](#l2-regularization)
      - [Batch Normalization](#batch-normalization)
   - [Mixup](#mixup)
4. [Results & Conclusion](#results--conclusion)
5. [Running the Notebook](#running-the-notebook)
6. [References](#references)
7. [Setup](#setup)

## Introduction

This notebook documents the design and fine-tuning of a CNN for classifying images from the CIFAR-10 dataset. The objective is to investigate the impact of various hyper-parameters on the model's performance and develop a robust CNN through iterative experimentation.

## Base Model Architecture

The base CNN architecture includes:
- **Convolutional Layers**: Two convolutional layers with ReLU activations and max-pooling.
- **Final Layers**: Flattened output passed through linear layers with cross-entropy loss for classification.

**Details:**
- Convolutional layers use "same" padding and a stride of 1.
- ReLU activations introduce non-linearity.
- Data is normalized to stabilize training.

## Hyper-parameter Tuning

### Network Setup
- **Channels and Kernels Tested**:
  - **Layer 1 (C1)**: Channels = [32, 64]; Kernels = [3, 5]
  - **Layer 2 (C2)**: Channels = [64, 128]; Kernels = [5, 7]
- **Optimal Configuration**: C1: 32 channels, 3x3 kernel; C2: 128 channels, 5x5 kernel.

### Learning Rate
- **Rates Tested**: 0.01, 0.001, 0.0001
- **Best Performance**: A learning rate of 0.0001 provided the most stability and accuracy.

### Batch Size
- **Sizes Tested**: 20, 40, 80
- **Optimal Size**: A batch size of 20 achieved the highest accuracy and lowest loss.

### Regularization Procedures

#### L2 Regularization
- **λ Values Tested**: 0.01, 0.001, 0.0001, 0.00001
- **Best Performance**: A regularization parameter of 0.00001 yielded the best results in terms of accuracy and loss.

#### Batch Normalization
- **Effect**: Improved model performance, particularly with larger batch sizes, by stabilizing training and reducing loss.

### Mixup
- **Approach**: Data augmentation by blending pairs of images.
- **Results**: Mixup led to decreased performance, suggesting it may require pre-training for effective results.

## Results & Conclusion

- **Best Configuration**: The optimal model configuration achieved a maximum accuracy of 72.5%.
- **Key Findings**:
  - Larger channels in deeper layers and increasing kernel sizes improve performance.
  - Lower learning rates prevent issues like the 'dying ReLU'.
  - Smaller batch sizes generally perform better with smaller learning rates.
  - L2 regularization with small λ values helps in avoiding overfitting.
  - Batch normalization improves stability, especially with larger batches.
  - Mixup did not enhance performance and warrants further investigation with pre-trained models.

## Running the Notebook

1. Open the notebook in Google Colab [here](https://github.com/username/repository/blob/main/filename.ipynb).
2. Ensure GPU acceleration is enabled (Runtime > Change runtime type > GPU).
3. Execute all cells sequentially to perform the experiments and view results.

## Setup

To run the notebook, you need to install and import the required libraries. Here is the setup code used in this notebook:

```python
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Normalize
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import notebook
from collections import Counter
import os
import time
from google.colab import drive
drive.mount('/content/gdrive')



