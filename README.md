# Convolutional Neural Network for CIFAR-10 Classification

## Overview

This repository contains a Google Colab notebook detailing the design and optimization of a Convolutional Neural Network (CNN) for multi-class classification on the CIFAR-10 dataset. The project investigates the impact of various hyper-parameters on model performance and refines the model to achieve optimal results.

## Project Structure

- `notebook.ipynb`: The Jupyter notebook containing the code and experiments for CNN design and hyper-parameter tuning.
- `cifar10_data.csv`: The CSV file containing the dataset used for training and evaluation.

## Dependencies

Ensure you have the following libraries installed:

```python
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import Normalize
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import notebook
from collections import Counter
import os
import time
from google.colab import drive

Setup
1. Clone the Repository
To get started, clone the repository to your local machine:

bash
Copy code
git clone https://github.com/your-username/your-repository.git
cd your-repository
2. Local Environment Setup
Ensure you have the required Python libraries installed. You can use pip to install the dependencies listed in the requirements.txt file (create one if needed):

bash
Copy code
pip install numpy pandas seaborn scikit-learn matplotlib torch torchvision tqdm
3. Google Colab Setup
If you are using Google Colab, follow these steps to set up your environment:

a. Mount Google Drive
Mount your Google Drive to access the dataset:

```python
from google.colab import drive
drive.mount('/content/gdrive')
b. Load the Dataset
Ensure that cifar10_data.csv is located in your Google Drive. Update the path as needed:

```python
import pandas as pd

# Load the CSV file from Google Drive
data = pd.read_csv('/content/gdrive/My Drive/path-to-your-file/cifar10_data.csv')
4. Run the Notebook
Open the notebook in Google Colab or a local Jupyter Notebook environment. Follow the steps in the notebook to:

Explore the Data: Understand the CIFAR-10 dataset and its characteristics.
Build the Base Model: Define a CNN architecture with convolutional layers, activation functions, and pooling.
Tune Hyper-parameters: Experiment with different hyper-parameters such as convolutional channels, kernel sizes, learning rates, and batch sizes.
Evaluate Performance: Use metrics like accuracy, loss, and macro F1 score to assess the model's performance.
Apply Regularization and Data Augmentation: Implement L2 regularization, batch normalization, and Mixup data augmentation to improve the model.
Results
The notebook includes detailed experimentation with the following hyper-parameters:

Convolutional Channels and Kernel Sizes: Impact on model performance.
Learning Rate: Effects on convergence and model stability.
Batch Size: Influence on training efficiency and accuracy.
Regularization Techniques: Use of L2 regularization and batch normalization.
Data Augmentation: Exploration of Mixup for improving generalization.
The results demonstrate the impact of these parameters on the CNN's performance, including accuracy, loss, and F1 score.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
CIFAR-10 Dataset
Google Colab
PyTorch
Contact
Feel free to reach out if you have any questions or need further assistance:

Author: Saoirse Lily Batheram Webb
Email: s.liilyy@hotmail.com



