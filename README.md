# FashionMNIST Image Classifier

A deep learning image classifier built with PyTorch to classify fashion items from the FashionMNIST dataset. The model achieves high accuracy in distinguishing between 10 different categories of clothing and accessories.

## Overview

This project implements a neural network classifier that can identify various fashion items including:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Features

- **Custom Neural Network Architecture**: Multi-layer perceptron with batch normalization and dropout
- **Data Augmentation**: Normalized grayscale image preprocessing
- **Training Optimization**: Adam optimizer with learning rate scheduling
- **Model Persistence**: Automatic saving of best-performing model weights
- **Comprehensive Evaluation**: Test accuracy metrics and mistake analysis

## Model Architecture

The classifier uses a fully connected neural network with the following structure:

- Input Layer: 784 features (flattened 28×28 images)
- Hidden Layer 1: 256 neurons + BatchNorm + ReLU + Dropout(0.05)
- Hidden Layer 2: 128 neurons + BatchNorm + ReLU + Dropout(0.05)
- Hidden Layer 3: 64 neurons + BatchNorm + ReLU + Dropout(0.05)
- Output Layer: 10 classes (softmax classification)

## Requirements

```txt
torch
torchvision
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fashionmnist
```

2. Install dependencies:
```bash
pip install torch torchvision matplotlib
```

## Usage

### Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook classifier.ipynb
```

The notebook includes:
1. Data loading and exploration
2. Model definition
3. Training loop with validation
4. Model evaluation
5. Individual image testing
6. Mistake analysis

### Using a Pre-trained Model

Load the saved model weights:

```python
import torch
from classifier import FashionClassifer

model = FashionClassifer()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

## Dataset

The project uses the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist):
- Training set: 60,000 images (split into 50,000 train / 10,000 validation)
- Test set: 10,000 images
- Image size: 28×28 grayscale
- Classes: 10 categories

The dataset is automatically downloaded when running the code.

## Training Details

- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-6)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 128
- **Epochs**: 10
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.1, patience=3)
- **Regularization**: Dropout (p=0.05) and Batch Normalization

## Results

The model achieves competitive accuracy on the FashionMNIST test set. The notebook includes visualization tools to:
- View correctly classified images
- Analyze misclassified examples
- Understand model performance across different categories

## Project Structure

```
fashionmnist/
├── classifier.ipynb       # Main Jupyter notebook with code
├── best_model.pth        # Saved model weights
├── data/                 # FashionMNIST dataset (auto-downloaded)
│   └── FashionMNIST/
│       └── raw/
├── README.md            # Project documentation
└── LICENSE              # MIT License
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FashionMNIST dataset by [Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- Built with [PyTorch](https://pytorch.org/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Created as a deep learning project for image classification.