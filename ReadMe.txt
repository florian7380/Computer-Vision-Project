ResNet50 CIFAR-10 Image Classification

This project implements transfer learning using a pre-trained ResNet50 model to classify images from the CIFAR-10 dataset.
Overview
Each of the notebooks demonstrates a two-stage training approach:

1. Initial training with frozen ResNet50 base layers
2. Fine-tuning with unfrozen base layers for improved performance

the difference is in the preprocessing steps as well as the number of dropout layers used. 
Dataset
CIFAR-10 consists of 60,000 32x32 color images in 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is split into 50,000 training images and 10,000 test images, with 5,000 images per class in the training set.
Model Architecture
The model uses ResNet50 pre-trained on ImageNet as a feature extractor, with custom classification layers on top:
ResNet50 (frozen initially)
    ↓
GlobalAveragePooling2D
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.25)
    ↓
Dense (64 units, ReLU)
    ↓
(Dropout (0.25) second Dropout not in every Notebook)
    ↓
Dense (10 units, Softmax)

Data Processing

Preprocessing

- Normalization: pixel values scaled from [0, 255] to [0, 1]
- Image size: 32x32 (native CIFAR-10 resolution) or enlarged to 224x224 (ResNet50 got trained for pictures this size)

Data Augmentation
The training pipeline includes:

- Random horizontal flips
- Random 90-degree rotations
- Random brightness adjustments (±20%)
- Random contrast adjustments (0.8x to 1.2x)

Pipeline Optimization

- Shuffling of training data
- Parallel preprocessing with tf.data.AUTOTUNE
- Batch size: 64
- Prefetching for improved performance

Training Strategy
Stage 1: Transfer Learning (Frozen Base)

- Epochs: 20 (with early stopping)
- Learning rate: 0.001
- Optimizer: Adam
- Base model: Frozen (not trainable)
- Early stopping: Monitors validation accuracy with patience of 5 epochs

Stage 2: Fine-Tuning (Unfrozen Base)

- Epochs: 20 (with early stopping)
- Learning rate: 0.0001 (reduced by 10x)
- Optimizer: Adam
- Base model: Unfrozen (all layers trainable)
- Early stopping: Monitors validation accuracy with patience of 10 epochs

Requirements:

pythonnumpy
matplotlib
pandas
tensorflow

Install dependencies:
bash
pip install numpy matplotlib pandas tensorflow

Usage

1. Run the notebook cells sequentially
2. The script will automatically:

- Download CIFAR-10 dataset
- Visualize sample images and class distribution
- Train the model in two stages
- Generate accuracy and loss plots
- Save the final model as resnet50_model.h5



Output
The notebook produces:

- Visualization of sample training images
- Bar chart of class distribution
- Training/validation accuracy and loss plots for both training stages
- Final test accuracy and loss metrics
- Saved model file: resnet50_model.h5

Expected Performance
Initial training typically shows:

- Slow start (epochs 1-4): validation accuracy around 10%
- Breakthrough (epochs 5-6): rapid improvement to 40-60%
- Convergence: validation accuracy reaching 60-70%

Fine-tuning can further improve performance by several percentage points.

Notes

- The model uses sparse_categorical_crossentropy loss, which accepts integer labels directly
- Early stopping helps prevent overfitting by restoring the best weights
- The lower learning rate during fine-tuning prevents forgetting of pre-trained features
- Data augmentation helps improve generalization and reduce overfitting of the complex ResNet50 structure on the simple dataset CIFAR-10