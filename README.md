# Face Expression Recognition

A deep learning project for recognizing facial expressions using Convolutional Neural Networks (CNN) built with PyTorch. This project implements two different approaches: a custom CNN architecture and a transfer learning model using ResNet18.

## Overview

This project implements two models to classify facial expressions into 7 categories: **angry**, **disgust**, **fear**, **happy**, **neutral**, **sad**, and **surprise**.

- **Model 0**: Custom CNN architecture trained from scratch
- **Model 1**: Transfer learning model using pretrained ResNet18

## Dataset

The model is trained on the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) from Kaggle, which contains labeled images of faces expressing different emotions.

- **Training set**: Located in `images/train/`
- **Test/Validation set**: Located in `images/validation/`
- **Image size**: 48×48 pixels
- **Number of classes**: 7

## Model Architectures

### Model 0: Custom CNN (`FaceExpressionModel`)

A custom CNN architecture trained from scratch:

### Convolutional Blocks (3 blocks)

1. **Conv Block 1**:

   - Conv2d(3 → 16 channels, kernel_size=3, padding=1)
   - BatchNorm2d(16)
   - ReLU activation
   - MaxPool2d(kernel_size=2, stride=2)

2. **Conv Block 2**:

   - Conv2d(16 → 32 channels, kernel_size=3, padding=1)
   - BatchNorm2d(32)
   - ReLU activation
   - MaxPool2d(kernel_size=2, stride=2)

3. **Conv Block 3**:
   - Conv2d(32 → 64 channels, kernel_size=3, padding=1)
   - BatchNorm2d(64)
   - ReLU activation
   - MaxPool2d(kernel_size=2, stride=2)

### Fully Connected Layers

- Flatten layer
- Linear(64 × 6 × 6 → 128)
- Dropout(0.25)
- ReLU activation
- Linear(128 → num_classes)

### Model 1: Transfer Learning (`ResNet18`)

A transfer learning approach using pretrained ResNet18:

#### Base Model

- **Architecture**: ResNet18 pretrained on ImageNet (`weights='DEFAULT'`)
- **Transfer Learning**: Uses pretrained convolutional layers as feature extractors

#### Custom Classification Head

The final fully connected layer is replaced with:

- Linear(ResNet18 features → 256)
- ReLU activation
- Dropout(0.25)
- Linear(256 → num_classes)

#### Advantages of Transfer Learning

- ✅ **Faster convergence**: Leverages pretrained features from ImageNet
- ✅ **Better performance**: Often achieves higher accuracy with fewer epochs
- ✅ **Robust features**: Pretrained layers capture general image features
- ✅ **Efficient training**: Requires fewer training epochs (typically 10 vs 5+)

## Features

- ✅ **MPS/GPU Support**: Automatically uses Apple Silicon GPU (MPS) or falls back to CPU
- ✅ **Data Augmentation**: Uses AutoAugment for training data
- ✅ **Batch Normalization**: Improves training stability and convergence
- ✅ **Dropout**: Prevents overfitting
- ✅ **Model Checkpointing**: Saves trained models and results

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
torch
torchvision
kagglehub
matplotlib
numpy
pillow
tqdm
pandas
scikit-learn
opencv-python
```

### Regenerating requirements.txt

To automatically regenerate `requirements.txt` from your conda environment:

**Option 1: Using the script**

```bash
./generate_requirements.sh
```

**Option 2: Manual command**

```bash
conda activate neuro && pip list --format=freeze | grep -E "(torch|torchvision|kagglehub|matplotlib|numpy|pillow|PIL|tqdm|pandas|scikit-learn|scikit-image|opencv)" > requirements.txt
```

**Option 3: Export all packages** (if you want everything)

```bash
conda activate neuro && pip freeze > requirements.txt
```

## Usage

1. **Open the notebook**:

   ```bash
   jupyter notebook lab.ipynb
   ```

2. **Run all cells** to:

   - Download the dataset (if not already downloaded)
   - Load and preprocess the data
   - Train the model
   - Evaluate performance
   - Save the trained model

3. **Training Parameters**:

   **Model 0 (Custom CNN)**:

   - Batch size: 32
   - Optimizer: Adam
   - Learning rate: 0.001
   - Loss function: CrossEntropyLoss (with class weights)
   - Default epochs: 5 (configurable)

   **Model 1 (Transfer Learning)**:

   - Batch size: 32
   - Optimizer: Adam
   - Learning rate: 0.0001 (smaller LR for fine-tuning)
   - Loss function: CrossEntropyLoss (with class weights)
   - Default epochs: 10

## Model Files

Trained models are saved in the `models/` directory:

- `model_0.pth`: Custom CNN model weights
- `model_0_results.json`: Model 0 training results and metrics
- `model_v1.pth`: ResNet18 transfer learning model weights
- `model_v1_results.json`: Model 1 training results and metrics

## Project Structure

```
face_expression_recognizer/
├── lab.ipynb                      # Main training notebook
├── models/                         # Saved models and results
│   ├── model_0.pth                # Custom CNN model
│   ├── model_0_results.json       # Custom CNN results
│   ├── model_v1.pth               # Transfer learning model
│   └── model_v1_results.json      # Transfer learning results
├── requirements.txt                # Python package dependencies
├── generate_requirements.sh        # Script to regenerate requirements.txt
└── README.md                       # This file
```

## Results

Both models achieve classification of 7 facial expressions. Training metrics (loss and accuracy) are saved in their respective JSON files and can be visualized using the plotting functions in the notebook.

### Model Comparison

- **Model 0**: Custom architecture, trains from scratch, good for understanding CNN fundamentals
- **Model 1**: Transfer learning approach, typically achieves better accuracy faster, leverages pretrained knowledge

You can compare the performance of both models by examining their respective results files and using the confusion matrix visualization functions in the notebook.

## Notes

- The dataset is automatically downloaded from Kaggle using `kagglehub` on first run
- Images are resized to 48×48 pixels during preprocessing
- Both models use MPS acceleration on Apple Silicon Macs for faster training
- Class weights are calculated and used for handling imbalanced datasets
- **Transfer Learning Tip**: Model 1 uses a smaller learning rate (0.0001) because the pretrained layers already have good feature representations - fine-tuning requires gentler updates
- The notebook includes functions to test both models with random images and visualize confusion matrices

## License

This project uses the Face Expression Recognition Dataset from Kaggle. Please refer to the dataset's license for usage terms.
