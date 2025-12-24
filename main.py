#!/usr/bin/env python3
"""
Face Expression Recognition - Command Line Interface
"""

import argparse
import os
import json
import random
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib
# Try to use an interactive backend
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        pass  # Use default backend
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode so plots don't block
from PIL import Image

import kagglehub

from setup import setup_device
from NNmodel import FaceExpressionModel, create_transfer_model
from helper import (
    train, plot_loss_curves, plot_confusion_matrix, 
    test_model_with_random_image
)


def setup_dataset():
    """Download and setup dataset."""
    dataset_dir = f"/Users/{os.getlogin()}/.cache/kagglehub/datasets/jonathanoheix/face-expression-recognition-dataset/versions/1"
    print(f"Dataset directory: {dataset_dir} -> checking if it exists")
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory: {dataset_dir} -> does not exist, downloading dataset")
        dataset_download_path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
    else:
        dataset_download_path = dataset_dir
    
    print("Path to dataset files:", dataset_download_path)
    return dataset_download_path


def load_datasets(dataset_path, batch_size=32):
    """Load training and test datasets."""
    image_folder = Path(dataset_path) / "images"
    train_path = image_folder / "train"
    test_path = image_folder / "validation"
    
    train_transform = transforms.Compose([
        transforms.AutoAugment(),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = ImageFolder(train_path, transform=train_transform)
    test_dataset = ImageFolder(test_path, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, test_dataset, train_loader, test_loader


def calculate_class_weights(dataset, device):
    """Calculate class weights for imbalanced dataset."""
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset:
        class_counts[label] += 1
    
    print(f"Class counts: {class_counts}")
    
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    
    class_weights = []
    for count in class_counts:
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights_tensor = torch.sqrt(class_weights_tensor)
    
    print(f"Calculated weights: {class_weights_tensor}")
    return class_weights_tensor


def train_model_v0(args, train_dataset, train_loader, test_loader, device, class_weights):
    """Train or load Model 0 (Custom CNN)."""
    model_dir = Path("./models")
    model_path = model_dir / "model_0.pth"
    results_path = model_dir / "model_0_results.json"
    
    model = FaceExpressionModel(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    lr = args.learning_rate if args.learning_rate else 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Check if model exists and --train flag
    if args.train and model_path.exists():
        print("Retraining Model 0 (existing model will be overwritten)...")
        model_results = train(
            model=model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            device=device
        )
        
        torch.save(model, model_path)
        with open(results_path, "w") as f:
            json.dump(model_results, f)
        
        if args.plot:
            plot_loss_curves(model_results)
    
    elif args.train:
        print("Training Model 0...")
        model_results = train(
            model=model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            device=device
        )
        
        torch.save(model, model_path)
        with open(results_path, "w") as f:
            json.dump(model_results, f)
        
        if args.plot:
            plot_loss_curves(model_results)
    
    elif model_path.exists():
        print("Loading saved Model 0...")
        model = torch.load(model_path, weights_only=False)
        if results_path.exists():
            with open(results_path, "r") as f:
                model_results = json.load(f)
            if args.plot:
                plot_loss_curves(model_results)
    else:
        print("No saved model found. Use --train to train a new model.")
        return None
    
    return model


def train_model_v1(args, train_dataset, train_loader, test_loader, device, class_weights):
    """Train or load Model 1 (Transfer Learning)."""
    model_dir = Path("./models")
    model_path = model_dir / "model_v1.pth"
    results_path = model_dir / "model_v1_results.json"
    
    model = create_transfer_model(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    lr = args.learning_rate if args.learning_rate else 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Check if model exists and --train flag
    if args.train and model_path.exists():
        print("Retraining Model 1 (existing model will be overwritten)...")
        model_results = train(
            model=model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            device=device
        )
        
        torch.save(model, model_path)
        with open(results_path, "w") as f:
            json.dump(model_results, f)
        
        if args.plot:
            plot_loss_curves(model_results)
    
    elif args.train:
        print("Training Model 1...")
        model_results = train(
            model=model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            device=device
        )
        
        torch.save(model, model_path)
        with open(results_path, "w") as f:
            json.dump(model_results, f)
        
        if args.plot:
            plot_loss_curves(model_results)
    
    elif model_path.exists():
        print("Loading saved Model 1...")
        model = torch.load(model_path, weights_only=False)
        if results_path.exists():
            with open(results_path, "r") as f:
                model_results = json.load(f)
            if args.plot:
                plot_loss_curves(model_results)
    else:
        print("No saved model found. Use --train to train a new model.")
        return None
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Face Expression Recognition - Train and Evaluate Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model v0 for 20 epochs
  python main.py --model v0 --train --epochs 20
  
  # Load and evaluate saved model v1
  python main.py --model v1 --eval --plot
  
  # Retrain model v0 with custom learning rate
  python main.py --model v0 --train --epochs 10 --lr 0.0005
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["v0", "v1"],
        required=True,
        help="Model to use: v0 (Custom CNN) or v1 (Transfer Learning)"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model (will retrain if model exists)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        dest="learning_rate",
        help="Learning rate (default: 0.001 for v0, 0.0001 for v1)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate model on test set"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot training curves and confusion matrix"
    )
    
    parser.add_argument(
        "--test-image",
        action="store_true",
        help="Test model with a random image"
    )
    
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Show confusion matrix"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("=" * 60)
    print("Face Expression Recognition")
    print("=" * 60)
    
    device = setup_device()
    dataset_path = setup_dataset()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, test_dataset, train_loader, test_loader = load_datasets(
        dataset_path, batch_size=args.batch_size
    )
    print(f"Classes: {train_dataset.classes}")
    
    # Calculate class weights
    print("\nCalculating class weights...")
    class_weights = calculate_class_weights(train_dataset, device)
    
    # Model selection and training/loading
    print(f"\n{'='*60}")
    print(f"Model: {args.model.upper()}")
    print(f"{'='*60}")
    
    if args.model == "v0":
        model = train_model_v0(args, train_dataset, train_loader, test_loader, device, class_weights)
    elif args.model == "v1":
        model = train_model_v1(args, train_dataset, train_loader, test_loader, device, class_weights)
    
    if model is None:
        return
    
    # Evaluation and visualization
    if args.eval or args.confusion_matrix or args.plot:
        print("\nEvaluating model...")
        if args.confusion_matrix or args.plot:
            plot_confusion_matrix(model, test_loader, train_dataset.classes, device=device)
    
    if args.test_image:
        print("\nTesting with random image...")
        test_model_with_random_image(test_dataset, model, train_dataset.classes, device=device)
    
    # Keep all plots open until user closes them
    if args.plot or args.confusion_matrix or args.test_image:
        print("\nAll plots displayed. Press Enter to exit (plots will remain open).")
        input()  # Wait for user input
    
    print("\nDone!")


if __name__ == "__main__":
    main()
