"""
Helper Functions for Face Expression Recognition Training and Evaluation
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
from PIL import Image


def accuracy_fn(y_pred, y_true):
    """Calculate accuracy percentage."""
    correct = torch.eq(y_pred, y_true).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """
    Perform one training step.
    
    Args:
        model: The neural network model
        dataloader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to run on (MPS/CPU/CUDA)
        
    Returns:
        tuple: (average_train_loss, average_train_accuracy)
    """
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    """
    Perform one test/evaluation step.
    
    Args:
        model: The neural network model
        dataloader: Test data loader
        loss_fn: Loss function
        device: Device to run on (MPS/CPU/CUDA)
        
    Returns:
        tuple: (average_test_loss, average_test_accuracy)
    """
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device: torch.device = None):
    """
    Train a model for multiple epochs.
    
    Args:
        model: The neural network model
        train_dataloader: Training data loader
        test_dataloader: Test data loader
        optimizer: Optimizer
        loss_fn: Loss function
        epochs: Number of training epochs
        device: Device to run on (MPS/CPU/CUDA)
        
    Returns:
        dict: Dictionary containing training history with keys:
            - train_loss: List of training losses per epoch
            - train_acc: List of training accuracies per epoch
            - test_loss: List of test losses per epoch
            - test_acc: List of test accuracies per epoch
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
        
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # Return the filled results at the end of the epochs
    return results


def plot_loss_curves(results: Dict[str, List[float]]):
    """
    Plot training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # Brief pause to allow plot to render


def plot_confusion_matrix(model, dataloader, class_names, device: torch.device = None):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        model: The neural network model
        dataloader: Data loader for evaluation
        class_names: List of class names
        device: Device to run on (MPS/CPU/CUDA)
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Move model to device
    model.to(device)
    model.eval()

    y_preds = []
    y_trues = []
    
    with torch.inference_mode():
        for X, y in dataloader:
            # Send data to device
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            logits = model(X)
            pred_labels = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            # Move back to CPU before converting to numpy
            y_preds.extend(pred_labels.cpu().numpy())
            y_trues.extend(y.cpu().numpy())
            
    # Create confusion matrix
    cm = confusion_matrix(y_trues, y_preds)
    
    # Plot nicely
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # Brief pause to allow plot to render


def test_model_with_random_image(dataset, model, class_names, device: torch.device = None):
    """
    Test model with a random image from the dataset.
    
    Args:
        dataset: Dataset to sample from
        model: The neural network model
        class_names: List of class names
        device: Device to run on (MPS/CPU/CUDA)
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Pick a random image from the dataset
    test_img_idx = random.randint(0, len(dataset) - 1)
    img, real_label = dataset[test_img_idx]

    # Prepare the image for the model
    model_input = img.unsqueeze(0).to(device)

    # Get the prediction
    model.eval()
    with torch.inference_mode():
        logits = model(model_input)
        predicted_label_idx = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()

    # Show image and prediction info
    plt.figure()
    plt.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray' if img.shape[0] == 1 else None)
    plt.axis('off')
    plt.title(f"Predicted: {class_names[predicted_label_idx]}\nReal: {class_names[real_label]}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # Brief pause to allow plot to render

