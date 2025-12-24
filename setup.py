"""
Device Setup for Face Expression Recognition
"""

import torch


def setup_device():
    """
    Setup and return the appropriate device (MPS for Apple Silicon or CPU).
    
    Returns:
        torch.device: The device to use for training/inference
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

