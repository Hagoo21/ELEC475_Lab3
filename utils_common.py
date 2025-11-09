"""
Common utilities and constants for PASCAL VOC 2012 segmentation.

This module consolidates shared code to avoid duplication across the project.
"""

import os
import torch
import numpy as np
from PIL import Image


# =============================================================================
# VOC Dataset Constants
# =============================================================================

# PASCAL VOC 2012 class names (21 classes including background)
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

# VOC color palette for visualization (RGB values)
VOC_COLORMAP = [
    [0, 0, 0],          # 0: background
    [128, 0, 0],        # 1: aeroplane
    [0, 128, 0],        # 2: bicycle
    [128, 128, 0],      # 3: bird
    [0, 0, 128],        # 4: boat
    [128, 0, 128],      # 5: bottle
    [0, 128, 128],      # 6: bus
    [128, 128, 128],    # 7: car
    [64, 0, 0],         # 8: cat
    [192, 0, 0],        # 9: chair
    [64, 128, 0],       # 10: cow
    [192, 128, 0],      # 11: diningtable
    [64, 0, 128],       # 12: dog
    [192, 0, 128],      # 13: horse
    [64, 128, 128],     # 14: motorbike
    [192, 128, 128],    # 15: person
    [0, 64, 0],         # 16: pottedplant
    [128, 64, 0],       # 17: sheep
    [0, 192, 0],        # 18: sofa
    [128, 192, 0],      # 19: train
    [0, 64, 128]        # 20: tvmonitor
]


# =============================================================================
# Device Setup
# =============================================================================

def get_device(device_name='cuda', verbose=True):
    """
    Get PyTorch device with optional GPU information.
    
    Args:
        device_name (str): Preferred device ('cuda' or 'cpu')
        verbose (bool): Whether to print device information
    
    Returns:
        torch.device: Selected device
    """
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


# =============================================================================
# Checkpoint Management
# =============================================================================

def load_checkpoint(checkpoint_path, model=None, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint with optional optimizer and scheduler states.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (nn.Module, optional): Model to load state into
        optimizer (Optimizer, optional): Optimizer to load state into
        scheduler (Scheduler, optional): Scheduler to load state into
        device (str or torch.device): Device to map checkpoint to
    
    Returns:
        dict: Checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'miou' in checkpoint:
        print(f"  mIoU:  {checkpoint['miou']:.4f}")
    
    return checkpoint


def load_training_history(checkpoint_dir='checkpoints'):
    """
    Load training history from checkpoint directory.
    
    Tries to load from checkpoint_latest.pth first (has full history),
    then falls back to training_history.pth.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
    
    Returns:
        dict: Training history with keys 'train_loss', 'val_loss', 'val_miou', 'val_pixel_acc'
        None if no history found
    """
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    history_path = os.path.join(checkpoint_dir, 'training_history.pth')
    
    # Try loading from latest checkpoint first
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False, map_location=torch.device('cpu'))
        if 'history' in checkpoint:
            return checkpoint['history']
    
    # Fall back to standalone history file
    if os.path.exists(history_path):
        history = torch.load(history_path, weights_only=False, map_location=torch.device('cpu'))
        return history
    
    return None


def find_best_epoch(history):
    """
    Find the epoch with the best validation mIoU.
    
    Args:
        history (dict): Training history dictionary
    
    Returns:
        tuple: (best_epoch, best_miou)
    """
    if history is None or 'val_miou' not in history:
        return None, None
    
    val_miou = history['val_miou']
    best_epoch = val_miou.index(max(val_miou)) + 1
    best_miou = max(val_miou)
    
    return best_epoch, best_miou


# =============================================================================
# Visualization Utilities
# =============================================================================

def colorize_mask(mask):
    """
    Convert segmentation mask with class indices to RGB colors.
    
    Args:
        mask (np.ndarray or torch.Tensor): Mask with class indices [H, W]
    
    Returns:
        np.ndarray: RGB image [H, W, 3] with dtype uint8
    """
    # Convert to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(np.int64)
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(VOC_COLORMAP):
        colored_mask[mask == class_id] = color
    
    return colored_mask


def get_voc_colormap_tensor():
    """
    Get VOC colormap as a PyTorch tensor for efficient indexing.
    
    Returns:
        torch.Tensor: Colormap tensor of shape [256, 3]
    """
    colormap = torch.zeros(256, 3, dtype=torch.uint8)
    
    for i, color in enumerate(VOC_COLORMAP):
        if i < len(VOC_COLORMAP):
            colormap[i] = torch.tensor(color, dtype=torch.uint8)
    
    return colormap


def print_class_iou(iou_per_class, class_names=None):
    """
    Print IoU for each class in a formatted table.
    
    Args:
        iou_per_class (list or np.ndarray): IoU scores per class
        class_names (list, optional): List of class names. Defaults to VOC_CLASSES.
    """
    if class_names is None:
        class_names = VOC_CLASSES
    
    print("\nPer-class IoU:")
    print("-" * 60)
    for i, (name, iou) in enumerate(zip(class_names, iou_per_class)):
        if isinstance(iou, float) and not np.isnan(iou):
            print(f"  {i:2d}. {name:15s}: {iou:.4f} ({iou*100:.2f}%)")
        else:
            print(f"  {i:2d}. {name:15s}: N/A")
    print("-" * 60)


# =============================================================================
# File/Path Utilities
# =============================================================================

def read_split_file(split_file_path):
    """
    Read image IDs from a VOC split file.
    
    Args:
        split_file_path (str): Path to split file (e.g., train.txt)
    
    Returns:
        list: List of image IDs
    """
    if not os.path.exists(split_file_path):
        return []
    
    with open(split_file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == "__main__":
    """Test common utilities."""
    print("Testing Common Utilities...\n")
    
    # Test VOC constants
    print(f"Number of VOC classes: {len(VOC_CLASSES)}")
    print(f"Number of colors in colormap: {len(VOC_COLORMAP)}")
    print(f"Classes: {', '.join(VOC_CLASSES[:5])}...")
    
    # Test device
    print("\nTesting device setup:")
    device = get_device('cuda', verbose=True)
    
    # Test colormap
    print("\nTesting colormap:")
    dummy_mask = np.random.randint(0, 21, (100, 100))
    colored = colorize_mask(dummy_mask)
    print(f"  Input mask shape: {dummy_mask.shape}")
    print(f"  Colored mask shape: {colored.shape}")
    print(f"  Colored mask dtype: {colored.dtype}")
    
    # Test colormap tensor
    colormap_tensor = get_voc_colormap_tensor()
    print(f"  Colormap tensor shape: {colormap_tensor.shape}")
    
    # Test print_class_iou
    dummy_ious = np.random.rand(21)
    print_class_iou(dummy_ious)
    
    print("\n[SUCCESS] All utilities tested!")

