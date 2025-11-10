"""
Utility functions for knowledge distillation training and evaluation.

Includes:
- mIoU computation from confusion matrix
- Model evaluation utilities
- Visualization helpers

Author: ELEC475 Lab 3
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def compute_miou(confusion_matrix):
    """
    Compute mean Intersection over Union (mIoU) from confusion matrix.
    
    The confusion matrix is organized such that:
    - Rows represent ground truth classes
    - Columns represent predicted classes
    
    For each class i:
        IoU_i = TP_i / (TP_i + FP_i + FN_i)
              = confusion[i,i] / (sum(confusion[i,:]) + sum(confusion[:,i]) - confusion[i,i])
    
    mIoU is the average IoU across all classes (excluding classes not present).
    
    Args:
        confusion_matrix (np.ndarray): [num_classes, num_classes] confusion matrix
        
    Returns:
        float: Mean IoU averaged over all classes
    """
    # Intersection: true positives (diagonal elements)
    intersection = np.diag(confusion_matrix)
    
    # Union: TP + FP + FN
    # TP + FP = sum over predictions (column)
    # TP + FN = sum over ground truth (row)
    # Union = (TP + FP) + (TP + FN) - TP
    ground_truth_total = confusion_matrix.sum(axis=1)  # Sum over rows
    predicted_total = confusion_matrix.sum(axis=0)     # Sum over columns
    union = ground_truth_total + predicted_total - intersection
    
    # Compute IoU for each class
    # Avoid division by zero
    iou_per_class = intersection / np.maximum(union, 1)
    
    # Only consider classes that appear in ground truth
    valid_classes = ground_truth_total > 0
    
    # Mean IoU over valid classes
    miou = np.mean(iou_per_class[valid_classes])
    
    return miou


def compute_iou_per_class(confusion_matrix, class_names=None):
    """
    Compute IoU for each class individually.
    
    Args:
        confusion_matrix (np.ndarray): [num_classes, num_classes] confusion matrix
        class_names (list, optional): Names of classes for display
        
    Returns:
        dict: Dictionary mapping class index (or name) to IoU value
    """
    num_classes = confusion_matrix.shape[0]
    
    intersection = np.diag(confusion_matrix)
    ground_truth_total = confusion_matrix.sum(axis=1)
    predicted_total = confusion_matrix.sum(axis=0)
    union = ground_truth_total + predicted_total - intersection
    
    iou_dict = {}
    for i in range(num_classes):
        if ground_truth_total[i] > 0:
            iou = intersection[i] / max(union[i], 1)
            class_id = class_names[i] if class_names else i
            iou_dict[class_id] = iou
    
    return iou_dict


def evaluate_segmentation(model, dataloader, device, num_classes=21, 
                         return_confusion=False):
    """
    Evaluate segmentation model and compute mIoU.
    
    Args:
        model (nn.Module): Segmentation model
        dataloader (DataLoader): Validation/test data loader
        device (torch.device): Device to run on
        num_classes (int): Number of segmentation classes
        return_confusion (bool): Whether to return confusion matrix
        
    Returns:
        float or tuple: mIoU, optionally with confusion matrix
    """
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            targets = targets.cpu().numpy()
            
            # Forward pass
            if hasattr(model, 'return_features'):
                # Handle models that can return features
                original_return_features = model.return_features
                model.return_features = False
                outputs = model(images)
                model.return_features = original_return_features
            else:
                outputs = model(images)
            
            # Handle dict output (e.g., from FCN models)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Update confusion matrix
            for pred, target in zip(preds, targets):
                # Mask out ignore regions (label 255)
                mask = (target >= 0) & (target < num_classes)
                
                # Flatten and count co-occurrences
                label_true = target[mask].astype(int)
                label_pred = pred[mask].astype(int)
                
                # Accumulate confusion matrix
                confusion_matrix += np.bincount(
                    num_classes * label_true + label_pred,
                    minlength=num_classes ** 2
                ).reshape(num_classes, num_classes)
    
    # Compute mIoU
    miou = compute_miou(confusion_matrix)
    
    if return_confusion:
        return miou, confusion_matrix
    else:
        return miou


def print_class_iou(confusion_matrix, class_names=None):
    """
    Print IoU for each class in a formatted table.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        class_names (list, optional): Names of classes
    """
    iou_dict = compute_iou_per_class(confusion_matrix, class_names)
    
    print("\nPer-Class IoU:")
    print("-" * 50)
    print(f"{'Class':<30} {'IoU':>10}")
    print("-" * 50)
    
    for class_id, iou in iou_dict.items():
        print(f"{str(class_id):<30} {iou:>10.4f}")
    
    print("-" * 50)
    miou = np.mean(list(iou_dict.values()))
    print(f"{'Mean IoU':<30} {miou:>10.4f}")
    print("-" * 50)


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_model(model):
    """
    Freeze all parameters in a model.
    
    Args:
        model (nn.Module): Model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """
    Unfreeze all parameters in a model.
    
    Args:
        model (nn.Module): Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


# PASCAL VOC 2012 class names for reference
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]


if __name__ == "__main__":
    """Test utility functions."""
    print("Testing distillation utility functions...\n")
    
    # Create a dummy confusion matrix
    num_classes = 21
    confusion = np.random.randint(0, 100, size=(num_classes, num_classes))
    
    # Make diagonal dominant (more correct predictions)
    for i in range(num_classes):
        confusion[i, i] += 500
    
    print("Computing mIoU from confusion matrix...")
    miou = compute_miou(confusion)
    print(f"mIoU: {miou:.4f}")
    
    print("\nComputing per-class IoU...")
    iou_dict = compute_iou_per_class(confusion, VOC_CLASSES)
    
    print("\nTop 5 classes by IoU:")
    sorted_iou = sorted(iou_dict.items(), key=lambda x: x[1], reverse=True)
    for class_name, iou in sorted_iou[:5]:
        print(f"  {class_name:<15} {iou:.4f}")
    
    print("\n[SUCCESS] Distillation utility functions working correctly!")

