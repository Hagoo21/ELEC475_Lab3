"""
Metrics utilities for semantic segmentation evaluation.

Implements mean Intersection-over-Union (mIoU) and other metrics
for PASCAL VOC 2012 segmentation.
"""

import torch
import numpy as np


class SegmentationMetrics:
    """
    Calculate segmentation metrics including mIoU.
    
    Args:
        num_classes (int): Number of segmentation classes
        ignore_index (int): Index to ignore in calculation (e.g., 255 for background)
    """
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions, targets):
        """
        Update confusion matrix with new predictions.
        
        Args:
            predictions (torch.Tensor or np.ndarray): Predicted class labels [B, H, W] or [H, W]
            targets (torch.Tensor or np.ndarray): Ground truth labels [B, H, W] or [H, W]
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Remove ignored indices
        mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        # Confusion matrix: rows are ground truth, columns are predictions
        indices = self.num_classes * targets + predictions
        confusion = np.bincount(indices.astype(int), minlength=self.num_classes ** 2)
        self.confusion_matrix += confusion.reshape(self.num_classes, self.num_classes)
    
    def get_miou(self):
        """
        Calculate mean Intersection-over-Union (mIoU).
        
        Returns:
            float: mIoU score
        """
        # IoU = TP / (TP + FP + FN)
        # TP is diagonal, FP is column sum - diagonal, FN is row sum - diagonal
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +  # Ground truth counts (TP + FN)
            self.confusion_matrix.sum(axis=0) -  # Prediction counts (TP + FP)
            intersection                          # Subtract TP once (counted twice)
        )
        
        # Avoid division by zero
        iou = np.zeros(self.num_classes)
        valid = union > 0
        iou[valid] = intersection[valid] / union[valid]
        
        # Mean IoU across classes
        miou = np.mean(iou[valid]) if valid.any() else 0.0
        
        return miou
    
    def get_iou_per_class(self):
        """
        Calculate IoU for each class.
        
        Returns:
            np.ndarray: IoU scores per class
        """
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            intersection
        )
        
        iou = np.zeros(self.num_classes)
        valid = union > 0
        iou[valid] = intersection[valid] / union[valid]
        
        return iou
    
    def get_pixel_accuracy(self):
        """
        Calculate pixel accuracy.
        
        Returns:
            float: Pixel accuracy
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else 0.0
    
    def get_mean_accuracy(self):
        """
        Calculate mean accuracy (accuracy per class, then averaged).
        
        Returns:
            float: Mean accuracy
        """
        class_correct = np.diag(self.confusion_matrix)
        class_total = self.confusion_matrix.sum(axis=1)
        
        accuracy = np.zeros(self.num_classes)
        valid = class_total > 0
        accuracy[valid] = class_correct[valid] / class_total[valid]
        
        return np.mean(accuracy[valid]) if valid.any() else 0.0


def compute_miou(predictions, targets, num_classes=21, ignore_index=255):
    """
    Compute mIoU for a batch of predictions.
    
    Args:
        predictions (torch.Tensor): Predicted logits [B, C, H, W] or class labels [B, H, W]
        targets (torch.Tensor): Ground truth labels [B, H, W]
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in calculation
    
    Returns:
        float: mIoU score
    """
    # Convert logits to class predictions if needed
    if predictions.dim() == 4:  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]
    
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    metrics.update(predictions, targets)
    
    return metrics.get_miou()


def compute_metrics_batch(predictions, targets, num_classes=21, ignore_index=255):
    """
    Compute multiple metrics for a batch.
    
    Args:
        predictions (torch.Tensor): Predicted logits [B, C, H, W] or class labels [B, H, W]
        targets (torch.Tensor): Ground truth labels [B, H, W]
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in calculation
    
    Returns:
        dict: Dictionary containing multiple metrics
    """
    # Convert logits to class predictions if needed
    if predictions.dim() == 4:  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1)  # [B, H, W]
    
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    metrics.update(predictions, targets)
    
    return {
        'miou': metrics.get_miou(),
        'pixel_acc': metrics.get_pixel_accuracy(),
        'mean_acc': metrics.get_mean_accuracy(),
        'iou_per_class': metrics.get_iou_per_class()
    }


# PASCAL VOC 2012 class names
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]


def print_class_iou(iou_per_class, class_names=VOC_CLASSES):
    """
    Print IoU for each class.
    
    Args:
        iou_per_class (np.ndarray): IoU scores per class
        class_names (list): List of class names
    """
    print("\nPer-class IoU:")
    print("-" * 40)
    for i, (name, iou) in enumerate(zip(class_names, iou_per_class)):
        print(f"  {i:2d}. {name:15s}: {iou:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    """Test metrics calculation."""
    print("Testing Segmentation Metrics...\n")
    
    # Create dummy predictions and targets
    batch_size = 2
    num_classes = 21
    height, width = 128, 128
    
    # Simulate predictions (logits)
    predictions = torch.randn(batch_size, num_classes, height, width)
    
    # Simulate targets
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Add some ignore indices
    targets[0, :10, :10] = 255
    
    # Compute metrics
    print(f"Input shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets:     {targets.shape}")
    print()
    
    # Test mIoU calculation
    miou = compute_miou(predictions, targets, num_classes=num_classes)
    print(f"mIoU: {miou:.4f}")
    print()
    
    # Test full metrics
    metrics = compute_metrics_batch(predictions, targets, num_classes=num_classes)
    print(f"Full metrics:")
    print(f"  mIoU:              {metrics['miou']:.4f}")
    print(f"  Pixel Accuracy:    {metrics['pixel_acc']:.4f}")
    print(f"  Mean Accuracy:     {metrics['mean_acc']:.4f}")
    
    # Print per-class IoU (first few classes)
    print(f"\nPer-class IoU (first 5 classes):")
    for i, (name, iou) in enumerate(zip(VOC_CLASSES[:5], metrics['iou_per_class'][:5])):
        print(f"  {name:15s}: {iou:.4f}")
    
    print("\n[SUCCESS] Metrics calculation tested successfully!")

