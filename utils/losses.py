"""
Custom loss functions for semantic segmentation.

Includes:
- Focal Loss: Addresses class imbalance by focusing on hard examples
- Dice Loss: Optimizes IoU directly
- Combined Loss: Focal + Dice for best results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Original paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(pt) = -α(1-pt)^γ * log(pt)
    
    Where:
    - pt: model's predicted probability for the true class
    - α: balancing factor (typically 0.25)
    - γ: focusing parameter (typically 2.0)
    
    The (1-pt)^γ term down-weights easy examples and focuses on hard ones.
    
    Args:
        alpha (float): Balancing factor for class imbalance
        gamma (float): Focusing parameter (higher = more focus on hard examples)
        ignore_index (int): Index to ignore in loss computation
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] logits from model
            targets: [B, H, W] ground truth labels
        
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                  ignore_index=self.ignore_index)
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            # Only average over valid pixels
            valid_mask = (targets != self.ignore_index)
            if valid_mask.sum() > 0:
                return focal_loss[valid_mask].mean()
            else:
                return focal_loss.sum() * 0  # Return 0 if no valid pixels
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Directly optimizes the Dice coefficient (similar to IoU).
    
    Dice = 2*|X∩Y| / (|X| + |Y|)
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        ignore_index (int): Index to ignore in loss computation
    """
    
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] logits from model
            targets: [B, H, W] ground truth labels
        
        Returns:
            Dice loss value (1 - Dice coefficient)
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Create one-hot encoding of targets
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Mask out ignore_index
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask
        
        # Compute Dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return dice loss (1 - dice)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss for best performance.
    
    Combines the benefits of both:
    - Focal Loss: Handles class imbalance, focuses on hard examples
    - Dice Loss: Directly optimizes IoU metric
    
    Args:
        focal_weight (float): Weight for focal loss component
        dice_weight (float): Weight for dice loss component
        focal_alpha (float): Alpha parameter for focal loss
        focal_gamma (float): Gamma parameter for focal loss
        ignore_index (int): Index to ignore in loss computation
    """
    
    def __init__(self, focal_weight=1.0, dice_weight=1.0, 
                 focal_alpha=0.25, focal_gamma=2.0, ignore_index=255):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, 
                                    ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] logits from model
            targets: [B, H, W] ground truth labels
        
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice


def get_loss_function(loss_type='focal', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type (str): 'ce', 'focal', 'dice', or 'combined'
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=kwargs.get('ignore_index', 255))
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0),
            ignore_index=kwargs.get('ignore_index', 255)
        )
    elif loss_type == 'dice':
        return DiceLoss(
            smooth=kwargs.get('smooth', 1.0),
            ignore_index=kwargs.get('ignore_index', 255)
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            focal_weight=kwargs.get('focal_weight', 1.0),
            dice_weight=kwargs.get('dice_weight', 1.0),
            focal_alpha=kwargs.get('focal_alpha', 0.25),
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            ignore_index=kwargs.get('ignore_index', 255)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    """Test loss functions."""
    print("Testing Loss Functions...\n")
    
    # Create dummy data
    batch_size = 2
    num_classes = 21
    height, width = 32, 32
    
    # Dummy logits and targets
    inputs = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Add some ignore pixels
    targets[0, 10:15, 10:15] = 255
    
    # Test each loss
    print("1. Cross Entropy Loss:")
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    loss_val = ce_loss(inputs, targets)
    print(f"   Loss: {loss_val.item():.4f}\n")
    
    print("2. Focal Loss:")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=255)
    loss_val = focal_loss(inputs, targets)
    print(f"   Loss: {loss_val.item():.4f}\n")
    
    print("3. Dice Loss:")
    dice_loss = DiceLoss(ignore_index=255)
    loss_val = dice_loss(inputs, targets)
    print(f"   Loss: {loss_val.item():.4f}\n")
    
    print("4. Combined Loss:")
    combined_loss = CombinedLoss(focal_weight=1.0, dice_weight=1.0, ignore_index=255)
    loss_val = combined_loss(inputs, targets)
    print(f"   Loss: {loss_val.item():.4f}\n")
    
    print("[SUCCESS] All loss functions working correctly!")

