"""
Knowledge Distillation Training Pipeline for Lightweight Segmentation
========================================================================

This script implements a comprehensive knowledge distillation approach combining:
1. Response-based distillation: Distill final output probabilities using KL divergence
2. Feature-based distillation: Match intermediate feature representations using cosine similarity

Teacher: FCN-ResNet50 (pretrained on PASCAL VOC, frozen)
Student: LightweightSegmentationModel (MobileNetV3-Small backbone)

Key Components:
- α (alpha): Weight for cross-entropy loss with ground truth
- β (beta): Weight for KL divergence loss with teacher
- T (temperature): Softening parameter for probability distributions
- Feature alignment: Cosine similarity between student and teacher features

Mathematical Formulation:
-------------------------
L_total = α * L_CE + β * L_KD + γ * L_feat

Where:
  L_CE = CrossEntropy(student_logits, ground_truth)
    - Standard supervised loss, gradients flow to student
  
  L_KD = KLDiv(softmax(student/T), softmax(teacher/T)) * T²
    - Response-based distillation, gradients flow to student only (teacher frozen)
    - T² factor compensates for temperature scaling gradient magnitude
  
  L_feat = 1 - cosine_similarity(student_features, teacher_features)
    - Feature-based distillation across low/mid/high level features
    - Gradients flow to student only, teacher is frozen

Gradient Flow:
--------------
Student: ∂L_total/∂θ_student ≠ 0 (trainable)
Teacher: ∂L_total/∂θ_teacher = 0 (frozen, no gradients)

Author: ELEC475 Lab 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import os
import sys
import time
from tqdm import tqdm
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lightweight_segmentation import LightweightSegmentationModel
from utils.dataset import VOCSegmentationWithJointTransform
from utils.distillation_utils import compute_miou
import config


class KnowledgeDistillationLoss(nn.Module):
    """
    Combined Knowledge Distillation Loss with Response-based and Feature-based components.
    
    Loss Components:
    ----------------
    1. Cross-Entropy Loss: Standard supervised learning with ground truth labels
       L_CE = CE(student_logits, y)
       
    2. KL Divergence Loss: Response-based distillation from teacher's soft predictions
       L_KD = KL(softmax(student/T) || softmax(teacher/T)) * T²
       - Temperature T softens probability distributions
       - T² compensates for gradient magnitude reduction
       
    3. Cosine Similarity Loss: Feature-based distillation matching representations
       L_feat = mean(1 - cosine_sim(student_feat, teacher_feat))
       - Computed across multiple feature levels (low, mid, high)
       - Encourages similar intermediate representations
    
    Args:
        alpha (float): Weight for cross-entropy loss (default: 1.0)
        beta (float): Weight for KL divergence loss (default: 0.5)
        gamma (float): Weight for feature cosine loss (default: 0.3)
        temperature (float): Temperature for softening distributions (default: 4.0)
        num_classes (int): Number of segmentation classes (default: 21)
    """
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3, temperature=4.0, num_classes=21):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.num_classes = num_classes
        
        # Cross-entropy loss for ground truth supervision
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        
        # KL divergence for response-based distillation
        # Uses log-probabilities for numerical stability
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, student_features, 
                teacher_features, targets):
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: [B, C, H, W] - Student model outputs (raw logits)
            teacher_logits: [B, C, H, W] - Teacher model outputs (raw logits) 
            student_features: dict - {'low': [B, C1, H/4, W/4], 'mid': [B, C2, H/8, W/8], 
                                     'high': [B, C3, H/16, W/16]}
            teacher_features: dict - Same structure as student_features
            targets: [B, H, W] - Ground truth segmentation masks
            
        Returns:
            tuple: (total_loss, ce_loss, kd_loss, feat_loss) - All loss components
        """
        
        # ===== 1. Cross-Entropy Loss (Standard Supervision) =====
        # This provides direct supervision from ground truth labels
        # Gradients: ∂L_CE/∂student_logits flows back to student
        ce_loss = self.ce_loss(student_logits, targets)
        
        # ===== 2. KL Divergence Loss (Response-based Distillation) =====
        # Soften probability distributions using temperature T
        # Higher T → smoother distributions → more information in dark knowledge
        T = self.temperature
        
        # Convert logits to probabilities with temperature scaling
        # softmax(z/T) = exp(z/T) / sum(exp(z/T))
        student_soft = F.log_softmax(student_logits / T, dim=1)  # Log for numerical stability
        teacher_soft = F.softmax(teacher_logits / T, dim=1)      # Target probabilities
        
        # KL divergence: D_KL(P || Q) = sum(P * log(P/Q))
        # Measures how much teacher distribution differs from student
        # Multiply by T² to compensate for gradient magnitude: ∂(softmax(z/T))/∂z ∝ 1/T
        kd_loss = self.kl_loss(student_soft, teacher_soft) * (T * T)
        
        # Note: Teacher is frozen, so ∂kd_loss/∂teacher_logits is computed but not used
        # Only ∂kd_loss/∂student_logits flows back to student network
        
        # ===== 3. Feature Cosine Similarity Loss (Feature-based Distillation) =====
        # Match intermediate representations between student and teacher
        # This helps student learn similar feature hierarchies
        feat_loss = 0.0
        num_feat_levels = 0
        
        for level in ['low', 'mid', 'high']:
            if level in student_features and level in teacher_features:
                s_feat = student_features[level]  # Student features
                t_feat = teacher_features[level]  # Teacher features (frozen, no grad)
                
                # Align spatial dimensions if needed
                # Teacher may have different feature map sizes
                if s_feat.shape != t_feat.shape:
                    # Resize teacher features to match student
                    t_feat = F.interpolate(t_feat, size=s_feat.shape[2:],
                                          mode='bilinear', align_corners=False)
                    
                    # Project channels if different
                    # Use 1x1 conv for channel alignment (lazy initialization)
                    if s_feat.shape[1] != t_feat.shape[1]:
                        if not hasattr(self, f'channel_align_{level}'):
                            # Create channel alignment layer on first use
                            align_layer = nn.Conv2d(t_feat.shape[1], s_feat.shape[1], 
                                                   kernel_size=1, bias=False)
                            align_layer = align_layer.to(t_feat.device)
                            setattr(self, f'channel_align_{level}', align_layer)
                        
                        align_layer = getattr(self, f'channel_align_{level}')
                        t_feat = align_layer(t_feat)
                
                # Compute cosine similarity loss
                # Flatten spatial dimensions: [B, C, H, W] → [B, C, H*W]
                s_flat = s_feat.flatten(2)
                t_flat = t_feat.flatten(2).detach()  # Detach to ensure no teacher gradients
                
                # Cosine similarity: cos(θ) = (A·B) / (||A|| ||B||)
                # Range: [-1, 1], where 1 = perfectly aligned
                cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)  # [B, H*W]
                
                # Loss: 1 - cosine_similarity (minimize distance)
                # Range: [0, 2], where 0 = perfect match
                feat_loss += (1 - cos_sim).mean()
                num_feat_levels += 1
        
        # Average over feature levels
        if num_feat_levels > 0:
            feat_loss = feat_loss / num_feat_levels
        
        # ===== Total Loss (Weighted Combination) =====
        # α controls ground truth supervision strength
        # β controls soft target knowledge transfer
        # γ controls feature representation matching
        total_loss = (self.alpha * ce_loss + 
                     self.beta * kd_loss + 
                     self.gamma * feat_loss)
        
        return total_loss, ce_loss, kd_loss, feat_loss


class TeacherModelWrapper(nn.Module):
    """
    Wrapper for FCN-ResNet50 teacher model with feature extraction.
    
    The teacher model is frozen (no gradient updates) and used only for
    generating soft targets and intermediate features for the student.
    
    Feature Extraction Points:
    - Low-level:  After layer1 (stride 4)
    - Mid-level:  After layer2 (stride 8)  
    - High-level: After layer3 (stride 16)
    """
    
    def __init__(self, num_classes=21):
        super(TeacherModelWrapper, self).__init__()
        
        # Load pretrained FCN-ResNet50
        weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = fcn_resnet50(weights=weights, num_classes=num_classes)
        
        # Freeze all parameters - teacher is NOT trainable
        # This ensures ∂L/∂θ_teacher = 0
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set to evaluation mode (disables dropout, batchnorm updates)
        self.model.eval()
        
    @torch.no_grad()  # Explicitly disable gradient computation for efficiency
    def forward(self, x):
        """
        Forward pass through teacher model.
        
        Args:
            x: [B, 3, H, W] - Input images
            
        Returns:
            tuple: (logits, features_dict)
                - logits: [B, num_classes, H, W]
                - features: {'low': [...], 'mid': [...], 'high': [...]}
        """
        input_shape = x.shape[-2:]  # Capture original input size [H, W]
        features_dict = {}
        
        # ResNet backbone feature extraction
        x = self.model.backbone.conv1(x)
        x = self.model.backbone.bn1(x)
        x = self.model.backbone.relu(x)
        x = self.model.backbone.maxpool(x)
        
        # Low-level features (stride 4)
        x = self.model.backbone.layer1(x)
        features_dict['low'] = x
        
        # Mid-level features (stride 8)
        x = self.model.backbone.layer2(x)
        features_dict['mid'] = x
        
        # High-level features (stride 16)
        x = self.model.backbone.layer3(x)
        features_dict['high'] = x
        
        # Final features (stride 16)
        x = self.model.backbone.layer4(x)
        
        # FCN classifier head
        x = self.model.classifier(x)
        
        # Upsample to input resolution (not x.shape, but original input_shape!)
        logits = F.interpolate(x, size=input_shape, mode='bilinear', 
                              align_corners=False)
        
        return logits, features_dict


def train_one_epoch(student, teacher, dataloader, optimizer, criterion, 
                   device, epoch, total_epochs):
    """
    Train student model for one epoch using knowledge distillation.
    
    Gradient Flow:
    - Student: Receives gradients from all loss components
    - Teacher: Frozen, no gradient updates
    
    Args:
        student: Student model (trainable)
        teacher: Teacher model (frozen)
        dataloader: Training data loader
        optimizer: Optimizer for student parameters
        criterion: KnowledgeDistillationLoss instance
        device: torch.device
        epoch: Current epoch number
        total_epochs: Total number of epochs
        
    Returns:
        dict: Average losses for the epoch
    """
    student.train()  # Enable training mode (dropout, batchnorm updates)
    teacher.eval()   # Keep teacher in eval mode (frozen)
    
    # Loss accumulators
    total_loss_avg = 0.0
    ce_loss_avg = 0.0
    kd_loss_avg = 0.0
    feat_loss_avg = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = images.to(device)
        targets = targets.to(device)
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # ===== Forward pass through student (with gradients) =====
        student_logits, student_features = student(images)
        
        # ===== Forward pass through teacher (no gradients) =====
        with torch.no_grad():
            teacher_logits, teacher_features = teacher(images)
            # Resize teacher output to match student if needed
            if teacher_logits.shape != student_logits.shape:
                teacher_logits = F.interpolate(teacher_logits, 
                                              size=student_logits.shape[2:],
                                              mode='bilinear', align_corners=False)
        
        # ===== Compute combined loss =====
        total_loss, ce_loss, kd_loss, feat_loss = criterion(
            student_logits, teacher_logits,
            student_features, teacher_features,
            targets
        )
        
        # ===== Backward pass (only student receives gradients) =====
        total_loss.backward()
        
        # ===== Update student parameters =====
        # Teacher parameters are frozen, so optimizer only updates student
        optimizer.step()
        
        # Accumulate losses
        total_loss_avg += total_loss.item()
        ce_loss_avg += ce_loss.item()
        kd_loss_avg += kd_loss.item()
        feat_loss_avg += feat_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Total': f'{total_loss.item():.4f}',
            'CE': f'{ce_loss.item():.4f}',
            'KD': f'{kd_loss.item():.4f}',
            'Feat': f'{feat_loss.item():.4f}'
        })
    
    # Compute averages
    num_batches = len(dataloader)
    return {
        'total_loss': total_loss_avg / num_batches,
        'ce_loss': ce_loss_avg / num_batches,
        'kd_loss': kd_loss_avg / num_batches,
        'feat_loss': feat_loss_avg / num_batches
    }


def evaluate(model, dataloader, device, num_classes=21):
    """
    Evaluate model on validation set and compute mIoU.
    
    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        device: torch.device
        num_classes: Number of classes
        
    Returns:
        float: Mean Intersection over Union (mIoU)
    """
    model.eval()
    
    # Confusion matrix for IoU computation
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            targets = targets.cpu().numpy()
            
            # Forward pass
            if hasattr(model, 'return_features') and model.return_features:
                outputs, _ = model(images)
            else:
                outputs = model(images)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Update confusion matrix
            for pred, target in zip(preds, targets):
                mask = (target >= 0) & (target < num_classes)
                confusion_matrix += np.bincount(
                    num_classes * target[mask].astype(int) + pred[mask],
                    minlength=num_classes ** 2
                ).reshape(num_classes, num_classes)
    
    # Compute mIoU
    miou = compute_miou(confusion_matrix)
    return miou


def main():
    """
    Main training pipeline for knowledge distillation.
    """
    # ===== Configuration =====
    print("=" * 80)
    print("Knowledge Distillation Training Pipeline")
    print("=" * 80)
    
    # Hyperparameters
    NUM_CLASSES = 21
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Knowledge distillation parameters
    ALPHA = 1.0    # Weight for cross-entropy loss
    BETA = 0.5     # Weight for KL divergence loss
    GAMMA = 0.3    # Weight for feature cosine loss
    TEMPERATURE = 4.0  # Softening temperature
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"\nDistillation parameters:")
    print(f"  α (CE weight): {ALPHA}")
    print(f"  β (KD weight): {BETA}")
    print(f"  γ (Feature weight): {GAMMA}")
    print(f"  T (Temperature): {TEMPERATURE}")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ===== Load Dataset =====
    print("\nLoading PASCAL VOC 2012 dataset...")
    
    train_dataset = VOCSegmentationWithJointTransform(
        root=config.DATA_ROOT,
        image_set='train',
        image_size=512,
        is_training=True
    )
    
    val_dataset = VOCSegmentationWithJointTransform(
        root=config.DATA_ROOT,
        image_set='val',
        image_size=512,
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility (avoids pickle issues)
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility (avoids pickle issues)
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # ===== Initialize Models =====
    print("\nInitializing models...")
    
    # Student model (trainable)
    student = LightweightSegmentationModel(
        num_classes=NUM_CLASSES,
        pretrained=True,
        return_features=True  # Enable feature extraction for distillation
    ).to(device)
    
    # Load pre-trained student checkpoint (your already-trained model)
    # Find the checkpoint file ending with "_best.pth"
    pretrained_checkpoint = None
    if os.path.exists(config.CHECKPOINT_DIR):
        for filename in os.listdir(config.CHECKPOINT_DIR):
            if filename.endswith('_best.pth'):
                pretrained_checkpoint = os.path.join(config.CHECKPOINT_DIR, filename)
                break
    
    if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
        print(f"Loading pre-trained student checkpoint: {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, "
              f"mIoU: {checkpoint.get('best_miou', checkpoint.get('miou', 'unknown')):.4f}")
    else:
        print(f"WARNING: No checkpoint ending with '_best.pth' found in {config.CHECKPOINT_DIR}")
        print("Starting from ImageNet-pretrained backbone only (random decoder)")
    
    # Teacher model (frozen)
    teacher = TeacherModelWrapper(num_classes=NUM_CLASSES).to(device)
    
    # Count parameters
    student_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    
    print(f"\nStudent parameters: {student_params:,} ({student_params/1e6:.2f}M)")
    print(f"Teacher parameters: {teacher_params:,} ({teacher_params/1e6:.2f}M)")
    print(f"Compression ratio: {teacher_params/student_params:.2f}x")
    
    # ===== Setup Training =====
    print("\nSetting up training...")
    
    # Loss function
    criterion = KnowledgeDistillationLoss(
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        temperature=TEMPERATURE,
        num_classes=NUM_CLASSES
    ).to(device)
    
    # Optimizer (only student parameters)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )
    
    # ===== Baseline Evaluation (Student without KD) =====
    print("\n" + "=" * 80)
    print("Evaluating student BEFORE knowledge distillation...")
    print("=" * 80)
    
    student.return_features = False  # Disable features for faster evaluation
    baseline_miou = evaluate(student, val_loader, device, NUM_CLASSES)
    student.return_features = True  # Re-enable for training
    
    print(f"\nBaseline mIoU (student without KD): {baseline_miou:.4f}")
    
    # ===== Training Loop =====
    print("\n" + "=" * 80)
    print("Starting knowledge distillation training...")
    print("=" * 80)
    
    best_miou = 0.0
    training_history = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train one epoch
        train_losses = train_one_epoch(
            student, teacher, train_loader, optimizer, criterion,
            device, epoch, NUM_EPOCHS
        )
        
        # Evaluate on validation set
        student.return_features = False  # Disable features for evaluation
        val_miou = evaluate(student, val_loader, device, NUM_CLASSES)
        student.return_features = True  # Re-enable for training
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{NUM_EPOCHS} Summary")
        print(f"{'='*80}")
        print(f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        print(f"\nLoss Components:")
        print(f"  Total Loss: {train_losses['total_loss']:.4f}")
        print(f"  CE Loss:    {train_losses['ce_loss']:.4f}  (α={ALPHA})")
        print(f"  KD Loss:    {train_losses['kd_loss']:.4f}  (β={BETA}, T={TEMPERATURE})")
        print(f"  Feat Loss:  {train_losses['feat_loss']:.4f}  (γ={GAMMA})")
        print(f"\nValidation mIoU: {val_miou:.4f}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 
                                          'student_kd_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': val_miou,
                'baseline_miou': baseline_miou,
                'distillation_params': {
                    'alpha': ALPHA,
                    'beta': BETA,
                    'gamma': GAMMA,
                    'temperature': TEMPERATURE
                }
            }, checkpoint_path)
            print(f"✓ Best model saved (mIoU: {best_miou:.4f})")
        
        # Track history
        training_history.append({
            'epoch': epoch,
            'train_losses': train_losses,
            'val_miou': val_miou,
            'lr': current_lr
        })
    
    # ===== Final Evaluation =====
    print("\n" + "=" * 80)
    print("Training Complete - Final Results")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'Value':<15} {'Improvement'}")
    print(f"{'-'*30} {'-'*15} {'-'*15}")
    print(f"{'Baseline mIoU (no KD)':<30} {baseline_miou:.4f}")
    print(f"{'Best mIoU (with KD)':<30} {best_miou:.4f}        "
          f"{'+' if best_miou > baseline_miou else ''}"
          f"{(best_miou - baseline_miou):.4f} "
          f"({100*(best_miou - baseline_miou)/baseline_miou:+.2f}%)")
    
    print(f"\n{'Student Parameters':<30} {student_params/1e6:.2f}M")
    print(f"{'Teacher Parameters':<30} {teacher_params/1e6:.2f}M")
    print(f"{'Compression Ratio':<30} {teacher_params/student_params:.2f}x")
    
    print("\n" + "=" * 80)
    print(f"Best model saved to: {config.CHECKPOINT_DIR}/student_kd_best.pth")
    print("=" * 80)
    
    return training_history


if __name__ == "__main__":
    main()

