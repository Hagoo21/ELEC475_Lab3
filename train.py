"""
Training script for Lightweight Segmentation Model on PASCAL VOC 2012.

This script implements:
- Data loading with proper preprocessing
- Training loop with CrossEntropyLoss
- Validation with mIoU computation
- Model checkpointing (saves best model)
- Learning rate scheduling

Usage:
    python train.py --epochs 50 --batch_size 8 --image_size 512 --lr 1e-4
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from lightweight_segmentation_model import LightweightSegmentationModel, count_parameters
from utils_dataset import get_voc_dataloaders
from utils_metrics import SegmentationMetrics
import config  # Import global configuration


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
    
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    print(f"\nEpoch {epoch} - Training:")
    
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", 
                total=len(train_loader), leave=True)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        
        # Compute loss
        loss = criterion(logits, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    print(f"  Training complete - Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
    
    return avg_loss


def validate(model, val_loader, criterion, device, epoch, num_classes=21):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
        num_classes: Number of segmentation classes
    
    Returns:
        tuple: (avg_loss, miou, pixel_acc)
    """
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=255)
    
    print(f"\nEpoch {epoch} - Validation:")
    
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", 
                total=len(val_loader), leave=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(images)
            
            # Compute loss
            loss = criterion(logits, masks)
            total_loss += loss.item()
            
            # Compute metrics
            predictions = torch.argmax(logits, dim=1)
            metrics.update(predictions, masks)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Calculate final metrics
    avg_loss = total_loss / len(val_loader)
    miou = metrics.get_miou()
    pixel_acc = metrics.get_pixel_accuracy()
    elapsed = time.time() - start_time
    
    print(f"  Validation complete - Loss: {avg_loss:.4f} | mIoU: {miou:.4f} | "
          f"Pixel Acc: {pixel_acc:.4f} | Time: {elapsed:.1f}s")
    
    return avg_loss, miou, pixel_acc


def save_checkpoint(model, optimizer, scheduler, epoch, miou, save_path, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        miou: Current mIoU
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'miou': miou
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"  [BEST MODEL] Saved to {best_path}")


def main(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("Lightweight Segmentation Model - Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data root:      {args.data_root}")
    print(f"  Image size:     {args.image_size}×{args.image_size}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Num workers:    {args.num_workers}")
    print(f"  Device:         {args.device}")
    print()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = get_voc_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_set=args.train_set,
        val_set=args.val_set
    )
    
    print(f"  Training samples:   {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print()
    
    # Create model
    print("Creating model...")
    model = LightweightSegmentationModel(
        num_classes=args.num_classes,
        pretrained=True,
        return_features=False
    )
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size:           {total_params * 4 / (1024**2):.2f} MB")
    print()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Define learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        print(f"Using Cosine Annealing LR scheduler (T_max={args.epochs})")
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print(f"Using Step LR scheduler (step_size={args.step_size}, gamma={args.gamma})")
    else:
        scheduler = None
        print("No LR scheduler")
    print()
    
    # Training loop
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    best_miou = 0.0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'val_pixel_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_miou, val_pixel_acc = validate(
            model, val_loader, criterion, device, epoch, num_classes=args.num_classes
        )
        
        # Update learning rate
        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            print(f"\n  Learning rate: {current_lr:.6f} -> {new_lr:.6f}")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_miou'].append(val_miou)
        training_history['val_pixel_acc'].append(val_pixel_acc)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
        is_best = val_miou > best_miou
        
        if is_best:
            best_miou = val_miou
            print(f"\n  [NEW BEST] mIoU improved to {best_miou:.4f}")
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_miou, 
            checkpoint_path, is_best=is_best
        )
        
        # Save latest checkpoint
        latest_path = os.path.join(args.output_dir, 'checkpoint_latest.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'miou': val_miou,
            'best_miou': best_miou,
            'history': training_history
        }, latest_path)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Training complete
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved in: {args.output_dir}")
    print()
    
    # Save final training history
    history_path = os.path.join(args.output_dir, 'training_history.pth')
    torch.save(training_history, history_path)
    print(f"Training history saved: {history_path}")


if __name__ == '__main__':
    # Validate paths from config.py
    print("\nValidating paths from config.py...")
    if not config.validate_paths():
        print("[ERROR] Please fix DATA_ROOT in config.py")
        exit(1)
    print("[OK] Paths validated!\n")
    
    parser = argparse.ArgumentParser(description='Train Lightweight Segmentation Model')
    
    # Data parameters (path from config.py, rest are defaults)
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT,
                       help='Root directory of VOC dataset')
    parser.add_argument('--train_set', type=str, default='train',
                       choices=['train', 'trainval'],
                       help='Training set to use')
    parser.add_argument('--val_set', type=str, default='val',
                       help='Validation set to use')
    parser.add_argument('--num_classes', type=int, default=21,
                       help='Number of segmentation classes')
    
    # Model parameters
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size (default: 512)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--step_size', type=int, default=10,
                       help='Step size for StepLR (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR (default: 0.1)')
    
    # Output parameters (path from config.py)
    parser.add_argument('--output_dir', type=str, default=config.CHECKPOINT_DIR,
                       help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    main(args)

