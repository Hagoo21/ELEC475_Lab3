"""
Enhanced Training Script for Optimized Lightweight Segmentation Model

Improvements over standard training:
- Supports Focal Loss, Dice Loss, and Combined Loss
- Enhanced data augmentation (multi-scale + crop)
- Better for maintaining mIoU with reduced parameters

Usage:
    # Train with Focal Loss and enhanced augmentation
    python scripts/train_optimized.py --loss_type focal --enhanced_aug --epochs 100
    
    # Train with trainval split for best results
    python scripts/train_optimized.py --train_set trainval --loss_type combined --epochs 150
    
    # Resume training
    python scripts/train_optimized.py --resume --epochs 100
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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lightweight_segmentation import LightweightSegmentationModel, count_parameters
from utils.dataset import get_voc_dataloaders
from utils.metrics import SegmentationMetrics
from utils.losses import get_loss_function
from utils.common import get_device
import config


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    print(f"\nEpoch {epoch} - Training:")
    start_time = time.time()
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", 
                total=len(train_loader), leave=True)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    print(f"  Training complete - Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
    return avg_loss


def validate(model, val_loader, criterion, device, epoch, num_classes=21):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=255)
    
    print(f"\nEpoch {epoch} - Validation:")
    start_time = time.time()
    
    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", 
                total=len(val_loader), leave=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            metrics.update(predictions, masks)
            
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    miou = metrics.get_miou()
    pixel_acc = metrics.get_pixel_accuracy()
    elapsed = time.time() - start_time
    
    print(f"  Validation complete - Loss: {avg_loss:.4f} | mIoU: {miou:.4f} | "
          f"Pixel Acc: {pixel_acc:.4f} | Time: {elapsed:.1f}s")
    
    return avg_loss, miou, pixel_acc


def save_checkpoint(model, optimizer, scheduler, epoch, miou, save_path, is_best=False):
    """Save model checkpoint."""
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
    """Main training function."""
    print("=" * 80)
    print("Optimized Lightweight Segmentation Model - Enhanced Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data root:         {args.data_root}")
    print(f"  Train set:         {args.train_set}")
    print(f"  Image size:        {args.image_size}×{args.image_size}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Loss function:     {args.loss_type}")
    print(f"  Enhanced aug:      {args.enhanced_aug}")
    print(f"  Device:            {args.device}")
    print()
    
    device = get_device(args.device, verbose=True)
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders with enhanced augmentation option
    print("Loading datasets...")
    train_loader, val_loader = get_voc_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_set=args.train_set,
        val_set=args.val_set,
        use_enhanced_aug=args.enhanced_aug
    )
    
    print(f"  Training samples:   {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print()
    
    # Create OPTIMIZED model
    print("Creating optimized model...")
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
    
    # Define loss function (supports multiple types)
    print(f"Creating {args.loss_type} loss function...")
    criterion = get_loss_function(
        loss_type=args.loss_type,
        ignore_index=255,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma
    )
    print()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
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
    
    # Initialize training variables
    best_miou = 0.0
    start_epoch = 1
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'val_pixel_acc': []
    }
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint_latest.pth')
        if os.path.exists(checkpoint_path):
            print("=" * 80)
            print("Resuming from checkpoint")
            print("=" * 80)
            print(f"Loading checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_miou = checkpoint.get('best_miou', checkpoint.get('miou', 0.0))
            
            if 'history' in checkpoint:
                training_history = checkpoint['history']
                print(f"  Loaded training history: {len(training_history['train_loss'])} epochs")
            
            print(f"  Resuming from epoch: {start_epoch}")
            print(f"  Best mIoU so far: {best_miou:.4f}")
            print(f"  Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("=" * 80)
            print()
        else:
            print(f"Warning: --resume specified but no checkpoint found at {checkpoint_path}")
            print("Starting training from scratch...")
            print()
    
    # Training loop
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs + 1):
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
    # Validate paths
    print("\nValidating paths from config.py...")
    if not config.validate_paths():
        print("[ERROR] Please fix DATA_ROOT in config.py")
        exit(1)
    print("[OK] Paths validated!\n")
    
    parser = argparse.ArgumentParser(description='Train Optimized Lightweight Segmentation Model')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT)
    parser.add_argument('--train_set', type=str, default='train',
                       choices=['train', 'trainval'],
                       help='Use trainval for best results (2x more data)')
    parser.add_argument('--val_set', type=str, default='val')
    parser.add_argument('--num_classes', type=int, default=21)
    
    # Model parameters
    parser.add_argument('--image_size', type=int, default=512)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100,
                       help='Recommended: 100-150 epochs for best results')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    
    # Loss function parameters
    parser.add_argument('--loss_type', type=str, default='focal',
                       choices=['ce', 'focal', 'dice', 'combined'],
                       help='Loss function: focal (recommended), combined, dice, or ce')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Alpha for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma for focal loss')
    
    # Augmentation parameters
    parser.add_argument('--enhanced_aug', action='store_true',
                       help='Use enhanced augmentation (multi-scale + crop). Recommended!')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'])
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./checkpoints_optimized',
                       help='Separate directory for optimized model checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    main(args)

