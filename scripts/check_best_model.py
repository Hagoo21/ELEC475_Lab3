"""
Quick utility to check which epoch had the best performance.

Usage:
    python check_best_model.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.common import load_training_history, find_best_epoch

def check_best_model(checkpoint_dir='../checkpoints'):
    """Check which epoch had the best mIoU."""
    
    # Load training history
    history = load_training_history(checkpoint_dir)
    
    if history is None:
        print(f"Error: No training history found in {checkpoint_dir}/")
        print("Please make sure you've completed training first.")
        return
    
    # Find best epoch
    best_epoch, best_miou = find_best_epoch(history)
    
    if best_epoch is None:
        print(f"Error: Could not find best epoch in training history.")
        return
    
    # Load checkpoint_latest.pth to get current epoch info
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    
    if os.path.exists(latest_path):
        print(f"Loading from: {latest_path}\n")
        checkpoint = torch.load(latest_path, weights_only=False, map_location=torch.device('cpu'))
        
        val_miou = history['val_miou']
        
        print("=" * 70)
        print("BEST MODEL INFORMATION")
        print("=" * 70)
        print(f"  Best Epoch:      {best_epoch}")
        print(f"  Best mIoU:       {best_miou:.4f}")
        print(f"  Checkpoint File: checkpoint_epoch_{best_epoch}_best.pth")
        print(f"  Location:        {os.path.join(checkpoint_dir, f'checkpoint_epoch_{best_epoch}_best.pth')}")
        print("=" * 70)
        
        print(f"\nFinal Epoch:     {checkpoint['epoch']}")
        print(f"Final mIoU:      {checkpoint['miou']:.4f}")
        
        # Show top 5 epochs
        print("\n" + "=" * 70)
        print("TOP 5 EPOCHS BY mIoU")
        print("=" * 70)
        
        miou_with_epochs = [(miou, idx + 1) for idx, miou in enumerate(val_miou)]
        miou_with_epochs.sort(reverse=True)
        
        for rank, (miou, epoch) in enumerate(miou_with_epochs[:5], 1):
            print(f"  {rank}. Epoch {epoch:2d}: mIoU = {miou:.4f}")
        print("=" * 70)
    else:
        # Fallback if no latest checkpoint
        print("=" * 70)
        print("BEST MODEL INFORMATION")
        print("=" * 70)
        print(f"  Best Epoch:      {best_epoch}")
        print(f"  Best mIoU:       {best_miou:.4f}")
        print(f"  Checkpoint File: checkpoint_epoch_{best_epoch}_best.pth")
        print("=" * 70)
    
    # Check if best checkpoint exists
    best_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{best_epoch}_best.pth')
    if os.path.exists(best_checkpoint_path):
        print(f"\n[OK] Best checkpoint file exists and is ready to use!")
    else:
        print(f"\n[WARNING] Best checkpoint file not found at {best_checkpoint_path}")


if __name__ == '__main__':
    # Use absolute path to checkpoints directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, '..', 'checkpoints')
    check_best_model(checkpoint_dir)

