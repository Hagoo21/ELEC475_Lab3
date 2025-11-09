"""
Quick utility to check which epoch had the best performance.

Usage:
    python check_best_model.py
"""

import os
import torch

def check_best_model(checkpoint_dir='checkpoints'):
    """Check which epoch had the best mIoU."""
    
    # Try loading from checkpoint_latest.pth first (has best_miou directly)
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    history_path = os.path.join(checkpoint_dir, 'training_history.pth')
    
    if os.path.exists(latest_path):
        print(f"Loading from: {latest_path}\n")
        checkpoint = torch.load(latest_path, weights_only=False, map_location=torch.device('cpu'))
        
        history = checkpoint['history']
        val_miou = history['val_miou']
        
        best_epoch = val_miou.index(max(val_miou)) + 1
        best_miou = max(val_miou)
        
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
        
    elif os.path.exists(history_path):
        print(f"Loading from: {history_path}\n")
        history = torch.load(history_path, weights_only=False, map_location=torch.device('cpu'))
        
        val_miou = history['val_miou']
        best_epoch = val_miou.index(max(val_miou)) + 1
        best_miou = max(val_miou)
        
        print("=" * 70)
        print("BEST MODEL INFORMATION")
        print("=" * 70)
        print(f"  Best Epoch:      {best_epoch}")
        print(f"  Best mIoU:       {best_miou:.4f}")
        print(f"  Checkpoint File: checkpoint_epoch_{best_epoch}_best.pth")
        print("=" * 70)
        
    else:
        print(f"Error: No training history found in {checkpoint_dir}/")
        print("Please make sure you've completed training first.")
        return
    
    # Check if best checkpoint exists
    best_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{best_epoch}_best.pth')
    if os.path.exists(best_checkpoint_path):
        print(f"\n[OK] Best checkpoint file exists and is ready to use!")
    else:
        print(f"\n[WARNING] Best checkpoint file not found at {best_checkpoint_path}")


if __name__ == '__main__':
    check_best_model()

