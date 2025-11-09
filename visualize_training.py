"""
Visualization script for training metrics.

This script loads the training history and creates plots for:
- Training and Validation Loss
- Validation mIoU
- Validation Pixel Accuracy

Usage:
    python visualize_training.py
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils_common import load_training_history, find_best_epoch

def plot_training_history(history_path='checkpoints/training_history.pth', output_dir='visualizations'):
    """
    Load and plot training history.
    
    Args:
        history_path: Path to the training history file or checkpoint directory
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training history
    # If history_path is a directory, try to load from it
    if os.path.isdir(history_path):
        checkpoint_dir = history_path
        history = load_training_history(checkpoint_dir)
    else:
        # Otherwise, assume it's a file path
        print(f"Loading training history from: {history_path}")
        if os.path.exists(history_path):
            history = torch.load(history_path, weights_only=False)
        else:
            # Try treating parent directory as checkpoint dir
            checkpoint_dir = os.path.dirname(history_path) or 'checkpoints'
            history = load_training_history(checkpoint_dir)
    
    if history is None:
        print(f"Error: Could not load training history from {history_path}")
        return
    
    # Extract metrics
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    val_miou = history['val_miou']
    val_pixel_acc = history['val_pixel_acc']
    
    epochs = range(1, len(train_loss) + 1)
    
    # Find best epoch
    best_epoch, best_miou = find_best_epoch(history)
    
    print(f"Loaded {len(train_loss)} epochs of training data")
    print(f"\n{'='*60}")
    print(f"BEST MODEL:")
    print(f"  Epoch: {best_epoch}")
    print(f"  mIoU: {best_miou:.4f}")
    print(f"  Checkpoint: checkpoint_epoch_{best_epoch}_best.pth")
    print(f"{'='*60}")
    print(f"\nFinal Training Loss: {train_loss[-1]:.4f}")
    print(f"Final Validation Loss: {val_loss[-1]:.4f}")
    print(f"Final mIoU: {val_miou[-1]:.4f}")
    print()
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training and Validation Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(1, len(epochs))
    
    # Plot 2: Validation mIoU
    axes[1].plot(epochs, val_miou, 'g-', label='Validation mIoU', linewidth=2)
    axes[1].axhline(y=max(val_miou), color='g', linestyle='--', alpha=0.5, 
                    label=f'Best: {max(val_miou):.4f}')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('mIoU', fontsize=12)
    axes[1].set_title('Validation Mean IoU', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1, len(epochs))
    
    # Plot 3: Validation Pixel Accuracy
    axes[2].plot(epochs, val_pixel_acc, 'm-', label='Pixel Accuracy', linewidth=2)
    axes[2].axhline(y=max(val_pixel_acc), color='m', linestyle='--', alpha=0.5,
                    label=f'Best: {max(val_pixel_acc):.4f}')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Pixel Accuracy', fontsize=12)
    axes[2].set_title('Validation Pixel Accuracy', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(1, len(epochs))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
    # Create a separate detailed loss plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, marker='o', 
            markersize=3, markevery=max(1, len(epochs)//20))
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s',
            markersize=3, markevery=max(1, len(epochs)//20))
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1, len(epochs))
    
    # Add text box with summary statistics
    textstr = f'Final Train Loss: {train_loss[-1]:.4f}\n'
    textstr += f'Final Val Loss: {val_loss[-1]:.4f}\n'
    textstr += f'Min Train Loss: {min(train_loss):.4f}\n'
    textstr += f'Min Val Loss: {min(val_loss):.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save detailed loss plot
    loss_plot_path = os.path.join(output_dir, 'loss_curves_detailed.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Detailed loss plot saved to: {loss_plot_path}")
    
    plt.show()
    
    print("\nTraining Summary:")
    print("=" * 60)
    print(f"Total Epochs: {len(epochs)}")
    print(f"\nLoss:")
    print(f"  Final Training Loss:   {train_loss[-1]:.4f}")
    print(f"  Final Validation Loss: {val_loss[-1]:.4f}")
    print(f"  Min Training Loss:     {min(train_loss):.4f} (Epoch {train_loss.index(min(train_loss)) + 1})")
    print(f"  Min Validation Loss:   {min(val_loss):.4f} (Epoch {val_loss.index(min(val_loss)) + 1})")
    print(f"\nValidation Metrics:")
    print(f"  Final mIoU:            {val_miou[-1]:.4f}")
    print(f"  Best mIoU:             {max(val_miou):.4f} (Epoch {val_miou.index(max(val_miou)) + 1})")
    print(f"  Final Pixel Accuracy:  {val_pixel_acc[-1]:.4f}")
    print(f"  Best Pixel Accuracy:   {max(val_pixel_acc):.4f} (Epoch {val_pixel_acc.index(max(val_pixel_acc)) + 1})")
    print("=" * 60)


if __name__ == '__main__':
    # Check if training history exists
    history_path = 'checkpoints/training_history.pth'
    
    if not os.path.exists(history_path):
        print(f"Error: Training history not found at {history_path}")
        print("Please run training first to generate the history file.")
    else:
        plot_training_history(history_path)

