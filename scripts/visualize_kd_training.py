"""
Visualization script for Knowledge Distillation training metrics.

This script loads the KD training history and creates plots for:
- Total Loss, CE Loss, KD Loss, and Feature Loss
- Validation mIoU with baseline comparison
- Learning rate schedule

Usage:
    python scripts/visualize_kd_training.py
    python scripts/visualize_kd_training.py --history checkpoints/kd_training_history.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_kd_training_history(history_path='checkpoints/kd_training_history.pth', output_dir='visualizations'):
    """
    Load and plot knowledge distillation training history.
    
    Args:
        history_path: Path to the KD training history file
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training history
    print(f"Loading KD training history from: {history_path}")
    if not os.path.exists(history_path):
        print(f"Error: History file not found at {history_path}")
        print("\nMake sure you've run the knowledge distillation training first:")
        print("  python scripts/train_knowledge_distillation.py")
        return
    
    checkpoint = torch.load(history_path, weights_only=False, map_location=torch.device('cpu'))
    
    if 'history' not in checkpoint:
        print(f"Error: Invalid history file format")
        return
    
    history = checkpoint['history']
    distillation_params = checkpoint.get('distillation_params', {})
    baseline_miou = checkpoint.get('baseline_miou', None)
    best_miou = checkpoint.get('best_miou', None)
    
    # Extract metrics from history
    epochs = [entry['epoch'] for entry in history]
    total_loss = [entry['train_losses']['total_loss'] for entry in history]
    ce_loss = [entry['train_losses']['ce_loss'] for entry in history]
    kd_loss = [entry['train_losses']['kd_loss'] for entry in history]
    feat_loss = [entry['train_losses']['feat_loss'] for entry in history]
    val_miou = [entry['val_miou'] for entry in history]
    learning_rates = [entry['lr'] for entry in history]
    
    print(f"\nLoaded {len(epochs)} epochs of training data")
    
    # Print distillation parameters
    print(f"\n{'='*80}")
    print(f"Knowledge Distillation Parameters:")
    print(f"{'='*80}")
    print(f"  α (alpha):     {distillation_params.get('alpha', 'N/A')}  - CE loss weight")
    print(f"  β (beta):      {distillation_params.get('beta', 'N/A')}  - KD loss weight")
    print(f"  γ (gamma):     {distillation_params.get('gamma', 'N/A')}  - Feature loss weight")
    print(f"  T (temperature): {distillation_params.get('temperature', 'N/A')}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Training Summary:")
    print(f"{'='*80}")
    if baseline_miou:
        print(f"  Baseline mIoU (no KD): {baseline_miou:.4f}")
    if best_miou:
        print(f"  Best mIoU (with KD):   {best_miou:.4f}")
        if baseline_miou:
            improvement = best_miou - baseline_miou
            improvement_pct = 100 * improvement / baseline_miou
            print(f"  Improvement:           {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print(f"\n  Final Total Loss: {total_loss[-1]:.4f}")
    print(f"  Final CE Loss:    {ce_loss[-1]:.4f}")
    print(f"  Final KD Loss:    {kd_loss[-1]:.4f}")
    print(f"  Final Feat Loss:  {feat_loss[-1]:.4f}")
    print(f"  Final Val mIoU:   {val_miou[-1]:.4f}")
    print(f"{'='*80}\n")
    
    # ===== Create comprehensive visualization =====
    
    # Figure 1: All Loss Components
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Knowledge Distillation Training - Loss Components', 
                  fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, total_loss, 'b-', linewidth=2, marker='o', 
                    markersize=4, markevery=max(1, len(epochs)//20))
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(epochs[0], epochs[-1])
    
    # Add text with final value
    textstr = f'Final: {total_loss[-1]:.4f}\nMin: {min(total_loss):.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    axes[0, 0].text(0.05, 0.95, textstr, transform=axes[0, 0].transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props)
    
    # Plot 2: Cross-Entropy Loss
    axes[0, 1].plot(epochs, ce_loss, 'r-', linewidth=2, marker='s', 
                    markersize=4, markevery=max(1, len(epochs)//20))
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    alpha_val = distillation_params.get('alpha', 'N/A')
    axes[0, 1].set_title(f'Cross-Entropy Loss (α={alpha_val})', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(epochs[0], epochs[-1])
    
    textstr = f'Final: {ce_loss[-1]:.4f}\nMin: {min(ce_loss):.4f}'
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
    axes[0, 1].text(0.05, 0.95, textstr, transform=axes[0, 1].transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props)
    
    # Plot 3: KL Divergence Loss (Response-based Distillation)
    axes[1, 0].plot(epochs, kd_loss, 'g-', linewidth=2, marker='^', 
                    markersize=4, markevery=max(1, len(epochs)//20))
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    beta_val = distillation_params.get('beta', 'N/A')
    temp_val = distillation_params.get('temperature', 'N/A')
    axes[1, 0].set_title(f'KL Divergence Loss (β={beta_val}, T={temp_val})', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(epochs[0], epochs[-1])
    
    textstr = f'Final: {kd_loss[-1]:.4f}\nMin: {min(kd_loss):.4f}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    axes[1, 0].text(0.05, 0.95, textstr, transform=axes[1, 0].transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props)
    
    # Plot 4: Feature Loss (Feature-based Distillation)
    axes[1, 1].plot(epochs, feat_loss, 'm-', linewidth=2, marker='d', 
                    markersize=4, markevery=max(1, len(epochs)//20))
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    gamma_val = distillation_params.get('gamma', 'N/A')
    axes[1, 1].set_title(f'Feature Cosine Loss (γ={gamma_val})', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(epochs[0], epochs[-1])
    
    textstr = f'Final: {feat_loss[-1]:.4f}\nMin: {min(feat_loss):.4f}'
    props = dict(boxstyle='round', facecolor='plum', alpha=0.8)
    axes[1, 1].text(0.05, 0.95, textstr, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'kd_loss_components.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss components plot saved to: {plot_path}")
    
    plt.show()
    
    # ===== Figure 2: Validation mIoU =====
    fig2, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(epochs, val_miou, 'b-', linewidth=2.5, marker='o', 
            markersize=5, markevery=max(1, len(epochs)//20), label='Validation mIoU')
    
    # Add baseline if available
    if baseline_miou:
        ax.axhline(y=baseline_miou, color='orange', linestyle='--', linewidth=2,
                   label=f'Baseline (no KD): {baseline_miou:.4f}')
    
    # Mark best mIoU
    best_epoch_idx = val_miou.index(max(val_miou))
    best_epoch_num = epochs[best_epoch_idx]
    best_miou_val = max(val_miou)
    ax.plot(best_epoch_num, best_miou_val, 'r*', markersize=20, 
            label=f'Best: {best_miou_val:.4f} (Epoch {best_epoch_num})')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('mIoU', fontsize=14)
    ax.set_title('Knowledge Distillation - Validation mIoU', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(epochs[0], epochs[-1])
    
    # Add improvement annotation if baseline exists
    if baseline_miou:
        improvement = best_miou_val - baseline_miou
        improvement_pct = 100 * improvement / baseline_miou
        textstr = f'Best mIoU: {best_miou_val:.4f}\n'
        textstr += f'Baseline: {baseline_miou:.4f}\n'
        textstr += f'Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save validation plot
    val_plot_path = os.path.join(output_dir, 'kd_validation_miou.png')
    plt.savefig(val_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Validation mIoU plot saved to: {val_plot_path}")
    
    plt.show()
    
    # ===== Figure 3: Combined Loss Comparison =====
    fig3, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(epochs, total_loss, 'b-', linewidth=2, marker='o', markersize=3,
            markevery=max(1, len(epochs)//20), label='Total Loss', alpha=0.9)
    ax.plot(epochs, ce_loss, 'r--', linewidth=2, marker='s', markersize=3,
            markevery=max(1, len(epochs)//20), label='CE Loss', alpha=0.7)
    ax.plot(epochs, kd_loss, 'g--', linewidth=2, marker='^', markersize=3,
            markevery=max(1, len(epochs)//20), label='KD Loss', alpha=0.7)
    ax.plot(epochs, feat_loss, 'm--', linewidth=2, marker='d', markersize=3,
            markevery=max(1, len(epochs)//20), label='Feature Loss', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Knowledge Distillation - All Loss Components', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(epochs[0], epochs[-1])
    
    # Add formula text
    formula_text = 'L_total = α·L_CE + β·L_KD + γ·L_feat'
    ax.text(0.5, 0.02, formula_text, transform=ax.transAxes, 
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = os.path.join(output_dir, 'kd_all_losses.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined losses plot saved to: {combined_plot_path}")
    
    plt.show()
    
    # ===== Figure 4: Learning Rate Schedule =====
    fig4, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, learning_rates, 'purple', linewidth=2.5, marker='o', 
            markersize=4, markevery=max(1, len(epochs)//20))
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_yscale('log')  # Log scale for better visualization
    
    # Add initial and final LR
    textstr = f'Initial LR: {learning_rates[0]:.6f}\nFinal LR: {learning_rates[-1]:.6f}'
    props = dict(boxstyle='round', facecolor='lavender', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save LR plot
    lr_plot_path = os.path.join(output_dir, 'kd_learning_rate.png')
    plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Learning rate plot saved to: {lr_plot_path}")
    
    plt.show()
    
    print(f"\n{'='*80}")
    print(f"All plots saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Knowledge Distillation training history')
    parser.add_argument('--history', type=str, 
                       default=None,
                       help='Path to the KD training history file')
    parser.add_argument('--method', type=str,
                       choices=['response', 'feature', 'both'],
                       default=None,
                       help='Which method to visualize (auto-detects history file)')
    parser.add_argument('--output', type=str, 
                       default='visualizations',
                       help='Directory to save visualization plots')
    
    args = parser.parse_args()
    
    # Adjust path if running from scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Auto-detect history file based on method
    if args.history is None:
        if args.method:
            args.history = f'checkpoints_optimized/kd_training_history_{args.method}.pth'
        else:
            # Default to response method
            args.history = 'checkpoints_optimized/kd_training_history_response.pth'
            print(f"No method specified, defaulting to: {args.history}")
    
    if not os.path.isabs(args.history):
        args.history = os.path.join(script_dir, '', args.history)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, '', args.output)
    
    plot_kd_training_history(args.history, args.output)

