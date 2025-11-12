"""
Qualitative Evaluation Script for Lightweight Segmentation Model
=================================================================

This script generates qualitative results (visualizations) from the lightweight
segmentation model, including:
1. Both successful results and failure cases
2. Inference speed measurements (FPS)
3. Overlay visualizations of segmentation masks on images
4. Per-sample IoU metrics for ranking

The script evaluates on the validation set (not training).

Author: ELEC475 Lab 3
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os
import sys
import time
from tqdm import tqdm
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lightweight_segmentation import LightweightSegmentationModel
from utils.dataset import VOCSegmentationWithJointTransform, denormalize_image
from utils.distillation_utils import VOC_CLASSES
import config


# PASCAL VOC color palette for visualization
VOC_COLORMAP = np.array([
    [0, 0, 0],        # 0: background
    [128, 0, 0],      # 1: aeroplane
    [0, 128, 0],      # 2: bicycle
    [128, 128, 0],    # 3: bird
    [0, 0, 128],      # 4: boat
    [128, 0, 128],    # 5: bottle
    [0, 128, 128],    # 6: bus
    [128, 128, 128],  # 7: car
    [64, 0, 0],       # 8: cat
    [192, 0, 0],      # 9: chair
    [64, 128, 0],     # 10: cow
    [192, 128, 0],    # 11: diningtable
    [64, 0, 128],     # 12: dog
    [192, 0, 128],    # 13: horse
    [64, 128, 128],   # 14: motorbike
    [192, 128, 128],  # 15: person
    [0, 64, 0],       # 16: pottedplant
    [128, 64, 0],     # 17: sheep
    [0, 192, 0],      # 18: sofa
    [128, 192, 0],    # 19: train
    [0, 64, 128],     # 20: tvmonitor
], dtype=np.uint8)


def compute_sample_iou(pred_mask, gt_mask, num_classes=21):
    """
    Compute IoU for a single sample (image).
    
    Args:
        pred_mask (np.ndarray): Predicted mask [H, W]
        gt_mask (np.ndarray): Ground truth mask [H, W]
        num_classes (int): Number of classes
        
    Returns:
        float: Mean IoU for this sample
    """
    # Ignore regions with label 255
    valid_mask = (gt_mask >= 0) & (gt_mask < num_classes)
    
    if valid_mask.sum() == 0:
        return 0.0
    
    pred_valid = pred_mask[valid_mask]
    gt_valid = gt_mask[valid_mask]
    
    # Compute IoU for each class present
    ious = []
    present_classes = np.unique(gt_valid)
    
    for cls in present_classes:
        pred_cls = (pred_valid == cls)
        gt_cls = (gt_valid == cls)
        
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        
        if union > 0:
            ious.append(intersection / union)
    
    return np.mean(ious) if len(ious) > 0 else 0.0


def mask_to_rgb(mask, colormap=VOC_COLORMAP):
    """
    Convert segmentation mask to RGB image using VOC colormap.
    
    Args:
        mask (np.ndarray): Segmentation mask [H, W] with class indices
        colormap (np.ndarray): Color palette [num_classes, 3]
        
    Returns:
        np.ndarray: RGB image [H, W, 3]
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(colormap)):
        rgb[mask == class_id] = colormap[class_id]
    
    return rgb


def overlay_mask_on_image(image, mask, alpha=0.5, colormap=VOC_COLORMAP):
    """
    Overlay segmentation mask on image with transparency.
    
    Args:
        image (np.ndarray): RGB image [H, W, 3] in range [0, 1]
        mask (np.ndarray): Segmentation mask [H, W]
        alpha (float): Transparency of overlay (0=transparent, 1=opaque)
        colormap (np.ndarray): Color palette
        
    Returns:
        np.ndarray: Overlayed image [H, W, 3]
    """
    # Convert mask to RGB
    mask_rgb = mask_to_rgb(mask, colormap).astype(np.float32) / 255.0
    
    # Ensure image is in [0, 1] range
    image = np.clip(image, 0, 1)
    
    # Blend image and mask
    overlay = (1 - alpha) * image + alpha * mask_rgb
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def get_present_classes(mask, ignore_background=True):
    """
    Get list of classes present in a mask.
    
    Args:
        mask (np.ndarray): Segmentation mask [H, W]
        ignore_background (bool): Whether to ignore background class (0)
        
    Returns:
        list: List of class indices present
    """
    unique_classes = np.unique(mask)
    # Filter out ignore regions (255) and optionally background (0)
    classes = [c for c in unique_classes if c < 21 and c != 255]
    if ignore_background:
        classes = [c for c in classes if c != 0]
    return classes


def create_legend(present_classes):
    """
    Create legend patches for present classes.
    
    Args:
        present_classes (list): List of class indices
        
    Returns:
        list: List of matplotlib patches for legend
    """
    patches = []
    for cls_id in present_classes:
        color = VOC_COLORMAP[cls_id] / 255.0
        label = VOC_CLASSES[cls_id]
        patches.append(mpatches.Patch(color=color, label=label))
    return patches


def visualize_sample(image, gt_mask, pred_mask, iou, idx, save_path, title_suffix=""):
    """
    Create visualization with original image, ground truth, prediction, and overlay.
    
    Args:
        image (np.ndarray): Original image [3, H, W] normalized
        gt_mask (np.ndarray): Ground truth mask [H, W]
        pred_mask (np.ndarray): Predicted mask [H, W]
        iou (float): Sample IoU score
        idx (int): Sample index
        save_path (str): Path to save visualization
        title_suffix (str): Additional text for title
    """
    # Denormalize image for visualization
    image_vis = denormalize_image(torch.from_numpy(image)).cpu().numpy()
    image_vis = np.transpose(image_vis, (1, 2, 0))  # [H, W, 3]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Sample {idx} - IoU: {iou:.4f} {title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Original image
    axes[0, 0].imshow(image_vis)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. Ground truth mask
    gt_rgb = mask_to_rgb(gt_mask)
    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    axes[0, 1].axis('off')
    
    # 3. Predicted mask
    pred_rgb = mask_to_rgb(pred_mask)
    axes[1, 0].imshow(pred_rgb)
    axes[1, 0].set_title('Prediction', fontsize=14)
    axes[1, 0].axis('off')
    
    # 4. Overlay prediction on image
    overlay = overlay_mask_on_image(image_vis, pred_mask, alpha=0.5)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (Prediction on Image)', fontsize=14)
    axes[1, 1].axis('off')
    
    # Add legend with present classes
    gt_classes = get_present_classes(gt_mask, ignore_background=True)
    if gt_classes:
        legend_patches = create_legend(gt_classes)
        axes[0, 1].legend(handles=legend_patches, loc='upper right', 
                         bbox_to_anchor=(1.0, 1.0), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def evaluate_and_visualize(model, dataloader, device, output_dir, 
                          num_success=10, num_failure=10, num_samples=None):
    """
    Evaluate model and generate visualizations for best and worst cases.
    
    Args:
        model (nn.Module): Segmentation model
        dataloader (DataLoader): Validation data loader
        device (torch.device): Device to run on
        output_dir (str): Directory to save visualizations
        num_success (int): Number of successful cases to visualize
        num_failure (int): Number of failure cases to visualize
        num_samples (int, optional): Maximum number of samples to evaluate (None = all)
        
    Returns:
        dict: Evaluation metrics including average IoU and FPS
    """
    model.eval()
    
    # Storage for results
    all_ious = []
    all_indices = []
    all_images = []
    all_gt_masks = []
    all_pred_masks = []
    
    # Inference speed tracking
    inference_times = []
    
    print("\n" + "="*80)
    print("Evaluating model and collecting results...")
    print("="*80)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc='Processing')):
            # Check if we've reached the sample limit
            if num_samples is not None and len(all_ious) >= num_samples:
                break
            
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Measure inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            # Forward pass
            if hasattr(model, 'return_features'):
                original_return_features = model.return_features
                model.return_features = False
                outputs = model(images)
                model.return_features = original_return_features
            else:
                outputs = model(images)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # Record inference time per image
            inference_time = (end_time - start_time) / batch_size
            inference_times.append(inference_time)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Process each sample in batch
            for i in range(batch_size):
                if num_samples is not None and len(all_ious) >= num_samples:
                    break
                
                img = images[i].cpu().numpy()
                gt_mask = targets[i].numpy()
                pred_mask = preds[i]
                
                # Compute IoU for this sample
                sample_iou = compute_sample_iou(pred_mask, gt_mask)
                
                # Store results
                all_ious.append(sample_iou)
                all_indices.append(batch_idx * dataloader.batch_size + i)
                all_images.append(img)
                all_gt_masks.append(gt_mask)
                all_pred_masks.append(pred_mask)
    
    # Compute statistics
    all_ious = np.array(all_ious)
    avg_iou = np.mean(all_ious)
    median_iou = np.median(all_ious)
    std_iou = np.std(all_ious)
    
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    avg_fps = 1.0 / np.mean(inference_times)
    
    print(f"\n{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}")
    print(f"Total samples evaluated: {len(all_ious)}")
    print(f"\nIoU Statistics:")
    print(f"  Average IoU:  {avg_iou:.4f}")
    print(f"  Median IoU:   {median_iou:.4f}")
    print(f"  Std Dev:      {std_iou:.4f}")
    print(f"  Min IoU:      {np.min(all_ious):.4f}")
    print(f"  Max IoU:      {np.max(all_ious):.4f}")
    print(f"\nInference Speed:")
    print(f"  Average time per image: {avg_inference_time:.2f} ms")
    print(f"  Average FPS:            {avg_fps:.2f}")
    print(f"{'='*80}\n")
    
    # Sort samples by IoU
    sorted_indices = np.argsort(all_ious)
    
    # Select best cases (highest IoU)
    best_indices = sorted_indices[-num_success:][::-1]  # Reverse to get highest first
    
    # Select worst cases (lowest IoU)
    worst_indices = sorted_indices[:num_failure]
    
    # Create output directories
    success_dir = os.path.join(output_dir, 'successful_cases')
    failure_dir = os.path.join(output_dir, 'failure_cases')
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)
    
    # Visualize successful cases
    print("\nGenerating visualizations for successful cases...")
    for rank, idx in enumerate(best_indices):
        sample_idx = all_indices[idx]
        iou = all_ious[idx]
        
        save_path = os.path.join(success_dir, f'success_rank{rank+1}_sample{sample_idx}_iou{iou:.4f}.png')
        visualize_sample(
            all_images[idx], 
            all_gt_masks[idx], 
            all_pred_masks[idx],
            iou, 
            sample_idx, 
            save_path,
            title_suffix=f"(Success Rank {rank+1}/{num_success})"
        )
    
    # Visualize failure cases
    print("\nGenerating visualizations for failure cases...")
    for rank, idx in enumerate(worst_indices):
        sample_idx = all_indices[idx]
        iou = all_ious[idx]
        
        save_path = os.path.join(failure_dir, f'failure_rank{rank+1}_sample{sample_idx}_iou{iou:.4f}.png')
        visualize_sample(
            all_images[idx], 
            all_gt_masks[idx], 
            all_pred_masks[idx],
            iou, 
            sample_idx, 
            save_path,
            title_suffix=f"(Failure Rank {rank+1}/{num_failure})"
        )
    
    # Create summary plot
    print("\nGenerating summary histogram...")
    create_summary_plot(all_ious, avg_iou, avg_fps, output_dir)
    
    # Save detailed metrics to text file
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Qualitative Evaluation Results - Lightweight Segmentation Model\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples evaluated: {len(all_ious)}\n\n")
        f.write("IoU Statistics:\n")
        f.write(f"  Average IoU:  {avg_iou:.6f}\n")
        f.write(f"  Median IoU:   {median_iou:.6f}\n")
        f.write(f"  Std Dev:      {std_iou:.6f}\n")
        f.write(f"  Min IoU:      {np.min(all_ious):.6f}\n")
        f.write(f"  Max IoU:      {np.max(all_ious):.6f}\n\n")
        f.write("Inference Speed:\n")
        f.write(f"  Average time per image: {avg_inference_time:.2f} ms\n")
        f.write(f"  Average FPS:            {avg_fps:.2f}\n\n")
        f.write("="*80 + "\n\n")
        f.write("Top 10 Successful Cases (Highest IoU):\n")
        f.write("-"*80 + "\n")
        for rank, idx in enumerate(best_indices[:10]):
            sample_idx = all_indices[idx]
            iou = all_ious[idx]
            f.write(f"  Rank {rank+1}: Sample {sample_idx:4d} - IoU: {iou:.6f}\n")
        f.write("\n")
        f.write("Top 10 Failure Cases (Lowest IoU):\n")
        f.write("-"*80 + "\n")
        for rank, idx in enumerate(worst_indices[:10]):
            sample_idx = all_indices[idx]
            iou = all_ious[idx]
            f.write(f"  Rank {rank+1}: Sample {sample_idx:4d} - IoU: {iou:.6f}\n")
    
    print(f"\n  Saved detailed metrics to: {metrics_path}")
    
    return {
        'avg_iou': avg_iou,
        'median_iou': median_iou,
        'std_iou': std_iou,
        'avg_fps': avg_fps,
        'avg_inference_time_ms': avg_inference_time,
        'num_samples': len(all_ious)
    }


def create_summary_plot(all_ious, avg_iou, avg_fps, output_dir):
    """
    Create summary histogram of IoU distribution.
    
    Args:
        all_ious (np.ndarray): Array of IoU values
        avg_iou (float): Average IoU
        avg_fps (float): Average FPS
        output_dir (str): Directory to save plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Histogram
    ax.hist(all_ious, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(avg_iou, color='red', linestyle='--', linewidth=2, 
               label=f'Mean IoU: {avg_iou:.4f}')
    ax.axvline(np.median(all_ious), color='green', linestyle='--', linewidth=2,
               label=f'Median IoU: {np.median(all_ious):.4f}')
    
    ax.set_xlabel('IoU Score', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(f'IoU Distribution - Lightweight Model\n'
                 f'Avg FPS: {avg_fps:.2f}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'iou_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved summary plot to: {save_path}")


def main():
    """Main function for qualitative evaluation."""
    parser = argparse.ArgumentParser(description='Qualitative Evaluation of Lightweight Segmentation Model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: best checkpoint in checkpoint_dir)')
    parser.add_argument('--output_dir', type=str, default='./qualitative_results',
                        help='Directory to save visualizations')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_success', type=int, default=10,
                        help='Number of successful cases to visualize')
    parser.add_argument('--num_failure', type=int, default=10,
                        help='Number of failure cases to visualize')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (None = all)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size for evaluation')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Qualitative Evaluation - Lightweight Segmentation Model")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = LightweightSegmentationModel(num_classes=21, pretrained=False, return_features=False)
    
    # Find checkpoint if not specified
    if args.checkpoint is None:
        # Try multiple possible checkpoint directories
        possible_dirs = [
            config.CHECKPOINT_DIR,
            './checkpoints_optimized',
            './scripts/checkpoints_optimized',
            './checkpoints',
            './scripts/checkpoints',
        ]
        
        # Look for best checkpoint (prioritize KD model, then regular model)
        possible_checkpoints = [
            'student_kd_both_best.pth',
            'student_kd_best.pth',
            'checkpoint_epoch_94_best.pth',
        ]
        
        checkpoint_path = None
        checkpoint_dir = None
        
        # Search in all possible directories
        for dir_path in possible_dirs:
            if not os.path.exists(dir_path):
                continue
            
            for ckpt_name in possible_checkpoints:
                ckpt_path = os.path.join(dir_path, ckpt_name)
                if os.path.exists(ckpt_path):
                    checkpoint_path = ckpt_path
                    checkpoint_dir = dir_path
                    break
            
            if checkpoint_path is not None:
                break
        
        # If still not found, look for any file ending with _best.pth
        if checkpoint_path is None:
            for dir_path in possible_dirs:
                if not os.path.exists(dir_path):
                    continue
                for filename in os.listdir(dir_path):
                    if filename.endswith('_best.pth'):
                        checkpoint_path = os.path.join(dir_path, filename)
                        checkpoint_dir = dir_path
                        break
                if checkpoint_path is not None:
                    break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found. Searched directories: {possible_dirs}")
    else:
        checkpoint_path = args.checkpoint
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = VOCSegmentationWithJointTransform(
        root=config.DATA_ROOT,
        image_set='val',
        image_size=args.image_size,
        is_training=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Run evaluation and generate visualizations
    metrics = evaluate_and_visualize(
        model=model,
        dataloader=val_loader,
        device=device,
        output_dir=args.output_dir,
        num_success=args.num_success,
        num_failure=args.num_failure,
        num_samples=args.num_samples
    )
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Successful cases: {args.output_dir}/successful_cases/")
    print(f"  - Failure cases:    {args.output_dir}/failure_cases/")
    print(f"  - IoU distribution: {args.output_dir}/iou_distribution.png")
    print(f"  - Detailed metrics: {args.output_dir}/evaluation_metrics.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

