"""
Test and evaluation script for Lightweight Segmentation Model.

This script:
- Loads a trained model checkpoint
- Evaluates on validation images
- Saves predicted segmentation masks
- Computes and displays metrics

Usage:
    python test_model.py --checkpoint ./checkpoints/checkpoint_epoch_50_best.pth --output_dir ./results
"""

import os
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightweight_segmentation_model import LightweightSegmentationModel
from utils_dataset import VOCSegmentationWithJointTransform, denormalize_image
from utils_metrics import SegmentationMetrics
from utils_common import VOC_CLASSES, colorize_mask, load_checkpoint as load_checkpoint_common, print_class_iou


def visualize_predictions(images, masks, predictions, num_samples=4, save_path=None):
    """
    Visualize images with ground truth and predicted masks.
    
    Args:
        images (torch.Tensor): Batch of images [B, 3, H, W]
        masks (torch.Tensor): Ground truth masks [B, H, W]
        predictions (torch.Tensor): Predicted masks [B, H, W]
        num_samples (int): Number of samples to visualize
        save_path (str): Path to save figure
    """
    batch_size = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalize images
    images_denorm = denormalize_image(images.cpu())
    
    for i in range(batch_size):
        # Original image
        img = images_denorm[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        gt_mask = masks[i].cpu().numpy()
        gt_colored = colorize_mask(gt_mask)
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        pred_mask = predictions[i].cpu().numpy()
        pred_colored = colorize_mask(pred_mask)
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.close()


def save_prediction_masks(images, masks, predictions, image_ids, output_dir):
    """
    Save individual prediction masks as images.
    
    Args:
        images (torch.Tensor): Batch of images [B, 3, H, W]
        masks (torch.Tensor): Ground truth masks [B, H, W]
        predictions (torch.Tensor): Predicted masks [B, H, W]
        image_ids (list): List of image IDs
        output_dir (str): Output directory
    """
    batch_size = images.shape[0]
    
    # Create subdirectories
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'colored'), exist_ok=True)
    
    for i in range(batch_size):
        img_id = image_ids[i] if i < len(image_ids) else f"image_{i}"
        
        # Save prediction mask (grayscale, class indices)
        pred_mask = predictions[i].cpu().numpy().astype(np.uint8)
        pred_img = Image.fromarray(pred_mask)
        pred_img.save(os.path.join(output_dir, 'predictions', f'{img_id}.png'))
        
        # Save colored prediction
        pred_colored = colorize_mask(pred_mask)
        pred_colored_img = Image.fromarray(pred_colored)
        pred_colored_img.save(os.path.join(output_dir, 'colored', f'{img_id}.png'))


def evaluate_model(model, val_loader, device, output_dir, save_masks=True, 
                   visualize=True, num_classes=21):
    """
    Evaluate model on validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to use
        output_dir: Output directory for results
        save_masks: Whether to save prediction masks
        visualize: Whether to create visualizations
        num_classes: Number of classes
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=255)
    
    print("\nEvaluating model...")
    print("-" * 60)
    
    # Create output directories
    if save_masks:
        os.makedirs(output_dir, exist_ok=True)
    
    vis_count = 0
    max_vis = 10  # Maximum number of visualization images
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            # Update metrics
            metrics.update(predictions, masks)
            
            # Save masks (first few batches)
            if save_masks and batch_idx < 5:
                image_ids = [f"batch{batch_idx}_img{i}" for i in range(images.shape[0])]
                save_prediction_masks(images, masks, predictions, image_ids, output_dir)
            
            # Visualize (first few batches)
            if visualize and vis_count < max_vis:
                vis_path = os.path.join(output_dir, f'visualization_batch_{batch_idx}.png')
                visualize_predictions(images, masks, predictions, 
                                    num_samples=min(4, images.shape[0]), 
                                    save_path=vis_path)
                vis_count += 1
    
    # Compute final metrics
    miou = metrics.get_miou()
    pixel_acc = metrics.get_pixel_accuracy()
    mean_acc = metrics.get_mean_accuracy()
    iou_per_class = metrics.get_iou_per_class()
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean IoU (mIoU):      {miou:.4f}")
    print(f"Pixel Accuracy:       {pixel_acc:.4f}")
    print(f"Mean Accuracy:        {mean_acc:.4f}")
    
    print("\nPer-class IoU:")
    print("-" * 60)
    for i, (class_name, iou) in enumerate(zip(VOC_CLASSES, iou_per_class)):
        print(f"  {i:2d}. {class_name:15s}: {iou:.4f}")
    print("=" * 60)
    
    results = {
        'miou': miou,
        'pixel_acc': pixel_acc,
        'mean_acc': mean_acc,
        'iou_per_class': iou_per_class
    }
    
    # Save results to file
    results_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean IoU (mIoU):      {miou:.4f}\n")
        f.write(f"Pixel Accuracy:       {pixel_acc:.4f}\n")
        f.write(f"Mean Accuracy:        {mean_acc:.4f}\n\n")
        f.write("Per-class IoU:\n")
        f.write("-" * 60 + "\n")
        for i, (class_name, iou) in enumerate(zip(VOC_CLASSES, iou_per_class)):
            f.write(f"  {i:2d}. {class_name:15s}: {iou:.4f}\n")
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def load_checkpoint(checkpoint_path, model, device):
    """
    Load model checkpoint (wrapper for utils_common.load_checkpoint).
    
    Args:
        checkpoint_path (str): Path to checkpoint
        model: PyTorch model
        device: Device to load to
    
    Returns:
        dict: Checkpoint data
    """
    return load_checkpoint_common(checkpoint_path, model=model, device=device)


def main(args):
    """
    Main evaluation function.
    
    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("Lightweight Segmentation Model - Evaluation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Data root:      {args.data_root}")
    print(f"  Image size:     {args.image_size}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Output dir:     {args.output_dir}")
    print()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = VOCSegmentationWithJointTransform(
        root=args.data_root,
        image_set=args.val_set,
        image_size=args.image_size,
        is_training=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Validation batches: {len(val_loader)}\n")
    
    # Create model
    print("Creating model...")
    model = LightweightSegmentationModel(
        num_classes=args.num_classes,
        pretrained=False,
        return_features=False
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    
    # Evaluate
    results = evaluate_model(
        model, val_loader, device, args.output_dir,
        save_masks=args.save_masks,
        visualize=args.visualize,
        num_classes=args.num_classes
    )
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Lightweight Segmentation Model')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=21,
                       help='Number of segmentation classes')
    
    # Data parameters
    parser.add_argument('--data_root', type=str,
                       default='./data/VOC2012_train_val/VOC2012_train_val',
                       help='Root directory of VOC dataset')
    parser.add_argument('--val_set', type=str, default='val',
                       help='Validation set to use')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_masks', action='store_true', default=True,
                       help='Save predicted masks')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualization images')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    main(args)

