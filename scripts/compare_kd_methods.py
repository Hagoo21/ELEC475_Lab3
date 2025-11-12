"""
Knowledge Distillation Methods Comparison
==========================================

This script compares different knowledge distillation methods:
- Without KD (baseline)
- Response-based KD only
- Feature-based KD only
- Both methods combined (optional)

Generates a table with:
- mIoU (accuracy)
- Number of parameters
- Inference speed (ms per image)

Usage:
    python scripts/compare_kd_methods.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm

from models.lightweight_segmentation import LightweightSegmentationModel
from utils.dataset import VOCSegmentationWithJointTransform
from utils.distillation_utils import compute_miou
import config


def measure_inference_speed(model, device, num_samples=100, warmup=10):
    """
    Measure inference speed in milliseconds per image.
    
    Args:
        model: Model to evaluate
        device: torch.device
        num_samples: Number of samples to average over
        warmup: Number of warmup iterations
        
    Returns:
        float: Average inference time in milliseconds
    """
    model.eval()
    
    # Create dummy input (512x512 RGB image)
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure inference time
    print(f"  Measuring inference speed ({num_samples} iterations)...")
    times = []
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="  Benchmarking"):
            start = time.perf_counter()
            _ = model(dummy_input)
            
            # Synchronize GPU to get accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time


def evaluate_model(model, dataloader, device, num_classes=21):
    """
    Evaluate model and compute mIoU.
    
    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        device: torch.device
        num_classes: Number of classes
        
    Returns:
        float: Mean IoU
    """
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='  Evaluating mIoU'):
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
    
    miou = compute_miou(confusion_matrix)
    return miou


def load_model(checkpoint_path, device, num_classes=21):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch.device
        num_classes: Number of classes
        
    Returns:
        tuple: (model, checkpoint_dict)
    """
    model = LightweightSegmentationModel(
        num_classes=num_classes,
        pretrained=False,
        return_features=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def main():
    """
    Compare different knowledge distillation methods.
    """
    print("=" * 80)
    print("Knowledge Distillation Methods Comparison")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    NUM_CLASSES = 21
    BATCH_SIZE = 8
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = VOCSegmentationWithJointTransform(
        root=config.DATA_ROOT,
        image_set='val',
        image_size=512,
        is_training=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Results storage
    results = []
    
    # ===== 1. Baseline (Without KD) =====
    print("\n" + "=" * 80)
    print("1. Evaluating Baseline (Without Knowledge Distillation)")
    print("=" * 80)
    
    # Look for baseline checkpoint (not a KD checkpoint) in multiple locations
    baseline_checkpoint = None
    possible_dirs = [config.CHECKPOINT_DIR, './checkpoints_optimized', './scripts/checkpoints_optimized']
    
    for checkpoint_dir in possible_dirs:
        if os.path.exists(checkpoint_dir):
            for filename in os.listdir(checkpoint_dir):
                if filename.endswith('_best.pth') and 'kd' not in filename.lower():
                    baseline_checkpoint = os.path.join(checkpoint_dir, filename)
                    break
            if baseline_checkpoint:
                break
    
    if baseline_checkpoint and os.path.exists(baseline_checkpoint):
        print(f"Loading baseline model: {baseline_checkpoint}")
        baseline_model, baseline_ckpt = load_model(baseline_checkpoint, device, NUM_CLASSES)
        
        # Count parameters
        num_params = sum(p.numel() for p in baseline_model.parameters())
        
        # Measure inference speed
        print("\nMeasuring inference speed...")
        avg_time, std_time = measure_inference_speed(baseline_model, device)
        
        # Evaluate mIoU
        print("\nEvaluating accuracy...")
        miou = evaluate_model(baseline_model, val_loader, device, NUM_CLASSES)
        
        results.append({
            'method': 'Without KD (Baseline)',
            'miou': miou,
            'params': num_params,
            'inference_ms': avg_time,
            'inference_std': std_time
        })
        
        print(f"\nResults:")
        print(f"  mIoU: {miou:.4f}")
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  Inference: {avg_time:.2f} ± {std_time:.2f} ms/image")
        
        del baseline_model
        torch.cuda.empty_cache()
    else:
        print("WARNING: Baseline checkpoint not found!")
        print(f"Expected checkpoint in {config.CHECKPOINT_DIR} ending with '_best.pth' (not containing 'kd')")
    
    # ===== 2. Response-Based KD =====
    print("\n" + "=" * 80)
    print("2. Evaluating Response-Based Knowledge Distillation")
    print("=" * 80)
    
    # Try multiple possible checkpoint directories
    response_checkpoint = None
    for checkpoint_dir in possible_dirs:
        response_path = os.path.join(checkpoint_dir, 'student_kd_response_best.pth')
        if os.path.exists(response_path):
            response_checkpoint = response_path
            break
    
    if not response_checkpoint:
        response_checkpoint = os.path.join(config.CHECKPOINT_DIR, 'student_kd_response_best.pth')
    
    if os.path.exists(response_checkpoint):
        print(f"Loading response-based model: {response_checkpoint}")
        response_model, response_ckpt = load_model(response_checkpoint, device, NUM_CLASSES)
        
        num_params = sum(p.numel() for p in response_model.parameters())
        
        print("\nMeasuring inference speed...")
        avg_time, std_time = measure_inference_speed(response_model, device)
        
        print("\nEvaluating accuracy...")
        miou = evaluate_model(response_model, val_loader, device, NUM_CLASSES)
        
        results.append({
            'method': 'Response-Based',
            'miou': miou,
            'params': num_params,
            'inference_ms': avg_time,
            'inference_std': std_time
        })
        
        print(f"\nResults:")
        print(f"  mIoU: {miou:.4f}")
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  Inference: {avg_time:.2f} ± {std_time:.2f} ms/image")
        
        del response_model
        torch.cuda.empty_cache()
    else:
        print(f"WARNING: Response-based checkpoint not found at {response_checkpoint}")
        print("Run: python scripts/train_knowledge_distillation.py --method response")
    
    # ===== 3. Feature-Based KD =====
    print("\n" + "=" * 80)
    print("3. Evaluating Feature-Based Knowledge Distillation")
    print("=" * 80)
    
    # Try multiple possible checkpoint directories
    feature_checkpoint = None
    for checkpoint_dir in possible_dirs:
        feature_path = os.path.join(checkpoint_dir, 'student_kd_feature_best.pth')
        if os.path.exists(feature_path):
            feature_checkpoint = feature_path
            break
    
    if not feature_checkpoint:
        feature_checkpoint = os.path.join(config.CHECKPOINT_DIR, 'student_kd_feature_best.pth')
    
    if os.path.exists(feature_checkpoint):
        print(f"Loading feature-based model: {feature_checkpoint}")
        feature_model, feature_ckpt = load_model(feature_checkpoint, device, NUM_CLASSES)
        
        num_params = sum(p.numel() for p in feature_model.parameters())
        
        print("\nMeasuring inference speed...")
        avg_time, std_time = measure_inference_speed(feature_model, device)
        
        print("\nEvaluating accuracy...")
        miou = evaluate_model(feature_model, val_loader, device, NUM_CLASSES)
        
        results.append({
            'method': 'Feature-Based',
            'miou': miou,
            'params': num_params,
            'inference_ms': avg_time,
            'inference_std': std_time
        })
        
        print(f"\nResults:")
        print(f"  mIoU: {miou:.4f}")
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  Inference: {avg_time:.2f} ± {std_time:.2f} ms/image")
        
        del feature_model
        torch.cuda.empty_cache()
    else:
        print(f"WARNING: Feature-based checkpoint not found at {feature_checkpoint}")
        print("Run: python scripts/train_knowledge_distillation.py --method feature")
    
    # ===== Optional: Both Methods =====
    both_checkpoint = None
    for checkpoint_dir in possible_dirs:
        both_path = os.path.join(checkpoint_dir, 'student_kd_both_best.pth')
        if os.path.exists(both_path):
            both_checkpoint = both_path
            break
    
    if not both_checkpoint:
        both_checkpoint = os.path.join(config.CHECKPOINT_DIR, 'student_kd_both_best.pth')
    
    if os.path.exists(both_checkpoint):
        print("\n" + "=" * 80)
        print("4. Evaluating Combined (Response + Feature) Knowledge Distillation")
        print("=" * 80)
        
        print(f"Loading combined model: {both_checkpoint}")
        both_model, both_ckpt = load_model(both_checkpoint, device, NUM_CLASSES)
        
        num_params = sum(p.numel() for p in both_model.parameters())
        
        print("\nMeasuring inference speed...")
        avg_time, std_time = measure_inference_speed(both_model, device)
        
        print("\nEvaluating accuracy...")
        miou = evaluate_model(both_model, val_loader, device, NUM_CLASSES)
        
        results.append({
            'method': 'Both (Response + Feature)',
            'miou': miou,
            'params': num_params,
            'inference_ms': avg_time,
            'inference_std': std_time
        })
        
        print(f"\nResults:")
        print(f"  mIoU: {miou:.4f}")
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  Inference: {avg_time:.2f} ± {std_time:.2f} ms/image")
        
        del both_model
        torch.cuda.empty_cache()
    
    # ===== Print Comparison Table =====
    if len(results) > 0:
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        print()
        print(f"{'Knowledge Distillation':<35} {'mIoU':<12} {'# Parameters':<18} {'Inference Speed (ms)':<25}")
        print("-" * 90)
        
        for result in results:
            method = result['method']
            miou = f"{result['miou']:.4f}"
            params = f"{result['params']/1e6:.2f}M"
            inference = f"{result['inference_ms']:.2f} ± {result['inference_std']:.2f}"
            
            print(f"{method:<35} {miou:<12} {params:<18} {inference:<25}")
        
        print()
        print("=" * 80)
        
        # Calculate improvements over baseline
        if results[0]['method'] == 'Without KD (Baseline)' and len(results) > 1:
            print("\nIMPROVEMENTS OVER BASELINE:")
            print("-" * 80)
            baseline_miou = results[0]['miou']
            
            for result in results[1:]:
                improvement = result['miou'] - baseline_miou
                improvement_pct = (improvement / baseline_miou) * 100
                print(f"  {result['method']:<35} {improvement:+.4f} ({improvement_pct:+.2f}%)")
            print()
        
        # Save results to file
        output_file = os.path.join(config.CHECKPOINT_DIR, 'kd_comparison_results.txt')
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Knowledge Distillation Methods Comparison\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"{'Knowledge Distillation':<35} {'mIoU':<12} {'# Parameters':<18} {'Inference Speed (ms)':<25}\n")
            f.write("-" * 90 + "\n")
            
            for result in results:
                method = result['method']
                miou = f"{result['miou']:.4f}"
                params = f"{result['params']/1e6:.2f}M"
                inference = f"{result['inference_ms']:.2f} ± {result['inference_std']:.2f}"
                
                f.write(f"{method:<35} {miou:<12} {params:<18} {inference:<25}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
            if results[0]['method'] == 'Without KD (Baseline)' and len(results) > 1:
                f.write("\nIMPROVEMENTS OVER BASELINE:\n")
                f.write("-" * 80 + "\n")
                baseline_miou = results[0]['miou']
                
                for result in results[1:]:
                    improvement = result['miou'] - baseline_miou
                    improvement_pct = (improvement / baseline_miou) * 100
                    f.write(f"  {result['method']:<35} {improvement:+.4f} ({improvement_pct:+.2f}%)\n")
        
        print(f"\nResults saved to: {output_file}")
        print("=" * 80)
    else:
        print("\nERROR: No models found to compare!")
        print("\nTo generate comparison data:")
        print("  1. Train baseline: python scripts/train.py")
        print("  2. Train response-based: python scripts/train_knowledge_distillation.py --method response")
        print("  3. Train feature-based: python scripts/train_knowledge_distillation.py --method feature")


if __name__ == '__main__':
    main()

