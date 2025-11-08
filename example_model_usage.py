"""
Example usage of the Lightweight Segmentation Model

This script demonstrates:
1. Loading the model
2. Processing an image
3. Getting predictions
4. Extracting intermediate features (for KD)
"""

import torch
import torch.nn.functional as F
from lightweight_segmentation_model import (
    LightweightSegmentationModel, 
    count_parameters,
    print_model_summary
)


def example_basic_inference():
    """Example: Basic inference without feature extraction"""
    print("=" * 80)
    print("Example 1: Basic Inference")
    print("=" * 80)
    
    # Create model
    model = LightweightSegmentationModel(num_classes=21, pretrained=True)
    model.eval()
    
    # Example input (batch_size=1, 3 channels, 512x512)
    input_image = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        logits = model(input_image)
    
    # Get class predictions
    predictions = torch.argmax(logits, dim=1)
    
    print(f"Input shape:       {input_image.shape}")
    print(f"Logits shape:      {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique classes predicted: {torch.unique(predictions).tolist()}")
    print()


def example_with_features():
    """Example: Inference with intermediate feature extraction (for KD)"""
    print("=" * 80)
    print("Example 2: Inference with Feature Extraction (for Knowledge Distillation)")
    print("=" * 80)
    
    # Create model with feature extraction enabled
    model = LightweightSegmentationModel(
        num_classes=21, 
        pretrained=True, 
        return_features=True
    )
    model.eval()
    
    # Example input
    input_image = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        logits, features = model(input_image)
    
    print(f"Input shape:  {input_image.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"\nIntermediate features for KD:")
    print(f"  Low-level (stride 4):  {features['low'].shape}")
    print(f"  Mid-level (stride 8):  {features['mid'].shape}")
    print(f"  High-level (stride 16): {features['high'].shape}")
    print()


def example_probabilistic_output():
    """Example: Get class probabilities using softmax"""
    print("=" * 80)
    print("Example 3: Probabilistic Output")
    print("=" * 80)
    
    model = LightweightSegmentationModel(num_classes=21, pretrained=True)
    model.eval()
    
    input_image = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        logits = model(input_image)
        probabilities = F.softmax(logits, dim=1)
    
    print(f"Logits shape:        {logits.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities sum per pixel: {probabilities.sum(dim=1)[0, 0, 0]:.6f} (should be 1.0)")
    print()


def example_different_input_sizes():
    """Example: Model works with different input sizes (fully convolutional)"""
    print("=" * 80)
    print("Example 4: Different Input Sizes")
    print("=" * 80)
    
    model = LightweightSegmentationModel(num_classes=21, pretrained=True)
    model.eval()
    
    test_sizes = [(256, 256), (512, 512), (384, 384), (640, 480)]
    
    print("Testing different input resolutions:")
    for h, w in test_sizes:
        input_image = torch.randn(1, 3, h, w)
        with torch.no_grad():
            logits = model(input_image)
        print(f"  Input {h}x{w} -> Output {logits.shape[2]}x{logits.shape[3]}")
    print()


def example_batch_processing():
    """Example: Process multiple images at once"""
    print("=" * 80)
    print("Example 5: Batch Processing")
    print("=" * 80)
    
    model = LightweightSegmentationModel(num_classes=21, pretrained=True)
    model.eval()
    
    batch_sizes = [1, 4, 8, 16]
    
    print("Testing different batch sizes:")
    for batch_size in batch_sizes:
        input_batch = torch.randn(batch_size, 3, 512, 512)
        with torch.no_grad():
            logits = model(input_batch)
        print(f"  Batch size {batch_size:2d} -> Output shape {logits.shape}")
    print()


def example_parameter_count():
    """Example: Count and compare parameters"""
    print("=" * 80)
    print("Example 6: Parameter Count Comparison")
    print("=" * 80)
    
    # Lightweight model
    lightweight_model = LightweightSegmentationModel(num_classes=21, pretrained=True)
    total_params, trainable_params = count_parameters(lightweight_model)
    
    print(f"Lightweight Model (MobileNetV3-Small + ASPP):")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size:           {total_params * 4 / (1024**2):.2f} MB")
    print()
    
    # For comparison, typical segmentation models:
    print("Comparison with typical segmentation models:")
    print("  FCN-ResNet50:    ~35M parameters")
    print("  DeepLabV3+:      ~40M parameters")
    print("  Our model:       ~6.8M parameters (5-6x smaller!)")
    print()


if __name__ == "__main__":
    print("\nLightweight Segmentation Model - Usage Examples")
    print("=" * 80)
    print()
    
    # Run all examples
    example_basic_inference()
    example_with_features()
    example_probabilistic_output()
    example_different_input_sizes()
    example_batch_processing()
    example_parameter_count()
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)

