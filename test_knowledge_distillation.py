"""
Test script for Knowledge Distillation Pipeline

Verifies that all components work correctly:
1. Teacher model loads and is frozen
2. Student model loads and is trainable
3. Loss computation works correctly
4. Feature extraction and alignment works
5. Gradient flow is correct (student gets gradients, teacher doesn't)

Run this before training to ensure everything is set up properly.

Author: ELEC475 Lab 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.lightweight_segmentation import LightweightSegmentationModel
from train_knowledge_distillation import (
    KnowledgeDistillationLoss, 
    TeacherModelWrapper
)


def test_model_initialization():
    """Test that both models initialize correctly."""
    print("=" * 80)
    print("TEST 1: Model Initialization")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Initialize student
    print("Initializing student model...")
    student = LightweightSegmentationModel(
        num_classes=21,
        pretrained=True,
        return_features=True
    ).to(device)
    
    student_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"✓ Student initialized with {student_params:,} trainable parameters")
    
    # Initialize teacher
    print("\nInitializing teacher model...")
    teacher = TeacherModelWrapper(num_classes=21).to(device)
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    teacher_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    
    print(f"✓ Teacher initialized with {teacher_params:,} total parameters")
    print(f"✓ Teacher trainable parameters: {teacher_trainable} (should be 0)")
    
    assert teacher_trainable == 0, "ERROR: Teacher should have 0 trainable parameters!"
    
    print(f"\n✓ Compression ratio: {teacher_params / student_params:.2f}x")
    print("\n[PASSED] Model initialization test")
    
    return student, teacher, device


def test_forward_pass(student, teacher, device):
    """Test forward pass through both models."""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass")
    print("=" * 80)
    
    # Create dummy input
    batch_size = 2
    image_size = 256  # Use smaller size for testing
    dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
    dummy_target = torch.randint(0, 21, (batch_size, image_size, image_size)).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Target shape: {dummy_target.shape}\n")
    
    # Forward pass through student
    print("Testing student forward pass...")
    student.eval()
    with torch.no_grad():
        student_logits, student_features = student(dummy_input)
    
    print(f"✓ Student logits shape: {student_logits.shape}")
    print(f"✓ Student features:")
    for level, feat in student_features.items():
        print(f"    {level}: {feat.shape}")
    
    # Forward pass through teacher
    print("\nTesting teacher forward pass...")
    teacher.eval()
    with torch.no_grad():
        teacher_logits, teacher_features = teacher(dummy_input)
    
    print(f"✓ Teacher logits shape: {teacher_logits.shape}")
    print(f"✓ Teacher features:")
    for level, feat in teacher_features.items():
        print(f"    {level}: {feat.shape}")
    
    # Check output shapes
    assert student_logits.shape[0] == batch_size, "Wrong batch size!"
    assert student_logits.shape[1] == 21, "Wrong number of classes!"
    assert teacher_logits.shape[1] == 21, "Wrong number of classes!"
    
    print("\n[PASSED] Forward pass test")
    
    return student_logits, student_features, teacher_logits, teacher_features, dummy_target


def test_loss_computation(student_logits, student_features, 
                         teacher_logits, teacher_features, targets, device):
    """Test loss computation."""
    print("\n" + "=" * 80)
    print("TEST 3: Loss Computation")
    print("=" * 80)
    
    # Create loss function
    criterion = KnowledgeDistillationLoss(
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
        temperature=4.0,
        num_classes=21
    ).to(device)
    
    print("Computing distillation loss...")
    total_loss, ce_loss, kd_loss, feat_loss = criterion(
        student_logits, teacher_logits,
        student_features, teacher_features,
        targets
    )
    
    print(f"\n✓ Total Loss:   {total_loss.item():.4f}")
    print(f"✓ CE Loss:      {ce_loss.item():.4f}")
    print(f"✓ KD Loss:      {kd_loss.item():.4f}")
    print(f"✓ Feature Loss: {feat_loss.item():.4f}")
    
    # Verify losses are reasonable
    assert not torch.isnan(total_loss), "Total loss is NaN!"
    assert not torch.isnan(ce_loss), "CE loss is NaN!"
    assert not torch.isnan(kd_loss), "KD loss is NaN!"
    assert not torch.isnan(feat_loss), "Feature loss is NaN!"
    
    assert total_loss.item() > 0, "Total loss should be positive!"
    assert ce_loss.item() > 0, "CE loss should be positive!"
    assert kd_loss.item() > 0, "KD loss should be positive!"
    
    # Verify weighted combination
    expected_total = (1.0 * ce_loss + 0.5 * kd_loss + 0.3 * feat_loss).item()
    assert abs(total_loss.item() - expected_total) < 1e-4, \
        f"Loss combination incorrect! Expected {expected_total:.4f}, got {total_loss.item():.4f}"
    
    print("\n[PASSED] Loss computation test")
    
    return criterion


def test_gradient_flow(student, teacher, criterion, device):
    """Test that gradients flow correctly."""
    print("\n" + "=" * 80)
    print("TEST 4: Gradient Flow")
    print("=" * 80)
    
    student.train()
    teacher.eval()
    
    # Create dummy data
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    dummy_target = torch.randint(0, 21, (2, 256, 256)).to(device)
    
    print("Performing forward and backward pass...")
    
    # Forward pass
    student_logits, student_features = student(dummy_input)
    
    with torch.no_grad():
        teacher_logits, teacher_features = teacher(dummy_input)
    
    # Compute loss
    total_loss, ce_loss, kd_loss, feat_loss = criterion(
        student_logits, teacher_logits,
        student_features, teacher_features,
        dummy_target
    )
    
    # Backward pass
    total_loss.backward()
    
    # Check student gradients
    print("\nChecking student gradients...")
    student_has_grad = False
    for name, param in student.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                student_has_grad = True
                # Check gradient is not zero
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-6:  # Non-trivial gradient
                    print(f"✓ {name}: grad_norm = {grad_norm:.6f}")
                    break
    
    assert student_has_grad, "Student should have gradients!"
    print("✓ Student receives gradients correctly")
    
    # Check teacher gradients
    print("\nChecking teacher gradients...")
    teacher_has_grad = False
    for name, param in teacher.named_parameters():
        if param.grad is not None:
            teacher_has_grad = True
            print(f"✗ {name} has gradient (should be None!)")
            break
    
    assert not teacher_has_grad, "Teacher should NOT have gradients!"
    print("✓ Teacher is frozen (no gradients)")
    
    print("\n[PASSED] Gradient flow test")


def test_feature_alignment(student_features, teacher_features, device):
    """Test feature alignment mechanism."""
    print("\n" + "=" * 80)
    print("TEST 5: Feature Alignment")
    print("=" * 80)
    
    print("\nStudent vs Teacher feature shapes:")
    for level in ['low', 'mid', 'high']:
        s_shape = student_features[level].shape
        t_shape = teacher_features[level].shape
        print(f"{level:>5}: Student {s_shape} vs Teacher {t_shape}")
        
        # Test interpolation for spatial alignment
        if s_shape[2:] != t_shape[2:]:
            print(f"        → Spatial dimensions differ, interpolation needed")
            t_aligned = F.interpolate(teacher_features[level], 
                                     size=s_shape[2:],
                                     mode='bilinear', 
                                     align_corners=False)
            print(f"        → After interpolation: {t_aligned.shape}")
            assert t_aligned.shape[2:] == s_shape[2:], "Spatial alignment failed!"
        
        # Test channel alignment
        if s_shape[1] != t_shape[1]:
            print(f"        → Channel dimensions differ, projection needed")
            align_layer = nn.Conv2d(t_shape[1], s_shape[1], 
                                   kernel_size=1, bias=False).to(device)
            # First align spatially if needed
            t_feat = teacher_features[level]
            if s_shape[2:] != t_shape[2:]:
                t_feat = F.interpolate(t_feat, size=s_shape[2:],
                                      mode='bilinear', align_corners=False)
            t_aligned = align_layer(t_feat)
            print(f"        → After projection: {t_aligned.shape}")
            assert t_aligned.shape == s_shape, "Channel alignment failed!"
    
    print("\n✓ Feature alignment mechanisms work correctly")
    print("\n[PASSED] Feature alignment test")


def test_temperature_scaling(device):
    """Test temperature scaling behavior."""
    print("\n" + "=" * 80)
    print("TEST 6: Temperature Scaling")
    print("=" * 80)
    
    # Create dummy logits (confident prediction)
    logits = torch.tensor([[10.0, 1.0, 0.5, 0.3]]).to(device)
    
    print("Original logits:", logits[0].cpu().numpy())
    
    # Test different temperatures
    temperatures = [1.0, 2.0, 4.0, 8.0]
    print("\nSoftmax with different temperatures:")
    print(f"{'T':<5} {'Prob 0':<10} {'Prob 1':<10} {'Entropy':<10}")
    print("-" * 50)
    
    for T in temperatures:
        soft = F.softmax(logits / T, dim=1)
        entropy = -(soft * torch.log(soft + 1e-8)).sum().item()
        probs = soft[0].cpu().numpy()
        print(f"{T:<5} {probs[0]:<10.4f} {probs[1]:<10.4f} {entropy:<10.4f}")
    
    print("\n✓ Higher temperature → softer distribution → higher entropy")
    print("✓ This transfers more 'dark knowledge' about class relationships")
    print("\n[PASSED] Temperature scaling test")


def test_kl_divergence(device):
    """Test KL divergence computation."""
    print("\n" + "=" * 80)
    print("TEST 7: KL Divergence Loss")
    print("=" * 80)
    
    # Create teacher (confident) and student (less confident) distributions
    teacher_logits = torch.tensor([[10.0, 1.0, 0.5]]).to(device)
    student_logits = torch.tensor([[5.0, 2.0, 1.0]]).to(device)
    
    T = 4.0
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    # Compute KL divergence with temperature
    student_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    
    kl_loss = kl_loss_fn(student_soft, teacher_soft) * (T * T)
    
    print(f"Teacher distribution: {teacher_soft[0].cpu().numpy()}")
    print(f"Student distribution: {torch.exp(student_soft)[0].cpu().numpy()}")
    print(f"\nKL Divergence (T={T}): {kl_loss.item():.6f}")
    
    # Test that identical distributions give 0 loss
    identical_loss = kl_loss_fn(student_soft, torch.exp(student_soft)) * (T * T)
    print(f"KL Divergence (identical): {identical_loss.item():.6f} (should be ~0)")
    
    assert identical_loss.item() < 1e-4, "Identical distributions should give ~0 KL loss"
    
    print("\n✓ KL divergence computation is correct")
    print("\n[PASSED] KL divergence test")


def test_cosine_similarity(device):
    """Test cosine similarity loss."""
    print("\n" + "=" * 80)
    print("TEST 8: Cosine Similarity Loss")
    print("=" * 80)
    
    # Create similar and dissimilar feature maps
    batch_size, channels, height, width = 2, 64, 16, 16
    
    feat1 = torch.randn(batch_size, channels, height, width).to(device)
    feat2_similar = feat1 + 0.1 * torch.randn_like(feat1)  # Similar
    feat2_different = torch.randn_like(feat1)  # Different
    
    # Flatten spatial dimensions
    feat1_flat = feat1.flatten(2)
    feat2_similar_flat = feat2_similar.flatten(2)
    feat2_different_flat = feat2_different.flatten(2)
    
    # Compute cosine similarity
    cos_sim_similar = F.cosine_similarity(feat1_flat, feat2_similar_flat, dim=1).mean()
    cos_sim_different = F.cosine_similarity(feat1_flat, feat2_different_flat, dim=1).mean()
    
    # Compute loss (1 - cosine_similarity)
    loss_similar = (1 - cos_sim_similar).item()
    loss_different = (1 - cos_sim_different).item()
    
    print(f"Cosine similarity (similar features):   {cos_sim_similar.item():.4f}")
    print(f"Cosine similarity (different features): {cos_sim_different.item():.4f}")
    print(f"\nLoss (similar):   {loss_similar:.4f}")
    print(f"Loss (different): {loss_different:.4f}")
    
    assert loss_similar < loss_different, "Similar features should have lower loss!"
    
    print("\n✓ Similar features → higher cosine similarity → lower loss")
    print("✓ Different features → lower cosine similarity → higher loss")
    print("\n[PASSED] Cosine similarity test")


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "KNOWLEDGE DISTILLATION PIPELINE TESTS" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        # Test 1: Model initialization
        student, teacher, device = test_model_initialization()
        
        # Test 2: Forward pass
        student_logits, student_features, teacher_logits, teacher_features, targets = \
            test_forward_pass(student, teacher, device)
        
        # Test 3: Loss computation
        criterion = test_loss_computation(
            student_logits, student_features,
            teacher_logits, teacher_features,
            targets, device
        )
        
        # Test 4: Gradient flow
        test_gradient_flow(student, teacher, criterion, device)
        
        # Test 5: Feature alignment
        test_feature_alignment(student_features, teacher_features, device)
        
        # Test 6: Temperature scaling
        test_temperature_scaling(device)
        
        # Test 7: KL divergence
        test_kl_divergence(device)
        
        # Test 8: Cosine similarity
        test_cosine_similarity(device)
        
        # All tests passed
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 28 + "ALL TESTS PASSED!" + " " * 33 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        print("✓ Knowledge distillation pipeline is working correctly!")
        print("✓ You can now run: python train_knowledge_distillation.py")
        print()
        
    except AssertionError as e:
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 32 + "TEST FAILED!" + " " * 33 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        print(f"Error: {e}")
        raise
    
    except Exception as e:
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 32 + "TEST FAILED!" + " " * 33 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()

