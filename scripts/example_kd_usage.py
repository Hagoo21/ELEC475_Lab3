"""
Minimal Example: Knowledge Distillation Usage

This script demonstrates the core components of knowledge distillation
in a simplified, easy-to-understand format.

Author: ELEC475 Lab 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Example 1: Response-Based Distillation
def response_based_distillation_example():
    """
    Demonstrates how KL divergence with temperature softens distributions
    to transfer 'dark knowledge' from teacher to student.
    """
    print("=" * 80)
    print("EXAMPLE 1: Response-Based Distillation")
    print("=" * 80)
    
    # Teacher produces confident predictions
    teacher_logits = torch.tensor([[10.0, 2.0, 1.0, 0.5]])
    
    # Student produces less confident predictions
    student_logits = torch.tensor([[6.0, 3.0, 2.0, 1.0]])
    
    # Ground truth: class 0
    target = torch.tensor([0])
    
    # Hyperparameters
    alpha = 1.0      # CE weight
    beta = 0.5       # KD weight
    T = 4.0          # Temperature
    
    print(f"\nTeacher logits: {teacher_logits[0].numpy()}")
    print(f"Student logits: {student_logits[0].numpy()}")
    print(f"Ground truth: class {target.item()}\n")
    
    # 1. Cross-Entropy Loss (supervised learning)
    ce_loss = F.cross_entropy(student_logits, target)
    print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")
    print("  → Ensures student learns correct predictions from ground truth")
    
    # 2. KL Divergence Loss (knowledge distillation)
    student_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
    print(f"\nKL Divergence Loss: {kd_loss.item():.4f}")
    print("  → Student learns from teacher's soft predictions")
    
    # 3. Total Loss
    total_loss = alpha * ce_loss + beta * kd_loss
    print(f"\nTotal Loss: {total_loss.item():.4f}")
    print(f"  = {alpha} × {ce_loss.item():.4f} + {beta} × {kd_loss.item():.4f}")
    
    # Show temperature effect
    print(f"\n{'='*80}")
    print("Temperature Effect:")
    print(f"{'='*80}")
    print(f"{'Temp':<8} {'Teach Prob[0]':<15} {'Teach Prob[1]':<15} {'Entropy':<10}")
    print("-" * 80)
    
    for temp in [1.0, 2.0, 4.0, 8.0]:
        soft = F.softmax(teacher_logits / temp, dim=1)
        entropy = -(soft * torch.log(soft + 1e-8)).sum().item()
        print(f"{temp:<8} {soft[0,0].item():<15.4f} {soft[0,1].item():<15.4f} {entropy:<10.4f}")
    
    print("\n✓ Higher temperature → softer distribution → more information transfer")
    print("✓ At T=4, student learns that class 1 is somewhat similar to class 0")
    print()


# Example 2: Feature-Based Distillation
def feature_based_distillation_example():
    """
    Demonstrates how cosine similarity matches intermediate features
    between student and teacher.
    """
    print("=" * 80)
    print("EXAMPLE 2: Feature-Based Distillation")
    print("=" * 80)
    
    # Simulate feature maps from student and teacher
    # Shape: [batch, channels, height, width]
    batch, channels, h, w = 2, 64, 16, 16
    
    teacher_features = torch.randn(batch, channels, h, w)
    
    # Case 1: Student features similar to teacher (good alignment)
    student_features_good = teacher_features + 0.1 * torch.randn_like(teacher_features)
    
    # Case 2: Student features different from teacher (poor alignment)
    student_features_bad = torch.randn(batch, channels, h, w)
    
    print(f"\nFeature map shape: {teacher_features.shape}")
    print(f"  Batch size: {batch}")
    print(f"  Channels: {channels}")
    print(f"  Spatial: {h}×{w}")
    
    # Flatten spatial dimensions for cosine similarity
    teacher_flat = teacher_features.flatten(2)  # [batch, channels, h*w]
    student_good_flat = student_features_good.flatten(2)
    student_bad_flat = student_features_bad.flatten(2)
    
    # Compute cosine similarity
    cos_sim_good = F.cosine_similarity(teacher_flat, student_good_flat, dim=1).mean()
    cos_sim_bad = F.cosine_similarity(teacher_flat, student_bad_flat, dim=1).mean()
    
    # Compute loss (1 - cosine_similarity)
    loss_good = 1 - cos_sim_good
    loss_bad = 1 - cos_sim_bad
    
    print(f"\nCase 1: Well-aligned student features")
    print(f"  Cosine similarity: {cos_sim_good.item():.4f}")
    print(f"  Feature loss: {loss_good.item():.4f}")
    
    print(f"\nCase 2: Poorly-aligned student features")
    print(f"  Cosine similarity: {cos_sim_bad.item():.4f}")
    print(f"  Feature loss: {loss_bad.item():.4f}")
    
    print(f"\n✓ Better alignment → higher cosine similarity → lower loss")
    print(f"✓ Loss reduced by {(loss_bad - loss_good).item():.4f} with good alignment")
    print()


# Example 3: Complete Mini Training Loop
def mini_training_loop_example():
    """
    Demonstrates a minimal training loop with knowledge distillation.
    """
    print("=" * 80)
    print("EXAMPLE 3: Mini Training Loop")
    print("=" * 80)
    
    # Hyperparameters
    alpha, beta, gamma, T = 1.0, 0.5, 0.3, 4.0
    lr = 0.01
    num_iterations = 5
    
    # Simulate student model (trainable)
    student_logits = nn.Parameter(torch.randn(1, 21, 4, 4))  # [B, C, H, W]
    student_features = nn.Parameter(torch.randn(1, 64, 8, 8))
    
    # Simulate teacher model (frozen)
    with torch.no_grad():
        teacher_logits = torch.randn(1, 21, 4, 4) * 2  # More confident
        teacher_features = torch.randn(1, 64, 8, 8)
    
    # Ground truth
    target = torch.randint(0, 21, (1, 4, 4))
    
    # Optimizer
    optimizer = torch.optim.SGD([student_logits, student_features], lr=lr)
    
    print(f"\nHyperparameters:")
    print(f"  α={alpha}, β={beta}, γ={gamma}, T={T}, lr={lr}")
    print(f"\nTraining for {num_iterations} iterations...\n")
    print(f"{'Iter':<6} {'CE Loss':<10} {'KD Loss':<10} {'Feat Loss':<10} {'Total':<10}")
    print("-" * 80)
    
    for iteration in range(1, num_iterations + 1):
        optimizer.zero_grad()
        
        # Compute losses
        # 1. Cross-entropy
        ce_loss = F.cross_entropy(student_logits, target, ignore_index=255)
        
        # 2. KL divergence
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        
        # 3. Feature cosine similarity
        s_flat = student_features.flatten(2)
        t_flat = teacher_features.flatten(2)
        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1).mean()
        feat_loss = 1 - cos_sim
        
        # Total loss
        total_loss = alpha * ce_loss + beta * kd_loss + gamma * feat_loss
        
        # Backward and update (only student parameters)
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        print(f"{iteration:<6} {ce_loss.item():<10.4f} {kd_loss.item():<10.4f} "
              f"{feat_loss.item():<10.4f} {total_loss.item():<10.4f}")
    
    print("\n✓ All losses should decrease over iterations")
    print("✓ Student learns from both ground truth (CE) and teacher (KD + Feat)")
    print()


# Example 4: Gradient Flow Verification
def gradient_flow_example():
    """
    Demonstrates that gradients flow only to student, not teacher.
    """
    print("=" * 80)
    print("EXAMPLE 4: Gradient Flow Verification")
    print("=" * 80)
    
    # Student (trainable)
    student_param = nn.Parameter(torch.randn(1, 10))
    
    # Teacher (frozen)
    teacher_param = torch.randn(1, 10)  # Not a Parameter, no gradients
    teacher_param.requires_grad = False
    
    # Compute loss involving both
    loss = ((student_param - teacher_param) ** 2).sum()
    
    print(f"\nBefore backward:")
    print(f"  Student grad: {student_param.grad}")
    print(f"  Teacher grad: {teacher_param.grad}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nAfter backward:")
    print(f"  Student grad: {student_param.grad is not None} (has gradients)")
    print(f"  Teacher grad: {teacher_param.grad is None} (no gradients)")
    
    print(f"\n✓ Gradients flow to student only")
    print(f"✓ Teacher remains frozen (no updates)")
    print(f"✓ This is critical for knowledge distillation")
    print()


if __name__ == "__main__":
    """
    Run all examples demonstrating knowledge distillation concepts.
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "KNOWLEDGE DISTILLATION EXAMPLES" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Run examples
    response_based_distillation_example()
    feature_based_distillation_example()
    mini_training_loop_example()
    gradient_flow_example()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Key Concepts:")
    print("  1. Response-based distillation: Student learns from teacher's soft outputs")
    print("  2. Feature-based distillation: Student matches teacher's internal features")
    print("  3. Temperature scaling: Softens distributions to transfer more knowledge")
    print("  4. Gradient flow: Only student updates, teacher stays frozen")
    print()
    print("Implementation:")
    print("  • L_total = α·L_CE + β·L_KD + γ·L_feat")
    print("  • L_CE: Cross-entropy with ground truth")
    print("  • L_KD: KL divergence with teacher (temperature-scaled)")
    print("  • L_feat: Cosine similarity with teacher features")
    print()
    print("For full training, run:")
    print("  → python scripts/test_knowledge_distillation.py  (verify implementation)")
    print("  → python scripts/train_knowledge_distillation.py  (train with KD)")
    print()
    print("For detailed documentation, see:")
    print("  → KNOWLEDGE_DISTILLATION_GUIDE.md")
    print("  → KNOWLEDGE_DISTILLATION_README.md")
    print()
    print("=" * 80)
    print()

