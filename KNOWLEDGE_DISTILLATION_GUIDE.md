# Knowledge Distillation Training Guide

## Overview

This guide explains the knowledge distillation (KD) pipeline implemented for training a lightweight semantic segmentation model using knowledge transfer from a larger teacher model.

## Architecture

### Teacher Model
- **Model**: FCN-ResNet50
- **Pretrained**: COCO + PASCAL VOC labels
- **Parameters**: ~35M (frozen during training)
- **Role**: Provides soft targets and intermediate features

### Student Model
- **Model**: LightweightSegmentationModel (MobileNetV3-Small backbone)
- **Parameters**: ~2.8M (trainable)
- **Role**: Learns from both ground truth labels and teacher's knowledge

## Knowledge Distillation Components

### 1. Response-Based Distillation

Transfers knowledge from the teacher's output probability distributions to the student.

**Mathematical Formulation:**

```
L_KD = KL(softmax(student_logits/T) || softmax(teacher_logits/T)) × T²
```

Where:
- `T` = Temperature parameter (default: 4.0) - softens probability distributions
- `T²` = Compensates for gradient magnitude reduction due to temperature scaling
- `KL` = Kullback-Leibler divergence measuring distribution similarity

**Intuition:**
- Temperature `T` makes the probability distribution "softer" (less peaked)
- Soft distributions carry more information about class relationships
- Example: If teacher is 90% confident about "cat" and 8% about "dog", this "dark knowledge" tells the student that cats and dogs are similar

**Gradient Flow:**
```
∂L_KD/∂θ_student ≠ 0  (student learns)
∂L_KD/∂θ_teacher = 0  (teacher frozen)
```

### 2. Feature-Based Distillation

Matches intermediate feature representations between student and teacher at multiple levels.

**Mathematical Formulation:**

```
L_feat = mean(1 - cosine_similarity(student_features, teacher_features))
```

Applied across three feature levels:
- **Low-level** (stride 4): Captures edges, textures, basic patterns
- **Mid-level** (stride 8): Captures object parts, local structures
- **High-level** (stride 16): Captures semantic information, global context

**Cosine Similarity:**
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Where:
- Range: [-1, 1], where 1 = perfectly aligned features
- Loss = 1 - cos(θ), minimizing distance between feature representations

**Intuition:**
- Forces student to learn similar internal representations as teacher
- Helps student develop better feature hierarchies
- Low-level features help with fine details, high-level with semantics

### 3. Supervised Learning (Cross-Entropy)

Standard supervised learning with ground truth labels.

**Mathematical Formulation:**

```
L_CE = CrossEntropy(student_logits, ground_truth)
```

**Role:**
- Ensures student learns correct predictions from labeled data
- Prevents student from just mimicking teacher's mistakes

## Total Loss Function

The complete training objective combines all three components:

```
L_total = α × L_CE + β × L_KD + γ × L_feat
```

**Default Hyperparameters:**
- `α = 1.0` - Weight for cross-entropy loss (ground truth supervision)
- `β = 0.5` - Weight for KL divergence loss (response-based distillation)
- `γ = 0.3` - Weight for feature cosine loss (feature-based distillation)
- `T = 4.0` - Temperature for softening distributions

**Intuition:**
- `α` controls how much the student follows ground truth
- `β` controls how much the student imitates teacher's output behavior
- `γ` controls how much the student matches teacher's internal features
- Balance is key: too much distillation can constrain student, too little wastes teacher knowledge

## Gradient Flow Analysis

### Student Model (Trainable)
```
θ_student ← θ_student - η × ∂L_total/∂θ_student
```

Receives gradients from:
1. ✓ Cross-entropy loss (supervised learning)
2. ✓ KL divergence loss (output distillation)
3. ✓ Cosine similarity loss (feature distillation)

### Teacher Model (Frozen)
```
∂L_total/∂θ_teacher = 0  (no updates)
```

Teacher is wrapped in `@torch.no_grad()` and `requires_grad=False`:
- No gradient computation → faster training
- No memory overhead for teacher gradients
- Teacher provides consistent knowledge throughout training

## Usage

### Basic Training

```bash
python scripts/train_knowledge_distillation.py
```

### Configuration

Edit hyperparameters in `train_knowledge_distillation.py`:

```python
# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

# Distillation parameters
ALPHA = 1.0        # CE loss weight
BETA = 0.5         # KD loss weight
GAMMA = 0.3        # Feature loss weight
TEMPERATURE = 4.0  # Softening temperature
```

### Tuning Tips

**Increase α (CE weight)** if:
- Student is learning wrong predictions from teacher
- Validation accuracy is low
- You have high-quality ground truth labels

**Increase β (KD weight)** if:
- You want student to closely mimic teacher
- Teacher is significantly better than student
- You have a good teacher model

**Increase γ (feature weight)** if:
- You want better feature representations
- Student struggles with fine details
- Feature maps show significant differences

**Increase T (temperature)** if:
- Distributions are too peaked (hard predictions)
- Want to transfer more "dark knowledge"
- Typical range: 1-10, start with 4

**Decrease T (temperature)** if:
- Want sharper, more confident predictions
- Teacher gives too soft predictions

## Expected Results

### Performance Metrics

| Metric | Baseline (No KD) | With KD | Improvement |
|--------|------------------|---------|-------------|
| mIoU | ~58% | ~60-62% | +2-4% |
| Parameters | 2.8M | 2.8M | Same |
| Inference Speed | Fast | Fast | Same |

### Training Progress

You should see:
1. **Epoch 1-5**: Rapid improvement as student learns from teacher
2. **Epoch 5-15**: Steady improvement, losses stabilize
3. **Epoch 15-30**: Fine-tuning, minor improvements

**Loss Components Over Time:**
- `CE Loss`: Should decrease steadily (learning correct predictions)
- `KD Loss`: Should decrease (matching teacher distributions)
- `Feat Loss`: Should decrease (aligning feature representations)

### Interpreting Loss Values

**Good training:**
```
Epoch 10/30
  CE Loss:    0.45  (decreasing)
  KD Loss:    0.25  (decreasing)
  Feat Loss:  0.30  (decreasing)
  Val mIoU:   0.60  (increasing)
```

**Potential issues:**
```
Epoch 10/30
  CE Loss:    1.20  (high - check learning rate)
  KD Loss:    0.05  (too low - teacher too similar, increase β)
  Feat Loss:  0.95  (high - features not aligning, check alignment layers)
  Val mIoU:   0.45  (low - check data quality)
```

## Output Files

After training completes:

### Checkpoint File
- **Location**: `checkpoints/student_kd_best.pth`
- **Contains**:
  - Model weights (state_dict)
  - Optimizer state
  - Best validation mIoU
  - Baseline mIoU (for comparison)
  - Distillation hyperparameters

### Loading Trained Model

```python
import torch
from models.lightweight_segmentation import LightweightSegmentationModel

# Create model
model = LightweightSegmentationModel(num_classes=21, pretrained=False)

# Load trained weights
checkpoint = torch.load('checkpoints/student_kd_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Check performance
print(f"Baseline mIoU: {checkpoint['baseline_miou']:.4f}")
print(f"KD mIoU: {checkpoint['miou']:.4f}")
print(f"Improvement: +{checkpoint['miou'] - checkpoint['baseline_miou']:.4f}")
```

## Advanced Topics

### Why Temperature Scaling?

Without temperature (`T=1`):
```
Teacher: [0.9, 0.08, 0.01, 0.01]  # Very confident
Student: [0.85, 0.10, 0.03, 0.02]
```
→ Student learns mostly from the dominant class (limited information)

With temperature (`T=4`):
```
Teacher: [0.4, 0.35, 0.15, 0.10]  # Softer distribution
Student: [0.38, 0.32, 0.18, 0.12]
```
→ Student learns relationships between classes (richer information)

### Why T² in the Loss?

The gradient of softmax with temperature is:

```
∂(softmax(z/T))/∂z ∝ 1/T
```

Multiplying by `T²` compensates for this:
- Keeps gradient magnitudes consistent
- Prevents KD loss from dominating when T is large
- Ensures balanced contribution to total loss

### Feature Alignment

Student and teacher may have different feature dimensions:
- **Spatial**: Resize teacher features to match student using bilinear interpolation
- **Channel**: Project teacher features using 1×1 convolution (learned on-the-fly)

This ensures cosine similarity can be computed correctly.

### Channel Alignment Layers

Automatically created when needed:

```python
if s_feat.shape[1] != t_feat.shape[1]:
    # Create 1x1 conv to align channels
    align_layer = nn.Conv2d(teacher_channels, student_channels, 
                           kernel_size=1, bias=False)
    t_feat = align_layer(t_feat)
```

These alignment layers are trainable and help the student learn how to map its features to teacher-like representations.

## Troubleshooting

### Low mIoU (<50%)
- Check if dataset path is correct
- Verify data augmentation isn't too aggressive
- Try increasing learning rate or training epochs
- Check if teacher model loads correctly

### KD Loss Not Decreasing
- Temperature might be too high/low - try T=3 or T=5
- Beta might be too small - try β=1.0
- Teacher and student outputs might not be aligned - check interpolation

### Feature Loss High (>0.8)
- Feature dimensions might not match - check alignment layers
- Gamma might be too high - try γ=0.1
- Feature extraction points might be wrong - verify model architecture

### Out of Memory
- Reduce batch size (try 4 or 2)
- Use mixed precision training (FP16)
- Reduce image size (try 384 instead of 512)

## References

### Key Papers
1. **Distilling the Knowledge in a Neural Network** (Hinton et al., 2015)
   - Original knowledge distillation paper
   - Introduced temperature scaling and soft targets

2. **FitNets: Hints for Thin Deep Nets** (Romero et al., 2015)
   - Feature-based distillation
   - Matching intermediate representations

3. **DeepLabV3+** (Chen et al., 2018)
   - ASPP module for semantic segmentation
   - Encoder-decoder architecture

### Implementation Notes
- Teacher model uses `@torch.no_grad()` for efficiency
- Student returns features via `return_features=True` flag
- Cosine similarity computed on flattened spatial dimensions
- Confusion matrix accumulates over entire validation set for accurate mIoU

## Summary

Knowledge distillation enables the lightweight student model (2.8M params) to achieve performance close to the large teacher model (35M params) through:

1. **Learning soft targets**: Captures class relationships via temperature-scaled distributions
2. **Matching features**: Develops similar internal representations across multiple levels
3. **Supervised learning**: Maintains accuracy on ground truth labels

The result is a compact model suitable for edge deployment with minimal performance loss compared to the large teacher.

---

**Author**: ELEC475 Lab 3  
**Last Updated**: November 2025

