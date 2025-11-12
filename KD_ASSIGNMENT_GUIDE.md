# Knowledge Distillation Assignment Guide

## Overview

This guide explains how to run the separate knowledge distillation experiments required for your assignment.

## Assignment Requirements

Your assignment asks you to compare:
1. **Without KD** - Baseline student model
2. **Response-Based KD** - Using KL divergence on output logits only
3. **Feature-Based KD** - Using cosine similarity on intermediate features only

## Quick Start - Run All Experiments

### Option 1: Automatic (Recommended)

Run all experiments automatically:

```bash
# Run response-based and feature-based (recommended for assignment)
python scripts/run_all_kd_experiments.py --epochs 30 --skip-combined
```

This will:
1. Train response-based KD model
2. Train feature-based KD model
3. Generate comparison table automatically

**Estimated time:** 6-10 hours total (depends on GPU)

### Option 2: Manual (Step by Step)

If you prefer to run experiments manually or need more control:

#### Step 1: Train Response-Based KD

```bash
python scripts/train_knowledge_distillation.py --method response --epochs 30
```

This trains using:
- **α = 1.0** (Cross-entropy with ground truth)
- **β = 0.7** (KL divergence with teacher outputs) **[ENABLED]**
- **γ = 0.0** (Feature matching) **[DISABLED]**
- **T = 4.0** (Temperature)

**Output:**
- `checkpoints_optimized/student_kd_response_best.pth`
- `checkpoints_optimized/kd_training_history_response.pth`

#### Step 2: Train Feature-Based KD

```bash
python scripts/train_knowledge_distillation.py --method feature --epochs 30
```

This trains using:
- **α = 1.0** (Cross-entropy with ground truth)
- **β = 0.0** (KL divergence) **[DISABLED]**
- **γ = 0.5** (Feature cosine similarity) **[ENABLED]**

**Output:**
- `checkpoints_optimized/student_kd_feature_best.pth`
- `checkpoints_optimized/kd_training_history_feature.pth`

#### Step 3: Generate Comparison Table

```bash
python scripts/compare_kd_methods.py
```

This will:
- Load baseline model (trained without KD)
- Load response-based model
- Load feature-based model
- Measure mIoU for each
- Measure inference speed (ms/image) for each
- Generate comparison table

**Output:**
- Screen display with formatted table
- `checkpoints_optimized/kd_comparison_results.txt`

## Understanding the Methods

### Response-Based Distillation

**Mathematical Form:**
```
L_response = KL(softmax(student_logits/T) || softmax(teacher_logits/T)) × T²
```

**What it does:**
- Transfers "dark knowledge" from teacher's soft predictions
- Uses temperature T to soften probability distributions
- T² compensates for gradient magnitude reduction

**PyTorch Implementation:**
```python
# Temperature scaling
student_soft = F.log_softmax(student_logits / T, dim=1)
teacher_soft = F.softmax(teacher_logits / T, dim=1)

# KL divergence
kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
```

**When β > 0, γ = 0:**
- Only uses output-level distillation
- Student learns to mimic teacher's final predictions
- Fast to compute (no feature extraction needed)

### Feature-Based Distillation

**Mathematical Form:**
```
L_feature = mean(1 - cosine_similarity(student_features, teacher_features))
```

Where cosine similarity for each feature level:
```
cos_sim(A, B) = (A · B) / (||A|| × ||B||)
```

**What it does:**
- Matches intermediate representations at 3 levels:
  - **Low** (stride 4): Edges, textures
  - **Mid** (stride 8): Object parts
  - **High** (stride 16): Semantic features
- Encourages similar internal representations
- Scale-invariant (only direction matters, not magnitude)

**PyTorch Implementation:**
```python
feat_loss = 0.0
for level in ['low', 'mid', 'high']:
    s_feat = student_features[level]  # [B, C, H, W]
    t_feat = teacher_features[level]  # [B, C, H, W]
    
    # Flatten spatial dimensions
    s_flat = s_feat.flatten(2)  # [B, C, H*W]
    t_flat = t_feat.flatten(2).detach()
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)  # [B, H*W]
    
    # Loss (1 - similarity, range [0, 2])
    feat_loss += (1 - cos_sim).mean()

feat_loss = feat_loss / 3  # Average over levels
```

**When β = 0, γ > 0:**
- Only uses intermediate feature distillation
- Student learns to have similar internal representations
- More computationally expensive (feature extraction overhead)

### Combined Loss

**Total Loss (used in training):**
```
L_total = α × L_CE + β × L_KD + γ × L_feat

Where:
  L_CE   = CrossEntropy(student_logits, ground_truth)
  L_KD   = KL divergence loss (response-based)
  L_feat = Cosine similarity loss (feature-based)
```

## Expected Results Format

After running comparison, you'll get a table like:

```
Knowledge Distillation          mIoU         # Parameters       Inference Speed (ms)
------------------------------------------------------------------------------------------
Without KD (Baseline)           0.6245       2.81M              45.23 ± 1.52
Response-Based                  0.6387       2.81M              45.67 ± 1.48
Feature-Based                   0.6412       2.81M              45.89 ± 1.55

IMPROVEMENTS OVER BASELINE:
  Response-Based                  +0.0142 (+2.27%)
  Feature-Based                   +0.0167 (+2.67%)
```

### What Each Column Means

1. **Knowledge Distillation**: Method used
2. **mIoU**: Mean Intersection over Union (accuracy metric)
   - Higher is better
   - Measures segmentation quality
3. **# Parameters**: Model size in millions of parameters
   - Same for all (same student architecture)
   - Shows model efficiency
4. **Inference Speed**: Time per image in milliseconds
   - Lower is better
   - Measured on single image (batch size 1)
   - Includes ± standard deviation

## Hardware and Performance Notes

### Recommended Hardware

- **GPU:** NVIDIA GPU with 6GB+ VRAM (GTX 1080 Ti, RTX 2060+, or better)
- **RAM:** 16GB+ system RAM
- **Storage:** 10GB+ free space for dataset and checkpoints

### Training Time Estimates

Per epoch (with batch_size=8):
- **GTX 1080 Ti:** ~8-10 minutes
- **RTX 3060:** ~5-7 minutes
- **RTX 4090:** ~2-3 minutes

Full 30 epochs:
- **GTX 1080 Ti:** ~4-5 hours
- **RTX 3060:** ~2.5-3.5 hours
- **RTX 4090:** ~1-1.5 hours

**Total for both experiments:** 2× training time

### Testing (Inference) Speed

Typical inference speeds (single 512×512 image):
- **CPU (Intel i7):** ~200-300 ms/image
- **GTX 1080 Ti:** ~40-50 ms/image
- **RTX 3060:** ~25-35 ms/image
- **RTX 4090:** ~15-20 ms/image

**Note:** All KD methods have similar inference speed (same model architecture)

## Visualizing Results

### Visualize Training Curves

For response-based:
```bash
python scripts/visualize_kd_training.py --method response
```

For feature-based:
```bash
python scripts/visualize_kd_training.py --method feature
```

This generates 4 plots:
1. Loss components (Total, CE, KD, Feature)
2. Validation mIoU with baseline
3. Combined loss comparison
4. Learning rate schedule

**Output:** `visualizations/kd_*.png`

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size
```bash
python scripts/train_knowledge_distillation.py --method response --batch_size 4
```

### Issue: Training Very Slow

**Possible causes:**
1. No GPU available (using CPU)
   - Check: `torch.cuda.is_available()`
   - Fix: Install CUDA-enabled PyTorch
2. num_workers > 0 on Windows
   - Already set to 0 in code

### Issue: Baseline Checkpoint Not Found

The comparison script needs your baseline model (trained without KD).

**Solution:**
```bash
# Train baseline model first
python scripts/train.py --epochs 30
```

This creates `checkpoints_optimized/checkpoint_epoch_*_best.pth`

### Issue: mIoU Not Improving

**Possible solutions:**
1. Train longer: `--epochs 50`
2. Lower learning rate: `--lr 0.0005`
3. Check if dataset is loaded correctly
4. Verify baseline model exists

## File Organization

After running all experiments:

```
ELEC475_Lab3/
├── checkpoints_optimized/
│   ├── checkpoint_epoch_30_best.pth              # Baseline (without KD)
│   ├── student_kd_response_best.pth             # Response-based KD
│   ├── student_kd_feature_best.pth              # Feature-based KD
│   ├── kd_training_history_response.pth         # Response training history
│   ├── kd_training_history_feature.pth          # Feature training history
│   └── kd_comparison_results.txt                # Comparison table
├── visualizations/
│   ├── kd_loss_components.png                   # Loss plots
│   ├── kd_validation_miou.png                   # mIoU plots
│   └── ...
└── scripts/
    ├── train_knowledge_distillation.py          # Main training script
    ├── compare_kd_methods.py                    # Comparison script
    ├── run_all_kd_experiments.py                # Batch runner
    └── visualize_kd_training.py                 # Visualization
```

## For Your Report

Include the following in your lab report:

### 1. Hardware Description

Example:
```
Hardware Used:
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- CPU: Intel i7-11700K
- RAM: 32GB DDR4
- OS: Windows 11
```

### 2. Training Performance

```
Training Time (30 epochs):
- Response-Based: 3.2 hours
- Feature-Based: 3.5 hours (slightly slower due to feature extraction)

Training configuration:
- Batch size: 8
- Learning rate: 0.001
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
```

### 3. Testing Performance

Use the table from `compare_kd_methods.py` output:

```
Knowledge Distillation          mIoU         # Parameters       Inference Speed
Without                         0.6245       2.81M              45.23 ms/image
Response-based                  0.6387       2.81M              45.67 ms/image
Feature-based                   0.6412       2.81M              45.89 ms/image
```

### 4. Analysis

**Response-Based:**
- Improvement: +1.42% mIoU
- Method: Transfers soft predictions using KL divergence with T=4.0
- Pros: Fast, learns output relationships
- Cons: Doesn't capture internal representations

**Feature-Based:**
- Improvement: +1.67% mIoU
- Method: Matches intermediate features using cosine similarity
- Pros: Better internal representations, slightly higher accuracy
- Cons: Slower training, more complex

### 5. Comparison with Lecture Material

The implementations match the lecture formulas:

**Response-based (from lecture):**
- Uses KL divergence with temperature scaling
- Formula: `KL(P||Q) × T²`
- Implemented exactly as taught

**Feature-based (from lecture):**
- Uses cosine similarity loss
- Formula: `1 - cos(student_feat, teacher_feat)`
- Applied at multiple feature levels (low/mid/high)

## Advanced: Custom Hyperparameters

You can experiment with different settings:

```bash
# Response-based with higher temperature
python scripts/train_knowledge_distillation.py \
    --method response \
    --temperature 6.0 \
    --epochs 30

# Feature-based with different batch size
python scripts/train_knowledge_distillation.py \
    --method feature \
    --batch_size 16 \
    --epochs 30
```

## Summary Checklist

- [ ] Have baseline model trained (from Lab 3 Part 1)
- [ ] Run response-based KD training
- [ ] Run feature-based KD training
- [ ] Generate comparison table
- [ ] Visualize training curves
- [ ] Record hardware specs
- [ ] Measure training time
- [ ] Measure inference speed
- [ ] Analyze results
- [ ] Write report section

---

**Questions?** Check:
- `KNOWLEDGE_DISTILLATION_README.md` - Comprehensive KD guide
- `KNOWLEDGE_DISTILLATION_GUIDE.md` - Mathematical details
- Code comments in `train_knowledge_distillation.py`

**Good luck with your assignment!** 🎓

