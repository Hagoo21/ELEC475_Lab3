# Knowledge Distillation - Quick Start Guide

## TL;DR - For Your Assignment

You need to train and compare **two separate KD methods**:

### 1. Train Response-Based KD
```bash
python scripts/train_knowledge_distillation.py --method response --epochs 30
```

### 2. Train Feature-Based KD
```bash
python scripts/train_knowledge_distillation.py --method feature --epochs 30
```

### 3. Generate Comparison Table
```bash
python scripts/compare_kd_methods.py
```

**Done!** You'll get a table showing mIoU, parameters, and inference speed for your report.

---

## OR: Run Everything Automatically

```bash
python scripts/run_all_kd_experiments.py --epochs 30 --skip-combined
```

This runs both experiments and generates the comparison table.

---

## What Each Method Does

| Method | What's Enabled | Formula |
|--------|---------------|---------|
| **Response-Based** | β=0.7, γ=0 | KL(student\|\|teacher) × T² |
| **Feature-Based** | β=0, γ=0.5 | 1 - cos(student_feat, teacher_feat) |

---

## Output Files

### Response-Based
- `checkpoints_optimized/student_kd_response_best.pth`
- `checkpoints_optimized/kd_training_history_response.pth`

### Feature-Based
- `checkpoints_optimized/student_kd_feature_best.pth`
- `checkpoints_optimized/kd_training_history_feature.pth`

### Comparison
- `checkpoints_optimized/kd_comparison_results.txt` ← **Use this for your report!**

---

## Visualize Results

```bash
# Response-based plots
python scripts/visualize_kd_training.py --method response

# Feature-based plots
python scripts/visualize_kd_training.py --method feature
```

---

## Expected Comparison Table

```
Knowledge Distillation          mIoU         # Parameters       Inference Speed
Without                         0.6245       2.81M              45.23 ms/image
Response-based                  0.6387       2.81M              45.67 ms/image
Feature-based                   0.6412       2.81M              45.89 ms/image
```

---

## Time Required

- **Each training:** 3-5 hours (30 epochs, GPU)
- **Total:** 6-10 hours for both methods
- **Comparison:** 5-10 minutes

---

## Troubleshooting

### Out of Memory?
```bash
python scripts/train_knowledge_distillation.py --method response --batch_size 4
```

### Baseline Not Found?
```bash
# Train baseline first
python scripts/train.py --epochs 30
```

---

## For Your Report

Copy the table from `kd_comparison_results.txt` into your report.

Add hardware info:
- GPU model
- Training time per method
- Inference speed from table

**That's it!** 🎉

See `KD_ASSIGNMENT_GUIDE.md` for detailed explanations.

