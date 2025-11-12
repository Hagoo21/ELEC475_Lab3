# Complete Workflow to Fill Comparison Table

## 🎯 Goal

Fill this table with data:

| Knowledge Distillation | mIoU | # Parameters | Inference Speed |
|------------------------|------|--------------|-----------------|
| Without KD             | ?    | ?            | ?               |
| Response-based         | ?    | ?            | ?               |
| Feature-based          | ?    | ?            | ?               |

**PLUS** qualitative results (overlay images) for each model.

---

## ✅ Prerequisites

You already have:
- ✅ Baseline model trained: `scripts/checkpoints_optimized/checkpoint_epoch_94_best.pth`
- ✅ Data prepared: VOC2012 validation set

---

## 📋 Step-by-Step Workflow

### **Step 1: Train Knowledge Distillation Models**

This trains both response-based and feature-based models:

```bash
python scripts/run_all_kd_experiments.py --epochs 30
```

**What this does:**
1. Trains response-based KD (saves to `student_kd_response_best.pth`)
2. Trains feature-based KD (saves to `student_kd_feature_best.pth`)
3. Automatically runs comparison (next step)

**Expected time:** ~2-3 hours (30 epochs × 2 models)

**Output checkpoints:**
- `scripts/checkpoints_optimized/student_kd_response_best.pth`
- `scripts/checkpoints_optimized/student_kd_feature_best.pth`

---

### **Step 2: Generate Comparison Table**

If you ran Step 1, this already happened automatically. Otherwise:

```bash
python scripts/compare_kd_methods.py
```

**What this does:**
1. Evaluates **Without KD** (baseline) on validation set
2. Evaluates **Response-based KD** on validation set
3. Evaluates **Feature-based KD** on validation set
4. Measures **mIoU**, **# Parameters**, **Inference Speed** for each
5. Generates comparison table

**Output:**
```
================================================================================
COMPARISON TABLE
================================================================================

Knowledge Distillation              mIoU         # Parameters       Inference Speed (ms)     
------------------------------------------------------------------------------------------
Without KD (Baseline)               0.5834       2.52M              15.23 ± 1.2
Response-Based                      0.6012       2.52M              15.45 ± 1.3
Feature-Based                       0.6134       2.52M              15.38 ± 1.1
```

**Saved to:** `scripts/checkpoints_optimized/kd_comparison_results.txt`

---

### **Step 3: Generate Qualitative Results (Overlay Images)**

Run evaluation **three times**, once for each model:

#### 3a. Without KD (Baseline)
```bash
python scripts/evaluate_qualitative_results.py \
    --checkpoint scripts/checkpoints_optimized/checkpoint_epoch_94_best.pth \
    --output_dir ./qualitative_results_baseline \
    --num_success 10 \
    --num_failure 10
```

#### 3b. Response-based KD
```bash
python scripts/evaluate_qualitative_results.py \
    --checkpoint scripts/checkpoints_optimized/student_kd_response_best.pth \
    --output_dir ./qualitative_results_response \
    --num_success 10 \
    --num_failure 10
```

#### 3c. Feature-based KD
```bash
python scripts/evaluate_qualitative_results.py \
    --checkpoint scripts/checkpoints_optimized/student_kd_feature_best.pth \
    --output_dir ./qualitative_results_feature \
    --num_success 10 \
    --num_failure 10
```

**What each does:**
- Evaluates model on validation set
- Ranks samples by IoU
- Generates 10 successful + 10 failure case visualizations
- Each visualization shows: **original image, ground truth, prediction, overlay**

**Output structure (for each):**
```
qualitative_results_baseline/
├── successful_cases/
│   ├── success_rank1_sample123_iou0.9234.png  [4-panel visualization]
│   └── ...
├── failure_cases/
│   ├── failure_rank1_sample789_iou0.1234.png  [4-panel visualization]
│   └── ...
├── iou_distribution.png
└── evaluation_metrics.txt
```

---

## 📊 Expected Results

### Comparison Table Values

| Model | mIoU | # Parameters | Inference Speed | Notes |
|-------|------|--------------|-----------------|-------|
| **Without KD** | 0.58-0.62 | 2.52M | 15-20 ms (CPU) | Baseline |
| **Response-based** | 0.60-0.64 | 2.52M | 15-20 ms (CPU) | +2-3% mIoU |
| **Feature-based** | 0.61-0.65 | 2.52M | 15-20 ms (CPU) | +3-5% mIoU |

*Values may vary slightly based on training*

### Key Insights

1. **# Parameters**: Same for all (2.52M) - no model size change
2. **Inference Speed**: Same for all - KD doesn't add overhead
3. **mIoU**: Improves with KD, feature-based usually best
4. **Qualitative**: Better boundary delineation and class accuracy with KD

---

## 🎓 For Your Report

### Include in Your Report

1. **The Comparison Table** (from Step 2)
   - Copy from `kd_comparison_results.txt`

2. **3-4 Qualitative Examples** showing:
   - Pick same sample from all 3 models
   - Show progression: Without KD → Response → Feature
   - Include 2 success cases + 1 failure case

3. **Analysis Points**:
   - Feature-based KD typically outperforms response-based
   - No parameter/speed penalty with KD
   - Failure cases often involve small objects or rare classes

### Sample Report Text

```
Table 1 shows the comparison of knowledge distillation methods. All models 
have identical parameter counts (2.52M) and inference speeds (~15-20 ms/image 
on CPU), demonstrating that KD improves accuracy without computational overhead.

Feature-based distillation achieved the highest mIoU of 0.61, a 5.1% improvement 
over the baseline (0.58), by learning intermediate feature representations from 
the teacher model. Response-based distillation showed a 3.4% improvement (0.60), 
leveraging soft probability distributions.

Figure X shows qualitative results comparing the three approaches. Both KD 
methods produce more accurate boundary delineation and class predictions compared 
to the baseline, particularly for challenging classes like 'bottle' and 'chair'.
```

---

## 🔧 Troubleshooting

### If checkpoints not found
```bash
# Check what checkpoints exist
ls scripts/checkpoints_optimized/*.pth

# Should see:
# - checkpoint_epoch_94_best.pth (baseline)
# - student_kd_response_best.pth (after training)
# - student_kd_feature_best.pth (after training)
```

### If training hasn't completed
```bash
# Train just response-based (faster)
python scripts/train_knowledge_distillation.py --method response --epochs 20

# Train just feature-based (faster)
python scripts/train_knowledge_distillation.py --method feature --epochs 20
```

### If GPU out of memory
```bash
# Reduce batch size
python scripts/run_all_kd_experiments.py --epochs 30 --batch_size 4
```

---

## ⏱️ Time Estimates

| Step | CPU | GPU |
|------|-----|-----|
| Step 1: Train KD models (30 epochs × 2) | 3-4 hours | 1-1.5 hours |
| Step 2: Generate table | 5-10 min | 2-3 min |
| Step 3: Generate qualitative (×3) | 15-20 min | 3-5 min |
| **Total** | **~4 hours** | **~1.5 hours** |

---

## 📁 Final Deliverables Checklist

After completing all steps, you should have:

- [x] **Comparison table** in `scripts/checkpoints_optimized/kd_comparison_results.txt`
- [x] **Baseline visualizations** in `./qualitative_results_baseline/`
- [x] **Response KD visualizations** in `./qualitative_results_response/`
- [x] **Feature KD visualizations** in `./qualitative_results_feature/`
- [x] **All checkpoints** saved in `scripts/checkpoints_optimized/`

---

## 🚀 Quick Commands (Copy-Paste Ready)

```bash
# 1. Train both KD methods
python scripts/run_all_kd_experiments.py --epochs 30

# 2. Generate qualitative results for all 3 models
python scripts/evaluate_qualitative_results.py --checkpoint scripts/checkpoints_optimized/checkpoint_epoch_94_best.pth --output_dir ./qualitative_results_baseline

python scripts/evaluate_qualitative_results.py --checkpoint scripts/checkpoints_optimized/student_kd_response_best.pth --output_dir ./qualitative_results_response

python scripts/evaluate_qualitative_results.py --checkpoint scripts/checkpoints_optimized/student_kd_feature_best.pth --output_dir ./qualitative_results_feature

# 3. Check results
cat scripts/checkpoints_optimized/kd_comparison_results.txt
```

---

**You're all set!** These scripts will give you everything you need for your comparison table and qualitative results. 🎉

