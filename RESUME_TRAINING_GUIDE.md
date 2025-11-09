# Resume Training Guide

## How to Resume Training

### Overview
The training script now supports resuming from `checkpoint_latest.pth`. This allows you to:
- Continue training after interruption
- Extend training beyond the original epoch count
- Resume with all optimizer/scheduler states preserved

---

## Usage Examples

### 1. Resume Training (Continue to 100 epochs)
If you trained for 50 epochs and want to continue to 100:

```bash
python train.py --resume --epochs 100
```

This will:
- Load model weights from epoch 50
- Load optimizer and scheduler states
- Load training history (all previous losses/metrics)
- Continue from epoch 51 to 100
- Preserve the best mIoU tracker

### 2. Resume with Different Settings
You can also change some parameters when resuming:

```bash
python train.py --resume --epochs 100 --lr 5e-5
```

**Note:** Changing learning rate will override the loaded optimizer state's LR.

### 3. Start Fresh Training (No Resume)
Just run without `--resume`:

```bash
python train.py --epochs 50
```

---

## What Gets Loaded from `checkpoint_latest.pth`

When `--resume` is used, the script loads:

✅ **Model weights** (`model_state_dict`)  
✅ **Optimizer state** (`optimizer_state_dict`)  
✅ **Scheduler state** (`scheduler_state_dict`)  
✅ **Training history** (losses, mIoU, pixel accuracy)  
✅ **Best mIoU** tracker  
✅ **Last completed epoch** number  

---

## Files Needed to Resume

**Minimum required:**
```
checkpoints/
└── checkpoint_latest.pth    ← Must exist in output_dir
```

**After you delete intermediate checkpoints, you only need:**
```
checkpoints/
├── checkpoint_latest.pth           ← For resuming training
└── checkpoint_epoch_49_best.pth    ← Your best model (backup)
```

---

## Common Scenarios

### Scenario 1: Training Interrupted at Epoch 30
```bash
# Resume and complete to epoch 50
python train.py --resume --epochs 50
```
→ Will continue from epoch 31 to 50

### Scenario 2: Want to Train Longer
```bash
# Trained to 50, want to go to 100
python train.py --resume --epochs 100
```
→ Will continue from epoch 51 to 100

### Scenario 3: Deleted All But Latest Checkpoint
```bash
# As long as checkpoint_latest.pth exists:
python train.py --resume --epochs 75
```
→ Works perfectly! Continues from where you left off

---

## Important Notes

⚠️ **Warning:** If you run `train.py` **without** `--resume`, it will:
- Start training from epoch 1
- Overwrite existing checkpoints
- Reset training history

✅ **Always use** `--resume` if you want to continue from where you left off!

---

## Verification

To check your current training status before resuming:

```bash
python check_best_model.py
```

This shows:
- Last completed epoch
- Best epoch and mIoU
- Which checkpoint files exist

---

## Examples

### Example 1: Standard Resume
```bash
# You trained for 50 epochs
# Now you want to train to 75 epochs total

python train.py --resume --epochs 75
```

**Output:**
```
========================================================================
Resuming from checkpoint
========================================================================
Loading checkpoint: checkpoints\checkpoint_latest.pth
  Loaded training history: 50 epochs
  Resuming from epoch: 51
  Best mIoU so far: 0.5205
  Current learning rate: 0.000001
========================================================================

========================================================================
Starting Training
========================================================================

========================================================================
Epoch 51/75
========================================================================
```

### Example 2: What Happens Without Resume
```bash
# Accidentally forgot --resume
python train.py --epochs 75
```

**Result:** Training starts from epoch 1 again! ❌

---

## Quick Reference

| Command | What it does |
|---------|-------------|
| `python train.py --epochs 50` | Start **new** training (epochs 1-50) |
| `python train.py --resume --epochs 75` | **Resume** from checkpoint (continue to epoch 75) |
| `python check_best_model.py` | Check current training status |
| `python visualize_training.py` | Visualize training curves |

