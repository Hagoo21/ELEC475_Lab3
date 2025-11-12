r"""
Simple Path Configuration
==========================

SETUP FOR YOUR OTHER COMPUTER:
1. Edit DATA_ROOT below (line 22) to point to your dataset
2. Run: python config.py (to test)
3. Run: python train.py (to train)

COMMON PATHS:
  Standard:           DATA_ROOT = './data/VOC2012_train_val/VOC2012_train_val'
  Absolute (Windows): DATA_ROOT = r'C:\Users\20sr91\ELEC475_Lab3\data\VOC2012_train_val\VOC2012_train_val'
  Absolute (Linux):   DATA_ROOT = '/home/username/ELEC475_Lab3/data/VOC2012_train_val/VOC2012_train_val'

That's it!
"""

import os

# =============================================================================
# EDIT THIS LINE FOR YOUR COMPUTER
# =============================================================================

# Where is your VOC2012 dataset? Update this path:
DATA_ROOT = './data/VOC2012_train_val/VOC2012_train_val'

# Examples for different computers:
# DATA_ROOT = './data/VOC2012_train_val/VOC2012_train_val'  # Standard
# DATA_ROOT = './data/VOC2012_train_val'                     # One level up
# DATA_ROOT = r'C:\Users\20sr91\ELEC475_Lab3\data\VOC2012_train_val\VOC2012_train_val'  # Absolute path

# =============================================================================
# Other paths (you probably don't need to change these)
# =============================================================================

CHECKPOINT_DIR = './checkpoints_optimized'
LOG_DIR = './logs'
TEST_DATA_ROOT = './data/VOC2012_test/VOC2012_test'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# Validation
# =============================================================================

def validate_paths():
    """Check if DATA_ROOT is correct."""
    if not os.path.exists(DATA_ROOT):
        print(f"[ERROR] DATA_ROOT not found: {DATA_ROOT}")
        return False
    
    # Check for required files
    train_txt = os.path.join(DATA_ROOT, 'ImageSets', 'Segmentation', 'train.txt')
    if not os.path.exists(train_txt):
        print(f"[ERROR] train.txt not found in: {DATA_ROOT}")
        print(f"Expected: {train_txt}")
        return False
    
    return True


# Auto-detect if current path doesn't work
if not validate_paths():
    print("\n[WARNING] Trying to auto-detect correct path...")
    
    possible_paths = [
        './data/VOC2012_train_val/VOC2012_train_val',
        './data/VOC2012_train_val',
        './data',
    ]
    
    for path in possible_paths:
        train_txt = os.path.join(path, 'ImageSets', 'Segmentation', 'train.txt')
        if os.path.exists(train_txt):
            print(f"[FOUND] Correct path: {path}")
            DATA_ROOT = path
            break
    else:
        print("\n[ERROR] Could not find dataset!")
        print("Please update DATA_ROOT in config.py")


if __name__ == "__main__":
    print("\nChecking configuration...")
    print(f"DATA_ROOT: {DATA_ROOT}")
    
    if validate_paths():
        print("[SUCCESS] Configuration is correct!\n")
    else:
        print("[ERROR] Please fix DATA_ROOT in config.py\n")
