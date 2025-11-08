"""
ELEC475 Lab 3 - Step 1: Test Pretrained FCN-ResNet50
====================================================
This script loads the pretrained FCN-ResNet50 model and evaluates it on
the PASCAL VOC 2012 validation dataset, computing mIoU and visualizing results.
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class VOC2012SegmentationDataset(Dataset):
    """
    Custom Dataset for PASCAL VOC 2012 Segmentation
    
    The dataset structure is:
    - Images: data/VOC2012_train_val/VOC2012_train_val/JPEGImages/
    - Masks: data/VOC2012_train_val/VOC2012_train_val/SegmentationClass/
    - Split files: data/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/
    """
    
    def __init__(self, root_dir, split='val', transform=None, target_transform=None):
        """
        Args:
            root_dir: Path to VOC2012_train_val/VOC2012_train_val/
            split: 'train', 'val', or 'trainval'
            transform: Transforms for input images
            target_transform: Transforms for segmentation masks
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Read the image IDs from the split file
        split_file = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.masks_dir = os.path.join(root_dir, 'SegmentationClass')
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # Load segmentation mask
        mask_path = os.path.join(self.masks_dir, f'{img_id}.png')
        mask = Image.open(mask_path)
        
        # Store original for visualization (resize to reasonable size and convert to numpy)
        original_image = image.copy()
        original_image = original_image.resize((520, 520))
        original_image = np.array(original_image)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        
        return image, mask, original_image


def compute_miou(predictions, targets, num_classes=21, ignore_index=255):
    """
    Compute mean Intersection over Union (mIoU)
    
    Args:
        predictions: Predicted class labels (H, W) or (B, H, W)
        targets: Ground truth labels (H, W) or (B, H, W)
        num_classes: Number of classes (21 for PASCAL VOC)
        ignore_index: Index to ignore (255 = boundary/void in VOC)
    
    Returns:
        miou: Mean IoU across all classes
        iou_per_class: IoU for each class
    
    The mIoU formula:
        IoU_c = (TP_c) / (TP_c + FP_c + FN_c)
        mIoU = (1/C) * Σ IoU_c for all classes c
    
    where:
        TP_c = True Positives for class c
        FP_c = False Positives for class c
        FN_c = False Negatives for class c
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Remove ignore_index pixels
    valid_mask = targets != ignore_index
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    iou_per_class = []
    
    for cls in range(num_classes):
        # True Positives: predicted as cls AND actually cls
        tp = torch.sum((predictions == cls) & (targets == cls)).float()
        
        # False Positives: predicted as cls but NOT actually cls
        fp = torch.sum((predictions == cls) & (targets != cls)).float()
        
        # False Negatives: NOT predicted as cls but actually cls
        fn = torch.sum((predictions != cls) & (targets == cls)).float()
        
        # IoU = TP / (TP + FP + FN)
        denominator = tp + fp + fn
        if denominator > 0:
            iou = tp / denominator
            iou_per_class.append(iou.item())
        else:
            # Class not present in ground truth or prediction
            iou_per_class.append(float('nan'))
    
    # Compute mean IoU (excluding NaN values for classes not present)
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    return miou, iou_per_class


def visualize_predictions(images, masks, predictions, num_samples=4):
    """
    Display original images alongside predicted segmentation masks
    
    Args:
        images: Numpy array of images (B, H, W, C)
        masks: Ground truth masks (B, H, W) tensor
        predictions: Predicted masks (B, H, W) tensor
        num_samples: Number of samples to display
    """
    # VOC colormap for visualization
    def get_voc_colormap():
        """Get the PASCAL VOC colormap for visualization"""
        colormap = torch.zeros(256, 3, dtype=torch.uint8)
        
        # Define colors for VOC classes (first 21)
        colors = [
            [0, 0, 0],        # 0: background
            [128, 0, 0],      # 1: aeroplane
            [0, 128, 0],      # 2: bicycle
            [128, 128, 0],    # 3: bird
            [0, 0, 128],      # 4: boat
            [128, 0, 128],    # 5: bottle
            [0, 128, 128],    # 6: bus
            [128, 128, 128],  # 7: car
            [64, 0, 0],       # 8: cat
            [192, 0, 0],      # 9: chair
            [64, 128, 0],     # 10: cow
            [192, 128, 0],    # 11: dining table
            [64, 0, 128],     # 12: dog
            [192, 0, 128],    # 13: horse
            [64, 128, 128],   # 14: motorbike
            [192, 128, 128],  # 15: person
            [0, 64, 0],       # 16: potted plant
            [128, 64, 0],     # 17: sheep
            [0, 192, 0],      # 18: sofa
            [128, 192, 0],    # 19: train
            [0, 64, 128],     # 20: tv/monitor
        ]
        
        for i, color in enumerate(colors):
            colormap[i] = torch.tensor(color, dtype=torch.uint8)
        
        return colormap
    
    colormap = get_voc_colormap()
    
    num_samples = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images[i].astype(np.uint8))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask (colorized)
        gt_mask = masks[i].cpu().numpy().astype(np.int64)
        gt_mask_colored = colormap[gt_mask].numpy()
        axes[i, 1].imshow(gt_mask_colored)
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        # Predicted mask (colorized)
        pred_mask = predictions[i].cpu().numpy().astype(np.int64)
        pred_mask_colored = colormap[pred_mask].numpy()
        axes[i, 2].imshow(pred_mask_colored)
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('step1_predictions.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to 'step1_predictions.png'")
    plt.show()


def main():
    """Main function to test pretrained FCN-ResNet50"""
    
    print("=" * 70)
    print("ELEC475 Lab 3 - Step 1: Testing Pretrained FCN-ResNet50")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # =========================================================================
    # 1. Load Pretrained Model
    # =========================================================================
    print("\n[1/5] Loading pretrained FCN-ResNet50 model...")
    
    # Load FCN-ResNet50 with COCO + VOC pretrained weights
    # Model outputs 21 classes for VOC (background + 20 object classes)
    model = torchvision.models.segmentation.fcn_resnet50(
        weights=torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    )
    model = model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout, batch norm in eval mode)
    
    print(f"✓ Model loaded successfully")
    print(f"  Output classes: 21 (VOC)")
    
    # =========================================================================
    # 2. Define Transforms and Load Dataset
    # =========================================================================
    print("\n[2/5] Setting up dataset and transforms...")
    
    # ImageNet normalization (the model was pretrained with these stats)
    # These values are the mean and std of ImageNet dataset
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # RGB means
        std=[0.229, 0.224, 0.225]     # RGB stds
    )
    
    # Transforms for input images
    # - Resize to 520x520 (common for FCN)
    # - Convert to tensor [C, H, W] with values in [0, 1]
    # - Normalize using ImageNet statistics
    image_transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Transforms for segmentation masks
    # - Resize to match image size
    # - Convert to tensor (values are class indices 0-20, 255 for ignore)
    mask_transform = transforms.Compose([
        transforms.Resize((520, 520), interpolation=Image.NEAREST),  # Use NEAREST to preserve class labels
        transforms.PILToTensor(),
    ])
    
    # Create dataset
    voc_root = './data/VOC2012_train_val/VOC2012_train_val'
    
    if not os.path.exists(voc_root):
        print(f"ERROR: Dataset not found at {voc_root}")
        print("Please ensure the VOC2012 dataset is downloaded and extracted correctly.")
        return
    
    val_dataset = VOC2012SegmentationDataset(
        root_dir=voc_root,
        split='val',
        transform=image_transform,
        target_transform=mask_transform
    )
    
    print(f"✓ Dataset loaded")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create DataLoader
    # We'll use a small batch for visualization
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # =========================================================================
    # 3. Run Inference on Validation Set
    # =========================================================================
    print("\n[3/5] Running inference on validation set...")
    
    all_predictions = []
    all_targets = []
    sample_images = []
    sample_masks = []
    sample_predictions = []
    
    with torch.no_grad():  # Disable gradient computation for inference
        for batch_idx, (images, masks, original_images) in enumerate(tqdm(val_loader, desc="Processing")):
            # Move to device
            images = images.to(device)
            masks = masks.to(device).squeeze(1)  # Remove channel dimension: (B, 1, H, W) -> (B, H, W)
            
            # Forward pass
            # Model outputs a dict with 'out' and 'aux' keys
            # 'out' shape: (B, 21, H, W) - logits for 21 classes
            output = model(images)['out']
            
            # Get predicted class for each pixel: argmax over class dimension
            # predictions shape: (B, H, W)
            predictions = torch.argmax(output, dim=1)
            
            # Store for mIoU computation
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
            
            # Save first batch for visualization
            if batch_idx == 0:
                # original_images is already a tensor/numpy array batch
                sample_images = original_images.numpy() if torch.is_tensor(original_images) else original_images
                sample_masks = masks.cpu()
                sample_predictions = predictions.cpu()
            
            # Process only first few batches for quick testing
            if batch_idx >= 10:  # Process ~44 images (11 batches × 4 images)
                break
    
    print(f"✓ Inference completed")
    
    # =========================================================================
    # 4. Compute mIoU
    # =========================================================================
    print("\n[4/5] Computing mean Intersection-over-Union (mIoU)...")
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute mIoU
    miou, iou_per_class = compute_miou(all_predictions, all_targets, num_classes=21)
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"{'='*70}")
    print(f"Mean IoU (mIoU): {miou:.4f} ({miou*100:.2f}%)")
    print(f"\nPer-class IoU:")
    
    # Class names for PASCAL VOC
    voc_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
        'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
        'train', 'tv/monitor'
    ]
    
    for cls_idx, (cls_name, iou) in enumerate(zip(voc_classes, iou_per_class)):
        if not np.isnan(iou):
            print(f"  {cls_idx:2d}. {cls_name:15s}: {iou:.4f} ({iou*100:.2f}%)")
        else:
            print(f"  {cls_idx:2d}. {cls_name:15s}: N/A (not present)")
    
    # =========================================================================
    # 5. Visualize Predictions
    # =========================================================================
    print(f"\n[5/5] Visualizing predictions...")
    visualize_predictions(sample_images, sample_masks, sample_predictions, num_samples=4)
    
    print(f"\n{'='*70}")
    print("Step 1 completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

