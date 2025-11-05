import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np

# Load pretrained FCN ResNet50 model with COCO+VOC weights
model = torchvision.models.segmentation.fcn_resnet50(
    weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1"
)
model.eval()  # Set to evaluation mode (disables dropout, batchnorm in eval mode)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

# Define transforms for input images
# FCN ResNet50 expects: 3-channel RGB images, normalized with ImageNet stats
img_transform = transforms.Compose([
    transforms.Resize((520, 520)),  # Resize to consistent size
    transforms.ToTensor(),  # Convert PIL Image to tensor [0,1], shape: [C, H, W]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Transform for target masks - just resize and convert to tensor
target_transform = transforms.Compose([
    transforms.Resize((520, 520), interpolation=transforms.InterpolationMode.NEAREST),  # Nearest neighbor for labels
    transforms.PILToTensor()  # Shape: [1, H, W] with integer class labels
])

# Download PASCAL VOC 2012 segmentation dataset (validation split)
dataset = VOCSegmentation(
    root='./data',
    year='2012',
    image_set='val',  # Use validation set
    download=True,
    transform=img_transform,
    target_transform=target_transform
)
print(f"Dataset loaded: {len(dataset)} images")

# Dataset EDA

# Inference

def compute_miou(pred, target, num_classes=21):
    """
    Compute mean Intersection-over-Union (mIoU)
    
    IoU = (True Positive) / (True Positive + False Positive + False Negative)
    mIoU = average IoU across all classes
    
    Args:
        pred: predicted segmentation mask [H, W] with class indices
        target: ground truth mask [H, W] with class indices
        num_classes: number of classes (21 for VOC: 20 objects + background)
    
    Returns:
        mIoU: mean IoU score
    """
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Ignore class 255 (border/void class in VOC)
    target[target == 255] = 0
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union > 0:
            ious.append(intersection / union)
    
    return np.mean(ious) if ious else 0.0

# Run inference on a few validation images
num_samples = 4
mious = []

fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

for i in range(num_samples):
    # Get image and ground truth mask
    img_tensor, target_tensor = dataset[i]
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    # Run inference (no gradient computation needed)
    with torch.no_grad():
        output = model(img_batch)['out']  # Shape: [1, 21, H, W] (21 class scores per pixel)
    
    # Get predicted class per pixel: argmax over class dimension
    pred_mask = output.argmax(1).squeeze(0)  # Shape: [H, W]
    
    # Compute mIoU
    target_mask = target_tensor.squeeze(0)  # Shape: [H, W]
    miou = compute_miou(pred_mask, target_mask)
    mious.append(miou)
    
    # Denormalize image for visualization
    img_display = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)
    
    # Display original image
    axes[i, 0].imshow(img_display)
    axes[i, 0].set_title(f"Original Image {i+1}")
    axes[i, 0].axis('off')
    
    # Display predicted segmentation mask
    axes[i, 1].imshow(pred_mask.cpu().numpy(), cmap='tab20', vmin=0, vmax=20)
    axes[i, 1].set_title(f"Predicted Mask (mIoU: {miou:.3f})")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()

print(f"\nMean mIoU across {num_samples} samples: {np.mean(mious):.3f}")

