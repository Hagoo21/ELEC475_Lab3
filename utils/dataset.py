"""
Dataset utilities for PASCAL VOC 2012 segmentation.

Provides DataLoader setup with proper preprocessing and augmentation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import numpy as np
from PIL import Image


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VOCSegmentationDataset(Dataset):
    """
    PASCAL VOC 2012 Segmentation Dataset with custom transforms.
    
    Args:
        root (str): Root directory of the VOC dataset (contains JPEGImages, SegmentationClass, etc.)
        image_set (str): 'train', 'trainval', or 'val'
        transform (callable, optional): Transform for images
        target_transform (callable, optional): Transform for masks
    """
    
    def __init__(self, root, image_set='train', transform=None, 
                 target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        
        # Paths to directories
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.mask_dir = os.path.join(root, 'SegmentationClass')
        self.splits_dir = os.path.join(root, 'ImageSets', 'Segmentation')
        
        # Read image IDs from split file
        split_file = os.path.join(self.splits_dir, f'{image_set}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.image_ids)} images for {image_set} set")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Get image and mask pair.
        
        Returns:
            tuple: (image, mask) where mask has values 0-20 for classes and 255 for ignore
        """
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, f'{img_id}.png')
        mask = Image.open(mask_path)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        
        return image, mask


class JointRandomResize:
    """
    Randomly resize image and mask together.
    
    Args:
        size (int or tuple): Target size
    """
    
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def __call__(self, image, mask):
        image = transforms.functional.resize(image, self.size, 
                                            interpolation=Image.BILINEAR)
        mask = transforms.functional.resize(mask, self.size, 
                                           interpolation=Image.NEAREST)
        return image, mask


class JointRandomHorizontalFlip:
    """
    Randomly flip image and mask horizontally together.
    
    Args:
        p (float): Probability of flipping
    """
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if torch.rand(1) < self.p:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return image, mask


def mask_transform(mask):
    """
    Transform mask from PIL to tensor.
    
    Args:
        mask (PIL.Image): Mask image
    
    Returns:
        torch.Tensor: Mask tensor with class indices
    """
    mask = np.array(mask, dtype=np.int64)
    # VOC masks use 255 for boundaries/ignore regions
    # Class labels are 0-20
    return torch.from_numpy(mask)


def joint_transform_train(image, mask, image_size):
    """
    Joint transforms for training (resize + random flip).
    
    Args:
        image (PIL.Image): Input image
        mask (PIL.Image): Input mask
        image_size (int): Target size
    
    Returns:
        tuple: Transformed (image, mask)
    """
    # Resize to fixed size (tuple)
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    image = transforms.functional.resize(image, size, interpolation=Image.BILINEAR)
    mask = transforms.functional.resize(mask, size, interpolation=Image.NEAREST)
    
    # Random horizontal flip
    if torch.rand(1) < 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)
    
    return image, mask


def joint_transform_val(image, mask, image_size):
    """
    Joint transforms for validation (resize only).
    
    Args:
        image (PIL.Image): Input image
        mask (PIL.Image): Input mask
        image_size (int): Target size
    
    Returns:
        tuple: Transformed (image, mask)
    """
    # Only resize to fixed size
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    image = transforms.functional.resize(image, size, interpolation=Image.BILINEAR)
    mask = transforms.functional.resize(mask, size, interpolation=Image.NEAREST)
    
    return image, mask


def get_transforms(image_size=512, is_training=True):
    """
    Get transforms for images and masks.
    
    Args:
        image_size (int): Target size for resizing
        is_training (bool): Whether in training mode (enables augmentation)
    
    Returns:
        tuple: (image_transform, mask_transform_fn, joint_transform_fn)
    """
    # Image transforms
    if is_training:
        image_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                  saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    # Joint transforms (applied to both image and mask)
    if is_training:
        joint_transforms_fn = lambda img, msk: joint_transform_train(img, msk, image_size)
    else:
        joint_transforms_fn = lambda img, msk: joint_transform_val(img, msk, image_size)
    
    return image_transform, mask_transform, joint_transforms_fn


class VOCSegmentationWithJointTransform(Dataset):
    """
    VOC dataset with joint transforms applied to both image and mask.
    
    Args:
        root (str): Root directory of the VOC dataset (contains JPEGImages, SegmentationClass, etc.)
        image_set (str): 'train', 'trainval', or 'val'
        image_size (int): Target size for resizing
        is_training (bool): Whether in training mode
    """
    
    def __init__(self, root, image_set='train', image_size=512, is_training=True):
        self.root = root
        self.image_set = image_set
        self.is_training = is_training
        
        # Load base dataset
        self.voc = VOCSegmentationDataset(root=root, image_set=image_set)
        
        # Get transforms
        self.image_transform, self.mask_transform, self.joint_transform = \
            get_transforms(image_size=image_size, is_training=is_training)
    
    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, idx):
        """
        Get transformed image and mask pair.
        
        Returns:
            tuple: (image, mask)
                - image: [3, H, W] normalized tensor
                - mask: [H, W] long tensor with class indices
        """
        image, mask = self.voc[idx]
        
        # Apply joint transforms (resize, flip)
        image, mask = self.joint_transform(image, mask)
        
        # Apply individual transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask


def get_voc_dataloaders(data_root, image_size=512, batch_size=8, 
                        num_workers=4, train_set='train', val_set='val'):
    """
    Create DataLoaders for PASCAL VOC 2012 segmentation.
    
    Args:
        data_root (str): Root directory containing VOC dataset
        image_size (int): Target size for resizing (default: 512)
        batch_size (int): Batch size (default: 8)
        num_workers (int): Number of worker processes (default: 4)
        train_set (str): Training set name ('train' or 'trainval')
        val_set (str): Validation set name ('val')
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Training dataset
    train_dataset = VOCSegmentationWithJointTransform(
        root=data_root,
        image_set=train_set,
        image_size=image_size,
        is_training=True
    )
    
    # Validation dataset
    val_dataset = VOCSegmentationWithJointTransform(
        root=data_root,
        image_set=val_set,
        image_size=image_size,
        is_training=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def denormalize_image(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize image tensor for visualization.
    
    Args:
        image_tensor (torch.Tensor): Normalized image [C, H, W] or [B, C, H, W]
        mean (list): Mean values used for normalization
        std (list): Std values used for normalization
    
    Returns:
        torch.Tensor: Denormalized image
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if image_tensor.dim() == 4:  # Batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1)
    
    return image


if __name__ == "__main__":
    """Test dataset loading and transforms."""
    import matplotlib.pyplot as plt
    
    print("Testing VOC Dataset Loading...\n")
    
    # Dataset path - use your actual structure
    data_root = "./data/VOC2012_train_val/VOC2012_train_val"
    
    if not os.path.exists(data_root):
        print(f"WARNING: Dataset not found at {data_root}")
        print("Please adjust the path to match your directory structure.")
        data_root = "./data"
    else:
        print(f"Found dataset at: {data_root}\n")
    
    # Create dataset
    try:
        print(f"Creating dataset with image_size=256...")
        train_dataset = VOCSegmentationWithJointTransform(
            root=data_root,
            image_set='train',
            image_size=256,
            is_training=True
        )
        
        # Get a sample
        if len(train_dataset) > 0:
            image, mask = train_dataset[0]
            
            print(f"\nSample data:")
            print(f"  Image shape: {image.shape}")
            print(f"  Mask shape:  {mask.shape}")
            print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Mask unique values: {torch.unique(mask).tolist()}")
            print(f"  Classes in mask: {[v for v in torch.unique(mask).tolist() if v != 255]}")
            
            # Test dataloader
            print(f"\nCreating DataLoader...")
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
            
            batch_images, batch_masks = next(iter(train_loader))
            print(f"  Batch images shape: {batch_images.shape}")
            print(f"  Batch masks shape:  {batch_masks.shape}")
            
            print("\n[SUCCESS] Dataset loading tested successfully!")
        else:
            print("\n[WARNING] Dataset is empty. Check the data path.")
            
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        print("\nExpected structure: data/VOC2012_train_val/VOC2012_train_val/")
        print("  - JPEGImages/")
        print("  - SegmentationClass/")
        print("  - ImageSets/Segmentation/")

