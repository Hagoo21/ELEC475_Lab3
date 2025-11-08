"""
Exploratory Data Analysis (EDA) for PASCAL VOC 2012 Segmentation Dataset

This script analyzes the dataset to understand:
- Number of images in each partition (train/val/test)
- Class names and distribution
- Dataset statistics and visualizations
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd

# PASCAL VOC 2012 class names (21 classes including background)
VOC_CLASSES = [
    'background',      # 0
    'aeroplane',       # 1
    'bicycle',         # 2
    'bird',            # 3
    'boat',            # 4
    'bottle',          # 5
    'bus',             # 6
    'car',             # 7
    'cat',             # 8
    'chair',           # 9
    'cow',             # 10
    'diningtable',     # 11
    'dog',             # 12
    'horse',           # 13
    'motorbike',       # 14
    'person',          # 15
    'pottedplant',     # 16
    'sheep',           # 17
    'sofa',            # 18
    'train',           # 19
    'tvmonitor'        # 20
]

# Color palette for visualization (Pascal VOC standard)
VOC_COLORMAP = [
    [0, 0, 0],         # background
    [128, 0, 0],       # aeroplane
    [0, 128, 0],       # bicycle
    [128, 128, 0],     # bird
    [0, 0, 128],       # boat
    [128, 0, 128],     # bottle
    [0, 128, 128],     # bus
    [128, 128, 128],   # car
    [64, 0, 0],        # cat
    [192, 0, 0],       # chair
    [64, 128, 0],      # cow
    [192, 128, 0],     # diningtable
    [64, 0, 128],      # dog
    [192, 0, 128],     # horse
    [64, 128, 128],    # motorbike
    [192, 128, 128],   # person
    [0, 64, 0],        # pottedplant
    [128, 64, 0],      # sheep
    [0, 192, 0],       # sofa
    [128, 192, 0],     # train
    [0, 64, 128]       # tvmonitor
]


def read_split_file(split_file_path):
    """Read image IDs from a split file."""
    if not os.path.exists(split_file_path):
        return []
    with open(split_file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def count_images_per_partition(data_root):
    """Count number of images in each partition."""
    print("\n" + "="*70)
    print("DATASET PARTITIONS")
    print("="*70)
    
    partitions = {}
    
    # Train and Val from VOC2012_train_val
    train_val_root = os.path.join(data_root, 'VOC2012_train_val', 'VOC2012_train_val')
    if os.path.exists(train_val_root):
        splits_dir = os.path.join(train_val_root, 'ImageSets', 'Segmentation')
        
        # Train set
        train_file = os.path.join(splits_dir, 'train.txt')
        train_ids = read_split_file(train_file)
        partitions['train'] = train_ids
        print(f"Train set:      {len(train_ids):5d} images")
        
        # Val set
        val_file = os.path.join(splits_dir, 'val.txt')
        val_ids = read_split_file(val_file)
        partitions['val'] = val_ids
        print(f"Validation set: {len(val_ids):5d} images")
        
        # Trainval set (combined)
        trainval_file = os.path.join(splits_dir, 'trainval.txt')
        trainval_ids = read_split_file(trainval_file)
        partitions['trainval'] = trainval_ids
        print(f"Train+Val set:  {len(trainval_ids):5d} images")
    
    # Test set from VOC2012_test
    test_root = os.path.join(data_root, 'VOC2012_test', 'VOC2012_test')
    if os.path.exists(test_root):
        test_splits_dir = os.path.join(test_root, 'ImageSets', 'Segmentation')
        test_file = os.path.join(test_splits_dir, 'test.txt')
        test_ids = read_split_file(test_file)
        partitions['test'] = test_ids
        print(f"Test set:       {len(test_ids):5d} images")
    
    print(f"\nTotal images:   {sum(len(ids) for ids in partitions.values() if ids in [partitions.get('train', []), partitions.get('val', []), partitions.get('test', [])])} images")
    print("="*70)
    
    return partitions


def analyze_class_distribution(data_root, partition='train', sample_size=None):
    """Analyze class distribution in segmentation masks."""
    print(f"\n" + "="*70)
    print(f"CLASS DISTRIBUTION ANALYSIS - {partition.upper()} SET")
    print("="*70)
    
    # Get the right root directory
    if partition in ['train', 'val', 'trainval']:
        root = os.path.join(data_root, 'VOC2012_train_val', 'VOC2012_train_val')
        mask_dir = os.path.join(root, 'SegmentationClass')
        splits_dir = os.path.join(root, 'ImageSets', 'Segmentation')
    else:  # test
        root = os.path.join(data_root, 'VOC2012_test', 'VOC2012_test')
        # Test set doesn't have segmentation masks
        print(f"WARNING: Test set does not have segmentation masks available.")
        return None, None
    
    # Read image IDs
    split_file = os.path.join(splits_dir, f'{partition}.txt')
    image_ids = read_split_file(split_file)
    
    if sample_size and sample_size < len(image_ids):
        print(f"Sampling {sample_size} images from {len(image_ids)} total images...")
        np.random.seed(42)
        image_ids = np.random.choice(image_ids, sample_size, replace=False).tolist()
    
    # Count pixels for each class
    class_pixel_counts = Counter()
    class_image_counts = Counter()  # How many images contain each class
    image_class_counts = []  # Number of classes per image
    
    print(f"Analyzing {len(image_ids)} images...")
    for img_id in tqdm(image_ids, desc="Processing masks"):
        mask_path = os.path.join(mask_dir, f'{img_id}.png')
        
        if not os.path.exists(mask_path):
            continue
        
        mask = np.array(Image.open(mask_path))
        
        # Get unique classes in this mask
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != 255]  # Remove ignore label
        
        # Count pixels for each class
        for class_id in unique_classes:
            pixel_count = np.sum(mask == class_id)
            class_pixel_counts[class_id] += pixel_count
            class_image_counts[class_id] += 1
        
        image_class_counts.append(len(unique_classes))
    
    # Create summary statistics
    total_pixels = sum(class_pixel_counts.values())
    
    print(f"\nCLASS STATISTICS:")
    print(f"{'Class ID':<10} {'Class Name':<15} {'Images':<10} {'Pixels':<15} {'% of Total':<12}")
    print("-" * 70)
    
    # Sort by class ID
    for class_id in range(len(VOC_CLASSES)):
        if class_id in class_pixel_counts:
            class_name = VOC_CLASSES[class_id]
            img_count = class_image_counts[class_id]
            pixel_count = class_pixel_counts[class_id]
            percentage = (pixel_count / total_pixels) * 100
            
            print(f"{class_id:<10} {class_name:<15} {img_count:<10} {pixel_count:<15,} {percentage:>6.2f}%")
    
    print("-" * 70)
    print(f"Total pixels:  {total_pixels:,}")
    
    # Image statistics
    print(f"\nIMAGE STATISTICS:")
    print(f"Average classes per image: {np.mean(image_class_counts):.2f}")
    print(f"Min classes per image:     {np.min(image_class_counts)}")
    print(f"Max classes per image:     {np.max(image_class_counts)}")
    print(f"Median classes per image:  {np.median(image_class_counts):.1f}")
    
    return class_pixel_counts, class_image_counts


def visualize_class_distribution(class_pixel_counts, class_image_counts, partition='train'):
    """Create visualization of class distribution."""
    if class_pixel_counts is None or class_image_counts is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Class Distribution Analysis - {partition.upper()} Set', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    class_ids = sorted(class_pixel_counts.keys())
    class_names = [VOC_CLASSES[i] for i in class_ids]
    pixel_counts = [class_pixel_counts[i] for i in class_ids]
    image_counts = [class_image_counts[i] for i in class_ids]
    
    # 1. Pixel distribution bar chart
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(class_ids)), pixel_counts, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Class', fontweight='bold')
    ax1.set_ylabel('Total Pixels', fontweight='bold')
    ax1.set_title('Total Pixels per Class')
    ax1.set_xticks(range(len(class_ids)))
    ax1.set_xticklabels([f"{i}\n{VOC_CLASSES[i][:6]}" for i in class_ids], 
                        rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Image frequency bar chart
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(class_ids)), image_counts, color='coral', alpha=0.8)
    ax2.set_xlabel('Class', fontweight='bold')
    ax2.set_ylabel('Number of Images', fontweight='bold')
    ax2.set_title('Number of Images Containing Each Class')
    ax2.set_xticks(range(len(class_ids)))
    ax2.set_xticklabels([f"{i}\n{VOC_CLASSES[i][:6]}" for i in class_ids], 
                        rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Pixel percentage pie chart (top 10 classes)
    ax3 = axes[1, 0]
    total_pixels = sum(pixel_counts)
    percentages = [(count / total_pixels) * 100 for count in pixel_counts]
    
    # Sort and get top 10
    sorted_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)[:10]
    top_names = [class_names[i] for i in sorted_indices]
    top_percentages = [percentages[i] for i in sorted_indices]
    
    colors = plt.cm.Set3(range(len(top_names)))
    wedges, texts, autotexts = ax3.pie(top_percentages, labels=top_names, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    ax3.set_title('Pixel Distribution (Top 10 Classes)')
    
    # 4. Log scale comparison
    ax4 = axes[1, 1]
    x = range(len(class_ids))
    width = 0.35
    ax4.bar([i - width/2 for i in x], pixel_counts, width, label='Pixels', 
            color='steelblue', alpha=0.8)
    ax4.bar([i + width/2 for i in x], 
            [c * (total_pixels / len(class_pixel_counts) / max(image_counts)) for c in image_counts], 
            width, label='Images (scaled)', color='coral', alpha=0.8)
    ax4.set_xlabel('Class', fontweight='bold')
    ax4.set_ylabel('Count (log scale)', fontweight='bold')
    ax4.set_title('Pixel Count vs Image Frequency (Log Scale)')
    ax4.set_yscale('log')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{i}" for i in class_ids], fontsize=8)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'eda_class_distribution_{partition}.png', dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Saved visualization to: eda_class_distribution_{partition}.png")
    plt.show()


def visualize_sample_images(data_root, partition='train', num_samples=6):
    """Visualize sample images with their segmentation masks."""
    print(f"\n" + "="*70)
    print(f"SAMPLE VISUALIZATIONS - {partition.upper()} SET")
    print("="*70)
    
    # Get the right root directory
    if partition in ['train', 'val', 'trainval']:
        root = os.path.join(data_root, 'VOC2012_train_val', 'VOC2012_train_val')
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')
        splits_dir = os.path.join(root, 'ImageSets', 'Segmentation')
    else:  # test
        root = os.path.join(data_root, 'VOC2012_test', 'VOC2012_test')
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = None  # No masks for test set
        splits_dir = os.path.join(root, 'ImageSets', 'Segmentation')
    
    # Read image IDs
    split_file = os.path.join(splits_dir, f'{partition}.txt')
    image_ids = read_split_file(split_file)
    
    # Random sample
    np.random.seed(42)
    sampled_ids = np.random.choice(image_ids, min(num_samples, len(image_ids)), replace=False)
    
    # Create visualization
    rows = num_samples
    cols = 3 if mask_dir else 1
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_id in enumerate(sampled_ids):
        # Load image
        img_path = os.path.join(image_dir, f'{img_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # Show original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Image: {img_id}', fontweight='bold')
        axes[idx, 0].axis('off')
        
        if mask_dir:
            # Load mask
            mask_path = os.path.join(mask_dir, f'{img_id}.png')
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))
                
                # Show mask
                axes[idx, 1].imshow(mask, cmap='tab20', vmin=0, vmax=20)
                axes[idx, 1].set_title('Segmentation Mask', fontweight='bold')
                axes[idx, 1].axis('off')
                
                # Show overlay
                overlay = np.array(image).copy()
                colored_mask = np.zeros_like(overlay)
                
                for class_id in range(len(VOC_CLASSES)):
                    if class_id < len(VOC_COLORMAP):
                        mask_class = (mask == class_id)
                        colored_mask[mask_class] = VOC_COLORMAP[class_id]
                
                # Blend
                alpha = 0.5
                overlay = (overlay * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
                
                axes[idx, 2].imshow(overlay)
                axes[idx, 2].set_title('Overlay', fontweight='bold')
                axes[idx, 2].axis('off')
                
                # Show classes present
                unique_classes = np.unique(mask)
                unique_classes = unique_classes[(unique_classes != 255) & (unique_classes < len(VOC_CLASSES))]
                classes_str = ', '.join([VOC_CLASSES[c] for c in unique_classes])
                axes[idx, 0].text(0.5, -0.1, f'Classes: {classes_str}', 
                                transform=axes[idx, 0].transAxes,
                                ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'eda_sample_images_{partition}.png', dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved sample visualizations to: eda_sample_images_{partition}.png")
    plt.show()


def analyze_image_sizes(data_root, partition='train', sample_size=100):
    """Analyze image dimensions in the dataset."""
    print(f"\n" + "="*70)
    print(f"IMAGE SIZE ANALYSIS - {partition.upper()} SET")
    print("="*70)
    
    # Get the right root directory
    if partition in ['train', 'val', 'trainval']:
        root = os.path.join(data_root, 'VOC2012_train_val', 'VOC2012_train_val')
        image_dir = os.path.join(root, 'JPEGImages')
        splits_dir = os.path.join(root, 'ImageSets', 'Segmentation')
    else:  # test
        root = os.path.join(data_root, 'VOC2012_test', 'VOC2012_test')
        image_dir = os.path.join(root, 'JPEGImages')
        splits_dir = os.path.join(root, 'ImageSets', 'Segmentation')
    
    # Read image IDs
    split_file = os.path.join(splits_dir, f'{partition}.txt')
    image_ids = read_split_file(split_file)
    
    # Sample
    if sample_size and sample_size < len(image_ids):
        np.random.seed(42)
        image_ids = np.random.choice(image_ids, sample_size, replace=False).tolist()
    
    widths = []
    heights = []
    aspect_ratios = []
    
    print(f"Analyzing {len(image_ids)} images...")
    for img_id in tqdm(image_ids, desc="Processing images"):
        img_path = os.path.join(image_dir, f'{img_id}.jpg')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
    
    print(f"\nIMAGE SIZE STATISTICS:")
    print(f"Width:  Min={min(widths):4d}, Max={max(widths):4d}, Mean={np.mean(widths):6.1f}, Median={np.median(widths):6.1f}")
    print(f"Height: Min={min(heights):4d}, Max={max(heights):4d}, Mean={np.mean(heights):6.1f}, Median={np.median(heights):6.1f}")
    print(f"Aspect Ratio: Min={min(aspect_ratios):.2f}, Max={max(aspect_ratios):.2f}, Mean={np.mean(aspect_ratios):.2f}")
    
    # Find most common sizes
    size_counter = Counter(zip(widths, heights))
    print(f"\nMOST COMMON IMAGE SIZES:")
    for (w, h), count in size_counter.most_common(10):
        print(f"  {w}x{h}: {count} images")
    
    return widths, heights, aspect_ratios


def generate_summary_report(data_root):
    """Generate a comprehensive summary report."""
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE SUMMARY REPORT")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("PASCAL VOC 2012 SEGMENTATION DATASET - EDA REPORT")
    report.append("="*70)
    report.append("")
    
    # Partition counts
    partitions = count_images_per_partition(data_root)
    report.append("DATASET PARTITIONS:")
    for name, ids in partitions.items():
        if ids:
            report.append(f"  {name:12s}: {len(ids):5d} images")
    report.append("")
    
    # Classes
    report.append("CLASSES (21 total):")
    for i, class_name in enumerate(VOC_CLASSES):
        report.append(f"  {i:2d}: {class_name}")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open('eda_summary_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n[SUCCESS] Summary report saved to: eda_summary_report.txt")


def main():
    """Main EDA function."""
    print("\n" + "="*70)
    print("PASCAL VOC 2012 SEGMENTATION DATASET")
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    # Dataset path
    data_root = './data'
    
    if not os.path.exists(data_root):
        print(f"[ERROR] Data directory not found at {data_root}")
        return
    
    # 1. Count images per partition
    partitions = count_images_per_partition(data_root)
    
    # 2. Analyze class distribution for train set
    class_pixel_counts, class_image_counts = analyze_class_distribution(
        data_root, partition='train', sample_size=None  # Set to None to analyze all images
    )
    
    # 3. Visualize class distribution
    if class_pixel_counts:
        visualize_class_distribution(class_pixel_counts, class_image_counts, partition='train')
    
    # 4. Visualize sample images
    visualize_sample_images(data_root, partition='train', num_samples=6)
    
    # 5. Analyze image sizes
    widths, heights, aspect_ratios = analyze_image_sizes(
        data_root, partition='train', sample_size=100
    )
    
    # 6. Generate summary report
    generate_summary_report(data_root)
    
    print("\n" + "="*70)
    print("[SUCCESS] EDA COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - eda_class_distribution_train.png")
    print("  - eda_sample_images_train.png")
    print("  - eda_summary_report.txt")
    print("\n")


if __name__ == "__main__":
    main()

