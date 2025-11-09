"""
Utilities package for PASCAL VOC 2012 segmentation.

This package contains:
- common: Shared constants and helper functions
- dataset: Dataset loading and preprocessing
- metrics: Evaluation metrics computation
"""

from .common import (
    VOC_CLASSES,
    VOC_COLORMAP,
    get_device,
    load_checkpoint,
    load_training_history,
    find_best_epoch,
    colorize_mask,
    get_voc_colormap_tensor,
    print_class_iou,
    read_split_file
)

from .dataset import (
    VOCSegmentationDataset,
    VOCSegmentationWithJointTransform,
    get_voc_dataloaders,
    denormalize_image,
    IMAGENET_MEAN,
    IMAGENET_STD
)

from .metrics import (
    SegmentationMetrics,
    compute_miou,
    compute_metrics_batch
)

__all__ = [
    # Common utilities
    'VOC_CLASSES',
    'VOC_COLORMAP',
    'get_device',
    'load_checkpoint',
    'load_training_history',
    'find_best_epoch',
    'colorize_mask',
    'get_voc_colormap_tensor',
    'print_class_iou',
    'read_split_file',
    
    # Dataset utilities
    'VOCSegmentationDataset',
    'VOCSegmentationWithJointTransform',
    'get_voc_dataloaders',
    'denormalize_image',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    
    # Metrics utilities
    'SegmentationMetrics',
    'compute_miou',
    'compute_metrics_batch',
]

