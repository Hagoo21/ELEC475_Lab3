"""
Models package for ELEC475 Lab 3 - Lightweight Semantic Segmentation

Contains:
- LightweightSegmentationModel: Semantic segmentation model (MobileNetV3-Small + ASPP)
- ASPPModule: Atrous Spatial Pyramid Pooling module
- Helper functions for parameter counting and model summary
"""

from .lightweight_segmentation import (
    LightweightSegmentationModel,
    ASPPModule,
    count_parameters,
    print_model_summary
)

__all__ = [
    'LightweightSegmentationModel',
    'ASPPModule',
    'count_parameters',
    'print_model_summary'
]

__version__ = '1.0.0'

