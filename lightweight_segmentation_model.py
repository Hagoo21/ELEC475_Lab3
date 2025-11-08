"""
Lightweight Semantic Segmentation Model for PASCAL VOC 2012

Architecture:
- Encoder: MobileNetV3-Small (pretrained on ImageNet)
- Context Module: ASPP (Atrous Spatial Pyramid Pooling)
- Decoder: Skip connections + Bilinear Upsampling
- Output: 21 classes (PASCAL VOC)

Author: ELEC475 Lab 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    
    Captures multi-scale contextual information using parallel atrous convolutions
    with different dilation rates.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels per branch
        dilation_rates (list): List of dilation rates for atrous convolutions
    """
    
    def __init__(self, in_channels, out_channels=256, dilation_rates=[1, 6, 12, 18]):
        super(ASPPModule, self).__init__()
        
        self.aspp_branches = nn.ModuleList()
        
        # Branch 1: 1x1 convolution
        self.aspp_branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Branches 2-4: 3x3 atrous convolutions with different dilation rates
        for dilation in dilation_rates[1:]:
            self.aspp_branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Project concatenated features
        # Total channels = out_channels * (len(dilation_rates) + 1) for branches + global pooling
        total_channels = out_channels * (len(dilation_rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        """
        Forward pass through ASPP module.
        
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
            
        Returns:
            torch.Tensor: Output feature map [B, out_channels, H, W]
        """
        res = []
        
        # Apply all ASPP branches
        for aspp_branch in self.aspp_branches:
            res.append(aspp_branch(x))
        
        # Global average pooling branch
        global_features = self.global_avg_pool(x)
        global_features = F.interpolate(global_features, size=x.shape[2:], 
                                       mode='bilinear', align_corners=False)
        res.append(global_features)
        
        # Concatenate all branches and project
        res = torch.cat(res, dim=1)
        return self.project(res)


class LightweightSegmentationModel(nn.Module):
    """
    Lightweight semantic segmentation model using MobileNetV3-Small backbone.
    
    Architecture:
    - Encoder: MobileNetV3-Small (pretrained on ImageNet) up to stride 16
    - Context: ASPP module with dilation rates {1, 6, 12, 18}
    - Decoder: Skip connection from low-level features + bilinear upsampling
    - Classifier: 1×1 convolution to 21 classes (PASCAL VOC)
    
    Feature taps for knowledge distillation:
    - Low-level:  stride 4  (early spatial information)
    - Mid-level:  stride 8  (intermediate features)
    - High-level: stride 16 (semantic features)
    
    Args:
        num_classes (int): Number of segmentation classes (default: 21 for PASCAL VOC)
        pretrained (bool): Use pretrained MobileNetV3 weights (default: True)
        return_features (bool): Return intermediate features for KD (default: False)
    """
    
    def __init__(self, num_classes=21, pretrained=True, return_features=False):
        super(LightweightSegmentationModel, self).__init__()
        
        self.num_classes = num_classes
        self.return_features = return_features
        
        # Load pretrained MobileNetV3-Small backbone
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small(weights=None)
        
        # Extract feature extractor (remove classifier)
        self.features = backbone.features
        
        # MobileNetV3-Small feature extraction points:
        # - Low-level (stride 4):  after layer 1  (16 channels)
        # - Mid-level (stride 8):  after layer 3  (24 channels)
        # - High-level (stride 16): after layer 8  (48 channels)
        # - Final (stride 16):     after layer 12 (576 channels)
        
        # Low-level feature projection (stride 4, 16 channels)
        self.low_level_project = nn.Sequential(
            nn.Conv2d(16, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # ASPP on high-level features (stride 16, 576 channels)
        self.aspp = ASPPModule(in_channels=576, out_channels=256, 
                              dilation_rates=[1, 6, 12, 18])
        
        # Decoder: Upsample ASPP output and fuse with low-level features
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Initialize decoder and classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for decoder and classifier."""
        for m in [self.low_level_project, self.decoder, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', 
                                           nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the segmentation model.
        
        Args:
            x (torch.Tensor): Input image [B, 3, H, W]
            
        Returns:
            If return_features is False:
                torch.Tensor: Segmentation logits [B, num_classes, H, W]
            If return_features is True:
                tuple: (logits, dict of feature maps)
                    - logits: [B, num_classes, H, W]
                    - features: {
                        'low': [B, C_low, H/4, W/4],
                        'mid': [B, C_mid, H/8, W/8],
                        'high': [B, C_high, H/16, W/16]
                    }
        """
        input_shape = x.shape[-2:]  # (H, W)
        
        # Feature extraction through MobileNetV3-Small backbone
        features_dict = {}
        
        # Low-level features (stride 4, after layer 1)
        x = self.features[0](x)   # First InvertedResidual block
        x = self.features[1](x)   # Second InvertedResidual block
        low_level_feat = x        # [B, 16, H/4, W/4]
        if self.return_features:
            features_dict['low'] = low_level_feat
        
        # Mid-level features (stride 8, after layer 3)
        x = self.features[2](x)
        x = self.features[3](x)
        mid_level_feat = x        # [B, 24, H/8, W/8]
        if self.return_features:
            features_dict['mid'] = mid_level_feat
        
        # High-level features (stride 16, after layer 8)
        for i in range(4, 9):
            x = self.features[i](x)
        high_level_feat = x       # [B, 48, H/16, W/16]
        if self.return_features:
            features_dict['high'] = high_level_feat
        
        # Continue to final features (stride 16)
        for i in range(9, len(self.features)):
            x = self.features[i](x)
        # x is now [B, 576, H/16, W/16]
        
        # Apply ASPP context module
        x = self.aspp(x)  # [B, 256, H/16, W/16]
        
        # Upsample to stride 4 (to match low-level features)
        x = F.interpolate(x, size=low_level_feat.shape[2:], 
                         mode='bilinear', align_corners=False)
        # x is now [B, 256, H/4, W/4]
        
        # Project and concatenate with low-level features
        low_level_feat = self.low_level_project(low_level_feat)  # [B, 48, H/4, W/4]
        x = torch.cat([x, low_level_feat], dim=1)  # [B, 304, H/4, W/4]
        
        # Decoder
        x = self.decoder(x)  # [B, 256, H/4, W/4]
        
        # Classifier
        x = self.classifier(x)  # [B, num_classes, H/4, W/4]
        
        # Upsample to input resolution
        logits = F.interpolate(x, size=input_shape, mode='bilinear', 
                              align_corners=False)  # [B, num_classes, H, W]
        
        if self.return_features:
            return logits, features_dict
        else:
            return logits


def count_parameters(model):
    """
    Count the number of trainable and total parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_summary(model):
    """
    Print a detailed summary of the model architecture and parameters.
    
    Args:
        model (nn.Module): PyTorch model
    """
    print("=" * 80)
    print("Model Architecture Summary")
    print("=" * 80)
    print(model)
    print("=" * 80)
    
    total_params, trainable_params = count_parameters(model)
    
    print(f"\nTotal parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Non-trainable params:  {total_params - trainable_params:,}")
    print(f"\nModel size (MB):       {total_params * 4 / (1024 ** 2):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    """
    Test the model with dummy input and verify output shapes.
    """
    print("Testing Lightweight Segmentation Model...\n")
    
    # Create model
    model = LightweightSegmentationModel(num_classes=21, pretrained=True, 
                                        return_features=True)
    model.eval()
    
    # Test input (batch_size=2, channels=3, height=512, width=512)
    dummy_input = torch.randn(2, 3, 512, 512)
    
    print(f"Input shape: {dummy_input.shape}\n")
    
    # Forward pass with features
    with torch.no_grad():
        logits, features = model(dummy_input)
    
    print("Output shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Low-level features:  {features['low'].shape}")
    print(f"  Mid-level features:  {features['mid'].shape}")
    print(f"  High-level features: {features['high'].shape}")
    print()
    
    # Test without features
    model.return_features = False
    with torch.no_grad():
        logits_only = model(dummy_input)
    print(f"Logits only shape: {logits_only.shape}\n")
    
    # Print model summary
    print_model_summary(model)
    
    # Verify parameter count
    total, trainable = count_parameters(model)
    print(f"\n[SUCCESS] Model successfully created with {total/1e6:.2f}M parameters")
    print(f"[SUCCESS] Lightweight architecture suitable for efficient inference")

