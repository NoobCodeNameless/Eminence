import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ViT_B_16_Weights, ViT_B_32_Weights, 
    ViT_L_16_Weights, ViT_L_32_Weights, 
    ViT_H_14_Weights, Swin_T_Weights
)
from typing import Literal, Optional, Union
from vit_pytorch import ViT as ViT_pytorch
from vit_pytorch import SimpleViT as SimpleViT_pytorch
from vit_pytorch.cct import CCT as CCT_pytorch

class VisionTransformerWrapper(nn.Module):
    """
    Vision Transformer wrapper class that supports 32x32 image inputs
    
    Args:
        model_name: Model name, supports 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14'
        num_classes: Number of classes for classification
        pretrained: Whether to use pretrained weights
    """
    def __init__(
        self, 
        model_name: Literal['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14', 'vit_b_8', 'vit_tiny_8', 'swin_t'],
        num_classes: int = 1000,
        pretrained: bool = False
    ):
        super().__init__()
        
        # Select weights
        weights = None
        if pretrained:
            weights_map = {
                'vit_b_16': ViT_B_16_Weights.DEFAULT,
                'vit_b_32': ViT_B_32_Weights.DEFAULT,
                'vit_l_16': ViT_L_16_Weights.DEFAULT,
                'vit_l_32': ViT_L_32_Weights.DEFAULT,
                'vit_h_14': ViT_H_14_Weights.DEFAULT,
                'swin_t': Swin_T_Weights.DEFAULT,
            }
            weights = weights_map.get(model_name, None)
        
        # # Create model
        # model_func_map = {
        #     'vit_b_16': models.vit_b_16,
        #     'vit_b_32': models.vit_b_32,
        #     'vit_l_16': models.vit_l_16,
        #     'vit_l_32': models.vit_l_32,
        #     'vit_h_14': models.vit_h_14,
        #     # not official implementation
        #     'vit_b_8': models.VisionTransformer(
        #         image_size=224, patch_size=8, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=num_classes
        #     ),
        #     'vit_tiny_8': models.VisionTransformer(
        #         image_size=224, patch_size=8, num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768, num_classes=num_classes
        #     ),
        # }
        
        # # Get model creation function
        # model_func = model_func_map[model_name]
        
        # # Create model
        # self.model = model_func(weights=weights)
        
        if model_name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14', 'swin_t']:
            model_func_map = {
                'vit_b_16': models.vit_b_16,
                'vit_b_32': models.vit_b_32,
                'vit_l_16': models.vit_l_16,
                'vit_l_32': models.vit_l_32,
                'vit_h_14': models.vit_h_14,
                'swin_t': models.swin_t,
            }
            model_func = model_func_map[model_name]
            self.model = model_func(weights=weights)
            
        elif model_name == 'vit_b_8':
            # 
            self.model = models.VisionTransformer(
                image_size=224, patch_size=8, num_layers=12, num_heads=12,
                hidden_dim=768, mlp_dim=3072, num_classes=num_classes
            )
            
        elif model_name == 'vit_tiny_8':
            # 
            self.model = models.VisionTransformer(
                image_size=224, patch_size=8, num_layers=12, num_heads=3,
                hidden_dim=192, mlp_dim=768, num_classes=num_classes
            )
        
        # Modify classification head
        if num_classes != 1000:
            if hasattr(self.model, 'heads'):
                # New version of torchvision
                self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
            elif hasattr(self.model, 'classifier'):
                self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
            elif hasattr(self.model, 'head'):  # torchvision Swin
                self.model.head = nn.Linear(self.model.head.in_features, num_classes)
            else:
                # Old version of torchvision
                self.model.heads = nn.Linear(self.model.hidden_dim, num_classes)
        
       
        self.input_size = 224
        self.upsample = nn.Upsample(size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample to the required input size
        x = self.upsample(x)
        return self.model(x)


def create_vit(
    model_name: Literal['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14', 'vit_b_8', 'vit_tiny_8', 'swin_t'],
    num_classes: int = 1000,
    pretrained: bool = False
) -> VisionTransformerWrapper:
    """
    Create a Vision Transformer model
    
    Args:
        model_name: Model name, supports 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14'
        num_classes: Number of classes for classification
        pretrained: Whether to use pretrained weights
        
    Returns:
        VisionTransformerWrapper instance
    """
    return VisionTransformerWrapper(model_name, num_classes, pretrained)

def vit_b_16(num_classes: int = 1000, pretrained: bool = False) -> VisionTransformerWrapper:
    return create_vit('vit_b_16', num_classes, pretrained)

def vit_b_32(num_classes: int = 1000, pretrained: bool = False) -> VisionTransformerWrapper:
    return create_vit('vit_b_32', num_classes, pretrained)

def vit_l_16(num_classes: int = 1000, pretrained: bool = False) -> VisionTransformerWrapper:
    return create_vit('vit_l_16', num_classes, pretrained)

def vit_l_32(num_classes: int = 1000, pretrained: bool = False) -> VisionTransformerWrapper:
    return create_vit('vit_l_32', num_classes, pretrained)

def vit_h_14(num_classes: int = 1000, pretrained: bool = False) -> VisionTransformerWrapper:
    return create_vit('vit_h_14', num_classes, pretrained)

def vit_b_8(num_classes: int = 1000, pretrained: bool = False) -> VisionTransformerWrapper:
    return create_vit('vit_b_8', num_classes, pretrained)

def vit_tiny_8(num_classes: int = 1000, pretrained: bool = False) -> VisionTransformerWrapper:
    return create_vit('vit_tiny_8', num_classes, pretrained)

def swin_t(num_classes: int = 1000, pretrained: bool = False) -> VisionTransformerWrapper:
    return create_vit('swin_t', num_classes, pretrained)

def ViT(num_classes: int = 1000, image_size: int = 32):
    return ViT_pytorch(image_size=image_size, patch_size=4, num_classes=num_classes, dim=192, depth=6, heads=6, mlp_dim=384)

def SimpleViT(num_classes: int = 1000, image_size: int = 32):
    return SimpleViT_pytorch(image_size=image_size, patch_size=4, num_classes=num_classes, dim=192, depth=6, heads=6, mlp_dim=384)

def CCT(num_classes: int = 1000, image_size: int = 32):
    return CCT_pytorch(
        img_size = image_size,
        embedding_dim = 192,
        n_conv_layers = 2,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        pooling_kernel_size = 2,
        pooling_stride = 2,
        pooling_padding = 0,
        num_layers = 7,
        num_heads = 6,
        mlp_ratio = 3.,
        num_classes = num_classes,
        positional_embedding = 'learnable',
    )
