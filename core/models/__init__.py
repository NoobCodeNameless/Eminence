from .autoencoder import AutoEncoder
from .baseline_MNIST_network import BaselineMNISTNetwork
from .resnet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .vgg import *
from .vit import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14, vit_b_8, vit_tiny_8, swin_t, ViT, SimpleViT, CCT
from .unet import UNet, UNetLittle

__all__ = [
    'AutoEncoder', 'BaselineMNISTNetwork', 'ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    'vgg11', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14', 'vit_b_8', 'vit_tiny_8', 'swin_t', # deprecated
    'ViT', 'SimpleViT', 'CCT', # new ViT models, lol
    'UNet', 'UNetLittle'
]

