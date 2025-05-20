import torch.nn as nn
import timm


class Siglip(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model('vit_large_patch16_siglip_384.v2_webli', pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
