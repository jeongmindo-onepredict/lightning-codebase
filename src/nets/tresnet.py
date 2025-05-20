import torch.nn as nn
import timm


class TResNetL(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model('tresnet_l', pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
