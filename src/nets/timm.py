import torch.nn as nn
import timm


class TimmModel(nn.Module):
    def __init__(self, model_name, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
