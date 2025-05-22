import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from typing import Optional, Union
import timm

class HFNetwork(nn.Module):
    """SigLIP2 기반 자동차 분류 네트워크"""
    
    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-384",
        num_classes: int = 400,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.1,
        img_size: Union[int, tuple] = 256,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.img_size = img_size
        
        self._init_hf_model(model_name, num_classes, freeze_backbone, dropout_rate)
    
    def _init_hf_model(self, model_name: str, num_classes: int, freeze_backbone: bool, dropout_rate: float):
        """HuggingFace Transformers 모델 초기화"""
        # SigLIP2 백본 로드
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 특성 차원 계산
        feature_dim = self._get_feature_dim()
        
        # 분류기 헤드 정의
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # 백본 동결
        if freeze_backbone:
            self._freeze_backbone_hf()
    
    def _get_feature_dim(self) -> int:
        """특성 차원 계산"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
            dummy_features = self.backbone.get_image_features(pixel_values=dummy_input)
            return dummy_features.shape[-1]

    def _freeze_backbone_hf(self):
        """HuggingFace 모델의 백본 동결"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone.get_image_features(pixel_values=x)
        return self.classifier(features)
