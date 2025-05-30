# from torchvision import transforms
# from .base import BaseTransforms


# class CarTransforms(BaseTransforms):
#     def __init__(self, img_size: int):
#         super().__init__()
#         self.img_size = img_size

#     def train_transform(self):
#         return transforms.Compose([
#             # 기본 리사이즈
#             transforms.Resize((self.img_size, self.img_size)),
            
#             # 부분 촬영 및 다양한 종횡비 대응
#             transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0), ratio=(0.5, 2.0)),
            
#             # 좌우 반전 (차량은 대칭성이 있음)
#             transforms.RandomHorizontalFlip(p=0.5),
            
#             # 약간의 회전 (촬영 각도 변화)
#             transforms.RandomRotation(15),
            
#             # 원근 변환 (언더뷰, 사이드뷰 등)
#             transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            
#             # 조명 변화 대응
#             transforms.ColorJitter(
#                 brightness=0.3,  # 작업등, 자연광 변화
#                 contrast=0.2,    # 흐린 날씨, 실내외 차이
#                 saturation=0.2,  # 카메라 설정 차이
#                 hue=0.1         # 조명 색온도 변화
#             ),
            
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
#             # 가려짐 시뮬레이션 (장비, 다른 차량에 의한)
#             transforms.RandomErasing(
#                 p=0.3, 
#                 scale=(0.02, 0.15), 
#                 ratio=(0.3, 3.0)
#             ),
#         ])

#     def val_transform(self):
#         return transforms.Compose([
#             transforms.Resize((self.img_size, self.img_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def test_transform(self):
#         return self.val_transform()


import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import math


class RandomResizedCropWithRatio(transforms.RandomResizedCrop):
    """PyTorch의 RandomResizedCrop을 확장하여 특정 비율 범위 지원"""
    
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), **kwargs):
        super().__init__(size, scale=scale, ratio=ratio, **kwargs)


class CarSpecificCrop:
    """차량 부위별 특화 크롭 전략"""
    
    def __init__(self, size, crop_type="full"):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        self.crop_type = crop_type
        
        # 차량 부위별 크롭 파라미터 정의
        self.crop_params = {
            "full": {"scale": (0.7, 1.0), "ratio": (0.8, 1.25)},          # 전체 차량
            "front": {"scale": (0.3, 0.7), "ratio": (1.2, 2.0)},         # 전면부 (그릴, 헤드라이트)
            "bottom": {"scale": (0.25, 0.6), "ratio": (1.5, 3.0)},       # 하단부 (휠, 하부 그릴)
            "side": {"scale": (0.4, 0.8), "ratio": (0.6, 1.8)},          # 측면부
            "rear": {"scale": (0.3, 0.7), "ratio": (1.0, 2.2)},          # 후면부
            "panorama": {"scale": (0.6, 1.0), "ratio": (2.0, 4.0)},      # 파노라마 뷰
            "detail": {"scale": (0.5, 0.8), "ratio": (1.0, 2.0)},        # 세부 영역
        }
    
    def __call__(self, img):
        params = self.crop_params[self.crop_type]
        crop_transform = RandomResizedCropWithRatio(
            size=self.size,
            scale=params["scale"],
            ratio=params["ratio"]
        )
        return crop_transform(img)


class RandomCarCrop:
    """차량 부위별 크롭을 랜덤하게 선택"""
    
    def __init__(self, size, strategies=None, probabilities=None):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        
        if strategies is None:
            strategies = ["full", "front", "bottom", "side", "rear"]
        
        if probabilities is None:
            probabilities = [0.4, 0.15, 0.15, 0.15, 0.15]  # full에 더 높은 확률
            
        self.strategies = strategies
        self.probabilities = probabilities
        
        # 각 전략별 transform 생성
        self.transforms = {
            strategy: CarSpecificCrop(self.size, strategy) 
            for strategy in strategies
        }
    
    def __call__(self, img):
        strategy = np.random.choice(self.strategies, p=self.probabilities)
        return self.transforms[strategy](img)


class EnhancedSharpen:
    """향상된 선명화 효과"""
    
    def __init__(self, alpha_range=(0.1, 0.3), p=0.3):
        self.alpha_range = alpha_range
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, torch.Tensor):
                # Tensor인 경우 PIL로 변환
                img = F.to_pil_image(img)
            
            alpha = random.uniform(*self.alpha_range)
            # UnsharpMask 효과 시뮬레이션
            blurred = img.filter(ImageFilter.GaussianBlur(radius=1))
            # 원본과 블러 이미지의 차이로 선명화
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1 + alpha)
            
        return img


class RandomNoise:
    """가우시안 노이즈 추가"""
    
    def __init__(self, var_range=(10.0, 30.0), p=0.25):
        self.var_range = var_range
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = img.numpy().transpose(1, 2, 0) * 255
                img_array = img_array.astype(np.uint8)
            
            var = random.uniform(*self.var_range)
            noise = np.random.normal(0, np.sqrt(var), img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            if isinstance(img, Image.Image):
                return Image.fromarray(noisy_img)
            else:
                return torch.from_numpy(noisy_img.transpose(2, 0, 1) / 255.0).float()
        
        return img


class RandomCoarseDropout:
    """CoarseDropout 시뮬레이션 (부분 가려짐)"""
    
    def __init__(self, max_holes=3, hole_size_range=(0.05, 0.15), p=0.2):
        self.max_holes = max_holes
        self.hole_size_range = hole_size_range
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, torch.Tensor):
                c, h, w = img.shape
                img_copy = img.clone()
            else:
                img_array = np.array(img)
                h, w = img_array.shape[:2]
                img_copy = img_array.copy()
            
            num_holes = random.randint(1, self.max_holes)
            
            for _ in range(num_holes):
                hole_h = int(h * random.uniform(*self.hole_size_range))
                hole_w = int(w * random.uniform(*self.hole_size_range))
                
                y = random.randint(0, max(1, h - hole_h))
                x = random.randint(0, max(1, w - hole_w))
                
                if isinstance(img, torch.Tensor):
                    img_copy[:, y:y+hole_h, x:x+hole_w] = 0
                else:
                    img_copy[y:y+hole_h, x:x+hole_w] = 0
            
            if isinstance(img, torch.Tensor):
                return img_copy
            else:
                return Image.fromarray(img_copy)
        
        return img


class CarTransforms(BaseTransforms):
    def __init__(self, img_size: int):
        super().__init__()
        self.img_size = img_size
        self.size = (img_size, img_size) if isinstance(img_size, int) else img_size

    def train_transform(self):
        return transforms.Compose([
            # 1단계: 원본 비율 보존하며 약간 확대 (30% 확률)
            transforms.RandomApply([
                transforms.Resize(max(self.size) + 100, interpolation=transforms.InterpolationMode.BILINEAR)
            ], p=0.3),
            
            # 2단계: 차량 부위별 다이나믹 크롭 전략 (80% 확률)
            transforms.RandomApply([
                RandomCarCrop(
                    size=self.size,
                    strategies=["full", "front", "bottom", "side", "rear"],
                    probabilities=[0.4, 0.15, 0.15, 0.15, 0.15]
                )
            ], p=0.8),
            
            # 3단계: 추가 세부 영역 강조 (20% 확률)
            transforms.RandomApply([
                transforms.RandomChoice([
                    CarSpecificCrop(self.size, "detail"),    # 그릴 패턴 강조
                    CarSpecificCrop(self.size, "bottom"),    # 휠 디자인 강조
                    RandomResizedCropWithRatio(self.size, scale=(0.6, 0.9), ratio=(0.8, 1.2)),  # 배지 영역
                ])
            ], p=0.2),
            
            # 4단계: 파노라마 스타일 크롭 (15% 확률)
            transforms.RandomApply([
                CarSpecificCrop(self.size, "panorama")
            ], p=0.15),
            
            # 5단계: 기하학적 변환 (40% 확률)
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        fill=0
                    ),
                    transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
                ])
            ], p=0.4),
            
            # 6단계: 좌우 반전 (50% 확률)
            transforms.RandomHorizontalFlip(p=0.5),
            
            # 7단계: 조명 및 색상 변화 (60% 확률)
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.ColorJitter(brightness=0.25, contrast=0.25),
                    transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
                ])
            ], p=0.6),
            
            # 8단계: 세부 특징 강조 (30% 확률)
            EnhancedSharpen(alpha_range=(0.1, 0.3), p=0.3),
            
            # 9단계: 현실적인 노이즈 (25% 확률)
            RandomNoise(var_range=(10.0, 30.0), p=0.25),
            
            # 10단계: 부분 가려짐 (20% 확률)
            RandomCoarseDropout(max_holes=3, hole_size_range=(0.05, 0.15), p=0.2),
            
            # 11단계: 최종 리사이즈 및 정규화
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def val_transform(self):
        return transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def test_transform(self):
        return self.val_transform()