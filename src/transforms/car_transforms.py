from torchvision import transforms
from .base import BaseTransforms


class CarTransforms(BaseTransforms):
    def __init__(self):
        super().__init__()

    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((300, 300)),  # 더 큰 이미지로 시작하여 디테일 보존
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # 더 공격적인 크롭으로 부분적 특징 학습
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),  # 회전 범위 증가
            transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # 확대/축소 및 이동 추가
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 원근감 변화 추가
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 색상 변화 강화
            transforms.RandomGrayscale(p=0.05),  # 간혹 흑백 이미지로 변환
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # 일부 영역 삭제로 특정 부분 의존도 감소
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.1),  # 흐림 효과 가끔 적용
        ])

    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def test_transform(self):
        return self.val_transform()
