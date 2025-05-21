from torchvision import transforms
from .base import BaseTransforms


class CarTransforms(BaseTransforms):
    def __init__(self, img_size=384):
        super().__init__()

        self.img_size = img_size

    def train_transform(self):
        return transforms.Compose([
            # PIL 이미지에 적용되는 변환
            transforms.Resize((self.img_size, self.img_size)),
            # transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            
            # PIL 이미지를 텐서로 변환
            transforms.ToTensor(),
            
            # 텐서에만 적용되는 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.1),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # ToTensor 이후에 위치
        ])

    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def test_transform(self):
        return self.val_transform()
