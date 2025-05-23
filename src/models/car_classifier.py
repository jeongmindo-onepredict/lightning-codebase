import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import Accuracy
try:
    import wandb
finally:
    pass

class CarClassifier(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        margin_init: float = 0.5,
        margin_final: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        vis_per_batch: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        
        self.net = net
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.vis_per_batch = vis_per_batch
        
        # 사용자 정의 손실 함수 초기화
        self.criterion = loss_fn
        
        # 메트릭 초기화
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.net.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.net.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.net.num_classes)
        
    def forward(self, x):
        return self.net(x)
    
    def on_fit_start(self) -> None:
        # 손실 함수의 num_epochs 업데이트
        self.criterion.num_epochs = self.trainer.max_epochs
        
        # WandbLogger 확인 및 시각화 설정
        self.is_wandb = isinstance(self.logger, WandbLogger)
        self.vis_per_batch = self.vis_per_batch if self.is_wandb else 0

    def training_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        
        # 손실 계산
        loss = self.criterion(logits, labels)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)
        
        # 로깅
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        # 에폭이 끝날 때마다 마진 업데이트
        self.criterion.update_margin()
        self.log("pmd/margin", self.criterion.current_margin)
    
    def on_validation_epoch_start(self) -> None:
        # 시각화를 위한 wandb 테이블 초기화
        if self.vis_per_batch and self.is_wandb:
            self.val_table = wandb.Table(columns=["image", "true_label", "pred_label"])
    
    def validation_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        
        # 기본 cross entropy 손실 (검증에서는 마진 없이)
        loss = F.cross_entropy(logits, labels)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)
        
        # 로깅
        self.log_dict(
            {
                "val/loss": loss,
                "val/acc": acc,
            },
            on_epoch=True,
            prog_bar=True,
        )
        
        # 샘플 시각화
        if self.vis_per_batch and self.is_wandb:
            self.visualize_samples(img, labels, preds, batch_idx)
        
        return {"loss": loss, "preds": preds, "labels": labels}
    
    def visualize_samples(self, images, labels, preds, batch_idx):
        if batch_idx % 5 != 1:  # 처음 몇 개 배치만 시각화
            return
            
        # 데이터로더에서 클래스 이름 가져오기
        if hasattr(self.trainer.datamodule.train_dataset, "classes"):
            class_names = self.trainer.datamodule.train_dataset.classes
        else:
            class_names = [str(i) for i in range(self.net.num_classes)]
            
        # ImageNet 정규화 값을 사용하여 역정규화 수행
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
            
        # 각 샘플에 대해 시각화
        for i in range(min(len(images), self.vis_per_batch)):
            # 이미지 역정규화
            img = images[i].clone()  # 원본 이미지 복사
            img = img * std + mean   # 역정규화
            
            # [0, 1] 범위로 클리핑 후 [0, 255] 범위로 변환
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype("uint8")
            
            true_label = class_names[labels[i].item()]
            pred_label = class_names[preds[i].item()]
            
            self.val_table.add_data(
                wandb.Image(img),
                true_label,
                pred_label
            )

    def on_validation_epoch_end(self) -> None:
        # Wandb 테이블 로깅
        if self.vis_per_batch and self.is_wandb and hasattr(self, "val_table"):
            self.logger.experiment.log({"val/samples": self.val_table})
    
    def test_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, labels)
        
        # 로깅
        self.log("test/acc", acc, on_epoch=True)
        
        return {"preds": preds, "labels": labels}
    
    def on_predict_start(self) -> None:
        """예측 시작 시 결과를 저장할 리스트 초기화"""
        self.predict_results = []
        
    def predict_step(self, batch, batch_idx):
        """예측 단계 - 라벨이 없는 데이터에 대한 예측"""
        img, filenames = batch  # predict_dataset에서 (이미지, 파일명) 반환
        logits = self(img)
        
        # softmax 확률 계산
        probs = F.softmax(logits, dim=1)
        
        # 배치 결과를 저장
        for i, filename in enumerate(filenames):
            # 파일명에서 확장자 제거하여 ID 생성
            file_id = filename.rsplit('.', 1)[0]  # .jpg, .png 등 확장자 제거
            
            # 각 클래스별 확률값과 함께 저장
            result = {
                'ID': file_id,
                'probabilities': probs[i].cpu().numpy()
            }
            self.predict_results.append(result)
        
        return {
            "probabilities": probs,
            "filenames": filenames
        }
    
    def on_predict_end(self) -> None:
        """예측 종료 시 CSV 파일로 저장"""
        import pandas as pd
        import numpy as np
        
        if not hasattr(self, 'predict_results') or not self.predict_results:
            print("No prediction results to save.")
            return
            
        # 클래스 이름 가져오기
        class_names = self.trainer.datamodule.predict_dataset.classes
        
        # DataFrame 생성을 위한 데이터 준비
        data = {'ID': []}
        for class_name in class_names:
            data[class_name] = []
        
        # 결과 데이터를 DataFrame 형식으로 변환
        for result in self.predict_results:
            data['ID'].append(result['ID'])
            probs = result['probabilities']
            
            for i, class_name in enumerate(class_names):
                data[class_name].append(probs[i])
        
        # DataFrame 생성 및 CSV 저장
        df = pd.DataFrame(data)
        
        # ID 기준으로 정렬 (파일명 순서)
        df = df.sort_values('ID').reset_index(drop=True)
        
        # CSV 파일로 저장
        csv_path = 'predictions.csv'
        df.to_csv(csv_path, index=False)
        
        # 메모리 정리
        del self.predict_results
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }