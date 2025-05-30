import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import Accuracy
import numpy as np
try:
    import wandb
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
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
        
        # Validation 결과 저장을 위한 리스트
        self.val_predictions = []
        self.val_references = []
        
    def forward(self, x):
        return self.net(x)
    
    def on_fit_start(self) -> None:
        # 손실 함수의 num_epochs 업데이트
        self.criterion.num_epochs = self.trainer.max_epochs
        
        # WandbLogger 확인 및 시각화 설정
        self.is_wandb = isinstance(self.logger, WandbLogger)
        self.vis_per_batch = self.vis_per_batch if self.is_wandb else 0
        
        # matplotlib 한글 폰트 설정 (한국어 클래스명 지원)
        try:
            plt.rcParams["font.family"] = "NanumGothic"
            plt.rcParams["axes.unicode_minus"] = False
        except:
            # 폰트 설정 실패 시 기본 설정 유지
            pass

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
        if self.vis_per_batch:
            self.val_table = wandb.Table(columns=["image", "true_label", "pred_label"])
        
        # Validation 결과 초기화
        self.val_predictions = []
        self.val_references = []
    
    def validation_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        
        # 기본 cross entropy 손실 (검증에서는 마진 없이)
        loss = F.cross_entropy(logits, labels)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)
        
        # 예측 결과 저장 (confusion matrix를 위해)
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_references.extend(labels.cpu().numpy())
        
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
        if self.vis_per_batch:
            self.visualize_samples(img, labels, preds, batch_idx)
        
        return {"loss": loss, "preds": preds, "labels": labels}
    
    def visualize_samples(self, images, labels, preds, batch_idx):
        if batch_idx % 5 != 1:  # 처음 몇 개 배치만 시각화
            return
            
        # 데이터로더에서 클래스 이름 가져오기
        if hasattr(self.trainer.datamodule.val_dataset, "classes"):
            class_names = self.trainer.datamodule.val_dataset.classes
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
        if self.vis_per_batch and hasattr(self, "val_table"):
            self.logger.experiment.log({"val/samples": self.val_table})
        
        # Confusion Matrix 생성 및 로깅
        if len(self.val_predictions) > 0:
            self.log_confusion_matrix()
    
    def log_confusion_matrix(self):
        """상위 60개 혼동 클래스에 대한 Confusion Matrix를 고해상도로 생성하고 W&B에 로깅"""
        try:
            # PIL 이미지 크기 제한 해제
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            
            # numpy 배열로 변환
            preds = np.array(self.val_predictions)
            refs = np.array(self.val_references)
            
            # 클래스 이름 가져오기
            if hasattr(self.trainer.datamodule.val_dataset, "classes"):
                labels = self.trainer.datamodule.val_dataset.classes
            else:
                labels = [f"Class_{i}" for i in range(self.net.num_classes)]
            
            # Confusion Matrix 생성
            cm = confusion_matrix(refs, preds)
            np.fill_diagonal(cm, 0)  # 정답 예측은 제거하여 혼동만 표시
            
            # Top-60 가장 혼동이 많은 클래스들 찾기
            top_n = min(60, len(labels))  # 최대 60개 또는 전체 클래스 수 중 작은 값
            misclassified_counts = cm.sum(axis=1)
            top_true_classes = np.argsort(misclassified_counts)[::-1][:top_n]
            
            # 각 혼동 클래스에 대해 가장 많이 혼동되는 예측 클래스 찾기
            top_confused_classes = set(top_true_classes)
            for cls in top_true_classes:
                most_confused_pred = np.argmax(cm[cls])
                top_confused_classes.add(most_confused_pred)
            
            # 서브 매트릭스 추출
            top_confused_classes = sorted(top_confused_classes)
            reduced_cm = cm[np.ix_(top_confused_classes, top_confused_classes)]
            reduced_labels = [labels[i] for i in top_confused_classes]
            
            # 클래스 이름이 너무 길면 축약
            def truncate_label(label, max_len=25):
                if len(label) > max_len:
                    return label[:max_len-3] + "..."
                return label
            
            truncated_labels = [truncate_label(label) for label in reduced_labels]
            
            # 고해상도 Confusion Matrix 플롯 생성
            plt.rcParams['font.family'] = 'NanumGothic'  # 한글 폰트 설정
            
            # 적절한 figure 크기와 DPI 설정
            fig = plt.figure(figsize=(25, 20), dpi=300)
            ax = fig.add_subplot(111)
            
            # Heatmap 생성 (텍스트 표시)
            sns.heatmap(
                reduced_cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=truncated_labels,
                yticklabels=truncated_labels,
                ax=ax,
                annot_kws={'size': 6},  # 텍스트 크기 조정
                cbar_kws={'shrink': 0.8},
                square=True
            )
            
            # 제목과 레이블 설정
            ax.set_title(f"Top-{top_n} Confused Classes (Validation) - Epoch {self.current_epoch}", 
                        fontsize=18, pad=20)
            ax.set_xlabel("Predicted Label", fontsize=14)
            ax.set_ylabel("True Label", fontsize=14)
            
            # 틱 라벨 설정 (회전각도 조정)
            ax.tick_params(axis='x', labelsize=8, rotation=45)
            ax.tick_params(axis='y', labelsize=8, rotation=0)
            
            # x축 라벨을 위쪽으로 이동
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            
            plt.tight_layout()
            
            # 로컬에 고해상도 이미지로 먼저 저장
            save_path = f"confusion_matrix_top60_epoch_{self.current_epoch}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.3)
            
            # W&B에는 파일 경로로 로깅 (메모리 효율적)
            try:
                self.logger.experiment.log({
                    "val/confusion_matrix_top60": wandb.Image(save_path, caption=f"Top-{top_n} Confused Classes - Epoch {self.current_epoch}")
                })
            except Exception as wandb_error:
                print(f"W&B logging failed: {wandb_error}")
                print("Confusion matrix saved locally only.")
            
            plt.close(fig)  # 메모리 절약을 위해 figure 명시적으로 닫기
            
            print(f"Top-{top_n} confusion matrix saved: {save_path}")
            print(f"Matrix shape: {reduced_cm.shape}")
            print(f"Selected classes: {len(top_confused_classes)}")
            print(f"Most confused classes: {[labels[i] for i in top_true_classes[:5]]}")
                
        except Exception as e:
            print(f"Error logging confusion matrix: {e}")
            import traceback
            traceback.print_exc()
    
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
        """예측 단계 - TTA(Test Time Augmentation) 적용"""        
        img, filenames = batch  # predict_dataset에서 (이미지, 파일명) 반환
        
        all_predictions = []
        
        # 원본 이미지 예측
        with torch.no_grad():
            logits = self(img)
            probs = F.softmax(logits, dim=1)
            all_predictions.append(probs)
        
            img_flipped = torch.flip(img, dims=[-1])  # 가로 뒤집기
            logits_flipped = self(img_flipped)
            probs_flipped = F.softmax(logits_flipped, dim=1)
            all_predictions.append(probs_flipped)
        
        # 모든 예측 결과의 평균 계산
        ensemble_probs = torch.stack(all_predictions).mean(dim=0)
        
        # 배치 결과를 저장
        for i, filename in enumerate(filenames):
            # 파일명에서 확장자 제거하여 ID 생성
            file_id = filename.rsplit('.', 1)[0]  # .jpg, .png 등 확장자 제거
            
            # 각 클래스별 확률값과 함께 저장
            result = {
                'ID': file_id,
                'probabilities': ensemble_probs[i].cpu().numpy()
            }
            self.predict_results.append(result)
        
        return {
            "probabilities": ensemble_probs,
            "filenames": filenames
        }

    def on_predict_end(self) -> None:
        """예측 종료 시 CSV 파일로 저장"""
        import pandas as pd
        import numpy as np
        
        if not hasattr(self, 'predict_results') or not self.predict_results:
            print("No prediction results to save.")
            return
            
        # 기존 클래스 이름 가져오기
        original_class_names = self.trainer.datamodule.predict_dataset.classes
        
        # 추가할 클래스들
        additional_classes = [
            'K5_하이브리드_3세대_2020_2023',
            '디_올뉴니로_2022_2025',
            '718_박스터_2017_2024'
        ]
        
        # 모든 클래스 이름 합치기
        all_class_names = original_class_names + additional_classes
        
        # DataFrame 생성을 위한 데이터 준비
        data = {'ID': []}
        for class_name in all_class_names:
            data[class_name] = []
        
        # 결과 데이터를 DataFrame 형식으로 변환
        for result in self.predict_results:
            data['ID'].append(result['ID'])
            probs = result['probabilities']
            
            # 기존 클래스들의 확률값 추가
            for i, class_name in enumerate(original_class_names):
                data[class_name].append(probs[i])
            
            # 추가 클래스들의 확률값을 0.0으로 설정
            for class_name in additional_classes:
                data[class_name].append(0.0)
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # ID 기준으로 정렬 (파일명 순서)
        df = df.sort_values('ID').reset_index(drop=True)
        
        # 컬럼 순서 정렬 (ID는 첫 번째, 나머지는 알파벳 순)
        class_columns = sorted([col for col in df.columns if col != 'ID'])
        df = df[['ID'] + class_columns]
        
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