import torch
import torch.nn as nn
import numpy as np
from networks.vision_transformer import VisionTransformer


# ------------------------- VisionTransformer + CausalCNN
class CausalCNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, kernel_size=3):
        super(CausalCNN, self).__init__()

        self.feature_extractor = nn.Sequential()
        current_dim = input_dim

        for i in range(num_layers):
            next_dim = max(current_dim // 2, output_dim)  # 차원을 점진적으로 줄임
            self.feature_extractor.add_module(
                f"conv_{i+1}",
                nn.Conv1d(
                    in_channels=current_dim,
                    out_channels=next_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,  # 입력 길이 유지
                ),
            )
            self.feature_extractor.add_module(f"relu_{i+1}", nn.ReLU())
            current_dim = next_dim  # 현재 차원 업데이트

        # 최종 출력 채널을 1차원으로 압축
        self.feature_extractor.add_module(
            "final_conv",
            nn.Conv1d(
                in_channels=current_dim,
                out_channels=output_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        # (batch_size, seq_len(보통은 여기가 dim으로, 차원변화됨), hidden_dim(feature값 유지)) -> (batch_size, 1, hidden_dim))
        x = self.feature_extractor(x)
        x = x.squeeze(
            1
        )  # (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim(=output_dim))
        # fc-layer로 2차원으로 줄이기

        return x


# ------------------------- ViT + CausalCNN 통합 클래스
class ViT_CausalCNN_FCLayer(nn.Module):
    def __init__(self, params):
        super(ViT_CausalCNN_FCLayer, self).__init__()

        conv_layers = params.get("hidden_dim")
        fin_output_dim = params.get("dense_layers")[0]

        self.fc = nn.Linear(conv_layers, fin_output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


# ViT + CausalCNN + FC-layer 통합 클래스
class ViT_CausalCNN(nn.Module):
    def __init__(self, params):
        super(ViT_CausalCNN, self).__init__()

        # CausalCNN - params
        seq_len = params.get("seq_len")
        num_layers = params.get("num_layers")

        self.vit = VisionTransformer(params)
        self.causal_cnn = CausalCNN(
            input_dim=seq_len, output_dim=1, num_layers=num_layers
        )
        self.fc = ViT_CausalCNN_FCLayer(params)

        self._to_linear = None
        self._get_conv_output_size(params)

    def forward(self, x):

        batch_size, seq_len, channels, img_h, img_w = x.shape
        # print('1', x.shape)
        # ViT 처리 (batch_size * seq_len, channels, height, width)
        x = x.view(-1, channels, img_h, img_w)  # 병렬 ViT 처리
        # print('2', x.shape)
        x = self.vit(x)  # (batch_size * seq_len, hidden_dim)
        # print('3', x.shape)
        # (batch_size, seq_len, hidden_dim) 형태로 복원
        x = x.view(batch_size, seq_len, -1)
        # print('4', x.shape)
        # CausalCNN 처리
        # input = (batch_size, seq_len, hidden_dim) ->  (batch_size, hidden_dim)
        x = self.causal_cnn(x)  # (batch_size, hidden_dim)
        # print('5', x.shape)
        output = self.fc(x)  # (batch_size, 2)
        # print('6', output.shape)
        return output

    def _get_conv_output_size(self, params):
        # self._to_linear = params.get('hidden_dim')
        fin_output_dim = params.get("dense_layers")[0]

        self._to_linear = fin_output_dim

    def get_output_size(self):
        return self._to_linear
