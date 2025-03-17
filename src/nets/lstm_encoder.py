import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, params):
        super(LSTMEncoder, self).__init__()

        input_size = params.get("input_size")
        input_channels = params.get("input_channels", 1)
        lstm_hidden_size = params.get("lstm_hidden_size", 256)
        lstm_layers = params.get("lstm_layers", 2)
        adaptive_pooling_output_size = params.get("adaptive_pooling_output_size")

        # LSTM 레이어 정의
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool1d(adaptive_pooling_output_size)

        # Flatten할 크기 계산
        self._to_linear = None
        self._get_lstm_output_size(params)

    def _get_lstm_output_size(self, params):
        # LSTM을 거친 후의 출력 크기를 계산하기 위한 더미 입력
        dummy_input = torch.zeros(1, params["input_channels"], params["input_size"])
        output, _ = self.lstm(dummy_input)
        output = output.transpose(1, 2)  # adaptive pooling을 위해 shape 변경
        output = self.adaptive_pool(output)
        self._to_linear = output.view(1, -1).size(1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)  # Adaptive pooling을 위해 차원 조정
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_output_size(self):
        return self._to_linear
