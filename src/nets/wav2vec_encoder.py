import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


## wav2vec
class Wav2VecEncoder(nn.Module):
    def __init__(self, params):
        super(Wav2VecEncoder, self).__init__()

        input_size = params.get("input_size")
        self.input_channels = params.get("input_channels", 1)

        lstm_layers = params.get("lstm_layers", 2)
        adaptive_pooling_output_size = params.get("adaptive_pooling_output_size")

        self.input_channels = params["input_channels"]
        self.input_size = params["input_size"]

        # pretrained_model 정의
        self.wav2vec2 = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        self.feature_extractor = self.wav2vec2.feature_extractor
        self.encoder = self.wav2vec2.encoder

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False
        # Adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool1d(adaptive_pooling_output_size)

        # Flatten할 크기 계산
        self._to_linear = None
        self._get_model_output_size(params)

    def _get_model_output_size(self, params):
        dummy_input = torch.zeros(1, params["input_channels"], params["input_size"])
        dummy_input = dummy_input.permute(0, 2, 1)
        dummy_input = dummy_input.reshape(dummy_input.size(0), -1)
        output, _ = self.feature_extractor(
            dummy_input, self.input_channels * self.input_size
        )
        output = self.encoder(output)
        output = output.transpose(1, 2)  # adaptive pooling을 위해 shape 변경
        output = self.adaptive_pool(output)
        self._to_linear = output.view(1, -1).size(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), -1)
        x, _ = self.feature_extractor(x, self.input_channels * self.input_size)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_output_size(self):
        return self._to_linear
