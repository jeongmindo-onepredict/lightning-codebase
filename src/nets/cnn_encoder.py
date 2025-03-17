import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, params):
        super(CNNEncoder, self).__init__()

        input_channels = params.get("input_channels", 1)
        conv_layers = params.get("conv_layers")
        adaptive_pooling_output_size = params.get(
            "adaptive_pooling_output_size"
        )  # Adaptive pooling output 크기

        self.feature_extractor = nn.Sequential()
        in_channels = input_channels
        for i, conv in enumerate(conv_layers):
            self.feature_extractor.add_module(
                f"conv_{i+1}",
                nn.Conv1d(
                    in_channels,
                    conv["out_channels"],
                    kernel_size=conv["kernel_size"],
                    stride=conv["stride"],
                    padding=conv["padding"],
                ),
            )

            self.feature_extractor.add_module(
                f"batchnorm_{i+1}", nn.BatchNorm1d(conv["out_channels"])
            )

            self.feature_extractor.add_module(f"relu_{i+1}", nn.ReLU())

            self.feature_extractor.add_module(
                f"pool_{i+1}", nn.MaxPool1d(kernel_size=2)
            )

            in_channels = conv["out_channels"]

        # Add an adaptive pooling layer at the end
        self.adaptive_pool = nn.AdaptiveAvgPool1d(adaptive_pooling_output_size)

        # Calculate the size of the flattened features after convolution and pooling
        self._to_linear = None
        self._get_conv_output_size(params)

    def _get_conv_output_size(self, params):
        # Create a dummy input to calculate the output size after CNN layers
        dummy_input = torch.zeros(1, params["input_channels"], params["input_size"])
        output = self.feature_extractor(dummy_input)
        # print(output.size())

        output = self.adaptive_pool(output)
        # print(output.size())

        self._to_linear = output.view(1, -1).size(1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_output_size(self):
        return self._to_linear
