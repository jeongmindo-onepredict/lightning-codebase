import torch
import torch.nn as nn


class FCEncoder(nn.Module):
    def __init__(self, params):
        super(FCEncoder, self).__init__()

        conv_layers = params.get("conv_layers")

        self.feature_extractor = nn.Sequential()
        in_channels = params.get("input_size")
        for i, conv in enumerate(conv_layers):
            self.feature_extractor.add_module(
                f"conv_{i+1}", nn.Linear(in_channels, conv["out_channels"])
            )

            self.feature_extractor.add_module(
                f"batchnorm_{i+1}", nn.LayerNorm(conv["out_channels"])
            )

            self.feature_extractor.add_module(f"relu_{i+1}", nn.ReLU())

            in_channels = conv["out_channels"]

        # Calculate the size of the flattened features after convolution and pooling
        self._to_linear = None
        self._get_conv_output_size(params)

    def _get_conv_output_size(self, params):
        # Create a dummy input to calculate the output size after CNN layers
        dummy_input = torch.zeros(2, params["input_size"])
        output = self.feature_extractor(dummy_input)
        # print(output.shape)
        self._to_linear = output.size(1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        return x

    def get_output_size(self):
        return self._to_linear
