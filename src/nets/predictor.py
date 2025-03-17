import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, input_size, params):
        super(Predictor, self).__init__()

        dense_layers = params.get("dense_layers")
        num_classes = params.get("num_classes")

        self.predictor = nn.Sequential()
        in_features = input_size
        for i, units in enumerate(dense_layers):
            self.predictor.add_module(f"fc_{i+1}", nn.Linear(in_features, units))
            self.predictor.add_module(f"relu_fc_{i+1}", nn.ReLU())
            in_features = units
        self.predictor.add_module("output", nn.Linear(in_features, num_classes))

    def forward(self, x):
        x = self.predictor(x)
        return x
