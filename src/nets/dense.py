import torch
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, input_size):
        super(Dense, self).__init__()

        self.predictor = nn.Sequential()
        in_features = input_size
        self.predictor.add_module("output", nn.Linear(in_features, in_features))

    def forward(self, x):
        x = self.predictor(x)
        return x
