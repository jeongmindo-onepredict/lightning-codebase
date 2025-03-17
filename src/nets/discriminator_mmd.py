import torch
import torch.nn as nn
from networks.reverse_layer_f import ReverseLayerF


class Discriminator_MMD(nn.Module):
    def __init__(self, input_size, params):
        super(Discriminator_MMD, self).__init__()

        dense_layers = params.get("dense_layers")
        num_classes = params.get("num_classes")

        self.discriminator = nn.Sequential()
        in_features = input_size
        for i, units in enumerate(dense_layers):
            self.discriminator.add_module(f"fc_{i+1}", nn.Linear(in_features, units))
            self.discriminator.add_module(f"relu_fc_{i+1}", nn.ReLU())
            in_features = units
        # MMD
        self.discriminator.add_module("output", nn.Linear(in_features, 1))
        self.discriminator.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x, alpha):
        reversed_input = ReverseLayerF.apply(x, alpha)
        x = self.discriminator(reversed_input)
        return x
