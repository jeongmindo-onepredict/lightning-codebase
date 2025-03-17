import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- GradCAM
class GradCAM(nn.Module):
    def __init__(self, model, module, layer):
        super().__init__()
        self.model = model
        self.module = module
        self.layer = layer
        self.register_hooks()
        self.model.eval()

    def register_hooks(self):
        for modue_name, module in self.model._modules.items():
            if modue_name == self.module:
                for layer_name, module in module._modules.items():
                    if layer_name == self.layer:
                        module.register_forward_hook(self.forward_hook)
                        module.register_backward_hook(self.backward_hook)

    def forward(self, input, target_index):

        outs = self.model(input)

        outs = outs.squeeze()  # [1, num_classes]  --> [num_classes]

        if target_index is None:
            target_index = outs.argmax()

        outs[target_index].backward(retain_graph=True)  # gradient
        a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)
        out = torch.sum(a_k * self.forward_result, dim=0).cpu()  # feature map
        out = torch.relu(out) / torch.max(out)
        out = F.interpolate(
            out.unsqueeze(0).unsqueeze(0),  # (batch, channel, H, W)
            input.size()[-2:],
            mode="bilinear",
            align_corners=True,
        )  # 4D로 바꿈
        return out.cpu().detach().squeeze().numpy()

    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])
