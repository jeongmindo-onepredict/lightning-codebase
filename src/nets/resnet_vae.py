# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from torchvision import models as tvision_mdoels
# import numpy as np


# # ------------------------- typevalidation 클래스
# class ResNetVAE(nn.Module):  # (코드 리팩토링 필요)
#     """
#     사전 훈련된 ResNet-18를 Encoder로 활용하여 구성한 VAE(Variational AutoEncoder) 모델 아키텍처
#     """

#     def __init__(
#         self,
#         freeze=True,
#         fc_hidden1=1024,
#         fc_hidden2=768,
#         latent_embed_dim=16,
#     ):
#         """
#         모델 아키텍처 내부 레이어 설정

#         Args:
#             freeze (bool, optional): pre-trained ResNet 모델의 파라미터 freeze 여부 . Defaults to True.
#             fc_hidden1 (int, optional): linear layer1 hidden size_. Defaults to 1024.
#             fc_hidden2 (int, optional): linear layer1 hidden size_. Defaults to 768.
#             latent_embed_dim (int, optional): . Defaults to 16.
#         """
#         super(ResNetVAE, self).__init__()
#         self.fc_hidden1, self.fc_hidden2, self.latent_embed_dim = (
#             fc_hidden1,
#             fc_hidden2,
#             latent_embed_dim,
#         )

#         # encoding components
#         resnet = tvision_mdoels.resnet18(
#             weights=tvision_mdoels.ResNet18_Weights.IMAGENET1K_V1
#         )
#         modules = list(resnet.children())[:-1]  # 최종 classification layer는 제외
#         self.resnet = nn.Sequential(*modules)

#         if freeze:
#             for param in self.resnet.parameters():
#                 param.requires_grad = False

#         self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
#         self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
#         self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
#         self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)

#         # latent vector layer 정의: mu & logvar
#         self.fc3_mu = nn.Linear(self.fc_hidden2, self.latent_embed_dim)
#         self.fc3_logvar = nn.Linear(self.fc_hidden2, self.latent_embed_dim)

#         # latent  -> hidden state layer
#         self.fc4 = nn.Linear(self.latent_embed_dim, self.fc_hidden2)
#         self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
#         self.fc5 = nn.Linear(self.fc_hidden2, 64 * 8 * 16)
#         self.fc_bn5 = nn.BatchNorm1d(64 * 8 * 16)
#         self.relu = nn.ReLU()

#         # Decoder
#         self.convTrans6 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=64,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 output_padding=1,
#             ),
#             nn.BatchNorm2d(32, momentum=0.01),
#             nn.ReLU(),
#         )
#         self.convTrans7 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=32,
#                 out_channels=16,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 output_padding=1,
#             ),
#             nn.BatchNorm2d(16, momentum=0.01),
#             nn.ReLU(),
#         )

#         self.convTrans8 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=16,
#                 out_channels=8,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 output_padding=1,
#             ),
#             nn.BatchNorm2d(8, momentum=0.01),
#             nn.ReLU(),  # y = (y1, y2, y3) \in [0 ,1]^3
#         )
#         self.convTrans9 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=8,
#                 out_channels=8,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 output_padding=1,
#             ),
#             nn.BatchNorm2d(8, momentum=0.01),
#             nn.ReLU(),
#         )

#         self.convTrans10 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=8,
#                 out_channels=3,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 output_padding=1,
#             ),
#             nn.BatchNorm2d(3, momentum=0.01),
#             nn.Sigmoid(),
#         )

#     def encode(self, x):
#         x = self.resnet(x)  # ResNet
#         x = x.view(x.size(0), -1)  # flatten output of conv
#         # FC layers
#         x = self.bn1(self.fc1(x))
#         x = self.relu(x)
#         x = self.bn2(self.fc2(x))
#         x = self.relu(x)
#         mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             # eps = Variable(std.data.new(std.size()).normal_()) # Variable : pytorch 예전 버전에서 tensor 래핑 시 사용
#             eps = std.data.new(std.size()).normal_()
#             z = eps.mul(std).add_(mu)
#             return z
#         else:
#             return mu

#     def decode(self, z):
#         x = self.relu(self.fc_bn4(self.fc4(z)))
#         x = self.relu(self.fc_bn5(self.fc5(x))).view(
#             -1, 64, 8, 16
#         )  # encoder output shape
#         x = self.convTrans6(x)
#         x = self.convTrans7(x)
#         x = self.convTrans8(x)
#         x = self.convTrans9(x)
#         x = self.convTrans10(x)
#         return x

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         x_reconst = self.decode(z)

#         return x_reconst, mu, logvar


# class LatentFeatureEmbeddingInference:
#     def __init__(self, model, device):
#         self.model = model
#         self.device = device
#         self.model.to(self.device)

#     @torch.inference_mode
#     def __call__(self, batch):  # batch:pd.DataFrame
#         self.model.eval()

#         x = torch.Tensor(np.vstack(batch["features"])).to(device=self.device)
#         x_, mu, logvar = self.model(x)
#         mu = mu.detach().cpu().numpy()

#         return {
#             "acq_time": batch.acq_time,
#             "mu": mu,
#         }


# class TypeClassifier(nn.Module):
#     def __init__(self, n_classes=2):
#         super(TypeClassifier, self).__init__()
#         self.model = torchvision.mobilenet_v3_small(weights="DEFAULT")
#         self.model.classifier[-1] = nn.Linear(
#             self.model.classifier[-1].in_features, n_classes
#         )

#     def forward(self, x):
#         return self.model(x)


# class TypeClassifierInference:
#     def __init__(self, model, device):
#         self.model = model
#         self.device = device

#     @torch.inference_mode
#     def __call__(self, batch):
#         self.model.eval()
#         x = torch.Tensor(np.vstack(batch["features"])).to(device=self.device)
#         y = self.model(x)
#         y = F.softmax(y, dim=-1).detach().cpu().numpy()
#         return {
#             "acq_time": batch.acq_time,
#             "motor_id": batch.motor_id,
#             "is_valid": batch.is_valid,
#             "prob": y[:, -1],
#         }
