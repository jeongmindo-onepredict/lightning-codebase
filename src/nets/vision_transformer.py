import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange


# ------------------------- VisionTransformer
class MHA(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()

        self.n_heads = n_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.scale = torch.sqrt(torch.tensor(hidden_dim / n_heads))

        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

        if self.fc_q.bias is not None:
            nn.init.constant_(self.fc_q.bias, 0)
        if self.fc_k.bias is not None:
            nn.init.constant_(self.fc_k.bias, 0)
        if self.fc_v.bias is not None:
            nn.init.constant_(self.fc_v.bias, 0)
        if self.fc_o.bias is not None:
            nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, x):

        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        Q = rearrange(
            Q,
            "batchsize imgnum (head dim) -> batchsize head imgnum dim",
            head=self.n_heads,
        )  # batchsize imgnum dim -> batchsize head imgnum dim
        K = rearrange(
            K,
            "batchsize imgnum (head dim) -> batchsize head imgnum dim",
            head=self.n_heads,
        )
        V = rearrange(
            V,
            "batchsize imgnum (head dim) -> batchsize head imgnum dim",
            head=self.n_heads,
        )

        attention_score = (
            Q @ K.transpose(-2, -1) / self.scale
        )  # batchsize head imgnum imgnum imgnum dim

        attention_weights = torch.softmax(
            attention_score, dim=-1
        )  # batchsize head imgnum imgnum imgnum dim

        attention = attention_weights @ V  # batchsize head imgnum dim

        x = rearrange(
            attention, "batchsize head imgnum dim -> batchsize imgnum (head dim)"
        )  # batchsize head imgnum dim -> batch_size imgnum dim
        x = self.fc_o(x)  # batchsize imgnum dim

        return x, attention_weights


class TransformerFeedForward(nn.Module):
    def __init__(self, hidden_dim, d_ff, drop_p):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, d_ff),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(d_ff, hidden_dim),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, d_ff, n_heads, drop_p):
        super().__init__()

        self.self_atten_LN = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_atten = MHA(hidden_dim, n_heads)

        self.FF_LN = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.FF = TransformerFeedForward(hidden_dim, d_ff, drop_p)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):

        residual = self.self_atten_LN(x)
        residual, atten_enc = self.self_atten(residual)
        residual = self.dropout(residual)
        x = x + residual

        residual = self.FF_LN(x)
        residual = self.FF(residual)
        residual = self.dropout(residual)
        x = x + residual

        return x, atten_enc


class TransformerEncoder(nn.Module):
    def __init__(self, seq_length, n_layers, hidden_dim, d_ff, n_heads, drop_p, params):
        super().__init__()

        self.pos_embedding = nn.Parameter(0.02 * torch.randn(seq_length, hidden_dim))
        self.dropout = nn.Dropout(drop_p)
        # self.layers = nn.ModuleList([
        #     ('layer_{}'.format(i), TransformerEncoderLayer(hidden_dim, d_ff, n_heads, drop_p))
        #     for i in range(n_layers)
        # ])
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, d_ff, n_heads, drop_p)
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.params = params

    def forward(self, src, atten_map_save=False):
        x = src + self.pos_embedding.expand_as(src)  # batch_size imgnum dim
        x = self.dropout(x)

        atten_encs = torch.tensor([]).to(self.params.get("device"))
        for layer in self.layers:
            x, atten_enc = layer(x)
            if atten_map_save is True:
                atten_encs = torch.cat([atten_encs, atten_enc[0].unsqueeze(0)], dim=0)

        x = x[:, 0, :]  # CLS의 출력 임베딩 벡터. shape = batch_size dim
        x = self.ln(x)

        return x  # , atten_encs # 실제 계산값들 확인 필요할때 사용


class VisionTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()

        image_size = params.get("image_size")
        patch_size = params.get("patch_size")
        n_layers = params.get("n_layers")
        hidden_dim = params.get("hidden_dim")
        d_ff = params.get("d_ff")
        n_heads = params.get("n_heads")
        representation_size = params.get("representation_size", 128)
        drop_p = params.get("drop_p", 0.0)
        num_classes = params.get("dense_layers")[
            0
        ]  # 여기서의 num_classes는 latent space 차원을 의미하기 때문에 dense_layers와 동일하게 맞추면 됨

        self.hidden_dim = hidden_dim
        self.output_size = params.get("dense_layers")[0]

        seq_length = (image_size[0] // patch_size[0]) + 1  # +1 은? cls 토큰!

        self.class_token = nn.Parameter(torch.zeros(hidden_dim))
        self.input_embedding = nn.Conv2d(
            3, hidden_dim, patch_size, stride=patch_size[0]
        )
        self.encoder = TransformerEncoder(
            seq_length, n_layers, hidden_dim, d_ff, n_heads, drop_p, params
        )

        heads_layers = []
        if representation_size is None:  # representation_size is None 은 fine-tuning
            self.head = nn.Linear(hidden_dim, hidden_dim)  # fine-tune 할 땐 이렇게
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, representation_size),  # pre-training 할 땐 MLP
                nn.Tanh(),
                nn.Linear(representation_size, hidden_dim),
            )

        # conv weight 초기화.
        fan_in = (
            self.input_embedding.in_channels
            * self.input_embedding.kernel_size[0]
            * self.input_embedding.kernel_size[1]
        )
        nn.init.trunc_normal_(self.input_embedding.weight, std=math.sqrt(1 / fan_in))
        if self.input_embedding.bias is not None:
            nn.init.zeros_(self.input_embedding.bias)
        # Linear weight 초기화
        if representation_size is None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:  # pre-training 때
            fan_in = self.head[0].in_features
            nn.init.trunc_normal_(self.head[0].weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.head[0].bias)

    def get_output_size(self):
        return self.output_size

    def forward(self, x):
        x = self.input_embedding(x)
        x = rearrange(
            x, "batchsize dim imgheight imgwidth -> batchsize (imgheight imgwidth) dim"
        )

        batch_class_token = self.class_token.expand(x.shape[0], 1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        enc_out = self.encoder(x)

        x = self.head(enc_out)

        return x
