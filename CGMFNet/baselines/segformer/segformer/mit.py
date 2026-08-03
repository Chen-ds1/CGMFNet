import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        features = self.proj(inputs)
        _, _, height, width = features.shape
        features = features.flatten(2).transpose(1, 2)
        return self.norm(features), height, width


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        sr_ratio=1,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads.')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inputs, height, width):
        batch_size, tokens, channels = inputs.shape
        queries = self.q(inputs)
        queries = queries.reshape(
            batch_size,
            tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            reduced = inputs.permute(0, 2, 1).reshape(
                batch_size,
                channels,
                height,
                width,
            )
            reduced = self.sr(reduced)
            reduced = reduced.reshape(batch_size, channels, -1).permute(0, 2, 1)
            reduced = self.sr_norm(reduced)
            keys_values = self.kv(reduced)
            keys_values = keys_values.reshape(
                batch_size,
                -1,
                2,
                self.num_heads,
                self.head_dim,
            ).permute(2, 0, 3, 1, 4)
        else:
            keys_values = self.kv(inputs)
            keys_values = keys_values.reshape(
                batch_size,
                tokens,
                2,
                self.num_heads,
                self.head_dim,
            ).permute(2, 0, 3, 1, 4)

        keys, values = keys_values[0], keys_values[1]
        attention = (queries @ keys.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(F.softmax(attention, dim=-1))
        output = (attention @ values).transpose(1, 2).reshape(
            batch_size,
            tokens,
            channels,
        )
        return self.proj_drop(self.proj(output))


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.conv1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
        )
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, height, width):
        batch_size, tokens, channels = inputs.shape
        residual = inputs.permute(0, 2, 1).reshape(
            batch_size,
            channels,
            height,
            width,
        )
        residual = self.conv1(residual)
        residual = residual.reshape(batch_size, channels, tokens).permute(0, 2, 1)

        output = self.fc1(inputs)
        output = output.permute(0, 2, 1).reshape(batch_size, -1, height, width)
        output = self.dwconv(output)
        output = output.reshape(batch_size, -1, tokens).permute(0, 2, 1)
        output = self.drop(self.act(output))
        output = self.drop(self.fc2(output))
        return output + residual


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        sr_ratio=1,
        mlp_ratio=4,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim,
            num_heads,
            sr_ratio,
            qkv_bias,
            attn_drop,
            proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, dim * mlp_ratio, proj_drop)

    def forward(self, inputs, height, width):
        output = inputs + self.drop_path(
            self.attn(self.norm1(inputs), height, width)
        )
        return output + self.drop_path(self.ffn(self.norm2(output), height, width))


class MiT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(8, 8, 4, 4),
        drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.num_stages = len(depths)
        self.patch_embeddings = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        drop_path_rates = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        block_offset = 0

        for stage_index in range(self.num_stages):
            if stage_index == 0:
                patch_embedding = OverlapPatchEmbedding(
                    in_channels,
                    embed_dims[stage_index],
                    patch_size=7,
                    stride=4,
                    padding=3,
                )
            else:
                patch_embedding = OverlapPatchEmbedding(
                    embed_dims[stage_index - 1],
                    embed_dims[stage_index],
                    patch_size=3,
                    stride=2,
                    padding=1,
                )
            self.patch_embeddings.append(patch_embedding)

            stage_blocks = nn.ModuleList()
            for depth_index in range(depths[stage_index]):
                stage_blocks.append(
                    TransformerBlock(
                        dim=embed_dims[stage_index],
                        num_heads=num_heads[stage_index],
                        sr_ratio=sr_ratios[stage_index],
                        mlp_ratio=mlp_ratios[stage_index],
                        qkv_bias=True,
                        attn_drop=drop_rate,
                        proj_drop=drop_rate,
                        drop_path=drop_path_rates[block_offset + depth_index],
                    )
                )
            self.blocks.append(stage_blocks)
            block_offset += depths[stage_index]
            self.norms.append(nn.LayerNorm(embed_dims[stage_index]))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode='fan_out',
                nonlinearity='relu',
            )

    def forward(self, inputs):
        features = []
        output = inputs
        for stage_index in range(self.num_stages):
            output, height, width = self.patch_embeddings[stage_index](output)
            for block in self.blocks[stage_index]:
                output = block(output, height, width)
            output = self.norms[stage_index](output)
            output = output.permute(0, 2, 1).reshape(
                output.shape[0],
                -1,
                height,
                width,
            )
            features.append(output)
        return features


if __name__ == '__main__':
    network = MiT()
    network.eval()
    sample = torch.randn(1, 3, 240, 320)
    with torch.inference_mode():
        outputs = network(sample)
    for index, feature in enumerate(outputs, start=1):
        print(f'Stage {index}: {tuple(feature.shape)}')
