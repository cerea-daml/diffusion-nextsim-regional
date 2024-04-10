#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/10/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable

# External modules
import torch.nn
from einops import rearrange
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel

from ddm_dynamical.layers import SinusoidalEmbedding, RandomFourierLayer

# Internal modules

main_logger = logging.getLogger(__name__)


backend_map = {
    "math": {
        "enable_math": True, "enable_flash": False,
        "enable_mem_efficient": False
    },
    "flash": {
        "enable_math": False, "enable_flash": True,
        "enable_mem_efficient": False
    },
    "efficient": {
        "enable_math": False, "enable_flash": False,
        "enable_mem_efficient": True
    },
    None: {
        "enable_math": True, "enable_flash": True,
        "enable_mem_efficient": True
    },
}


class SelfAttentionLayer(torch.nn.Module):
    """
    Self-Attention layer with implementation inspired by
    https://github.com/Stability-AI/generative-models/
    """
    def __init__(
            self,
            in_channels,
            n_embedding: int = 256,
            n_heads: int = 8,
            dropout_rate: float = 0.,
            attn_type: str = None
    ):
        super().__init__()
        self.embedding_layer = torch.nn.Linear(n_embedding, in_channels*2)
        self.norm = torch.nn.LayerNorm(
            in_channels, elementwise_affine=False
        )
        self.qkv_layer = torch.nn.Linear(in_channels, in_channels*3, bias=True)
        self.out_layer = torch.nn.Linear(in_channels, in_channels, bias=True)
        self.gamma = torch.nn.Parameter(torch.ones(in_channels)*1E-6)
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.attn_type = attn_type

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.embedding_layer(embedding).chunk(2, dim=-1)
        branch = self.norm(in_tensor) * (scale + 1) + shift
        qkv = self.qkv_layer(branch)
        b, l, khc = qkv.shape
        c = int(khc/self.n_heads/3)
        qkv = rearrange(
            qkv, "b l (k h c) -> k b h l c",
            b=b, l=l, h=self.n_heads, c=c, k=3
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            dropout_p=self.dropout_rate
        )
        attn_out = rearrange(
            attn_out, "b h l c -> b l (h c)",
            b=b, l=l, h=self.n_heads, c=c
        )
        return in_tensor + self.gamma * self.out_layer(attn_out)


class MLPLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            n_embedding: int = 256,
            mult: int = 1,
            dropout_rate: float = 0.,
    ):
        super().__init__()
        self.hidden_channels = in_channels * mult
        self.embedding_layer = torch.nn.Linear(n_embedding, in_channels*2)
        self.norm = torch.nn.LayerNorm(in_channels, elementwise_affine=False)
        self.branch_layer = torch.nn.Sequential(
            torch.nn.Linear(in_channels, self.hidden_channels),
            torch.nn.GELU(),
        )
        if dropout_rate > 0.:
            self.branch_layer.append(torch.nn.Dropout(p=dropout_rate))
        self.branch_layer.append(
            torch.nn.Linear(self.hidden_channels, in_channels)
        )
        self.gamma = torch.nn.Parameter(torch.ones(in_channels)*1E-6)

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.embedding_layer(embedding).chunk(2, dim=-1)
        branch = self.norm(in_tensor) * (scale + 1) + shift
        return (in_tensor + self.gamma * self.branch_layer(branch))


class DownScale(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_embedding: int = 256,
    ):
        super().__init__()
        self.embedding_layer = torch.nn.Conv2d(n_embedding, in_channels*2, 1)
        self.normalization = torch.nn.GroupNorm(1, in_channels, affine=False)
        self.down_layer = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=2, stride=2,
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.embedding_layer(embedding).chunk(2, dim=1)
        in_tensor = self.normalization(in_tensor) * (scale + 1) + shift
        return self.down_layer(in_tensor)


class UpScale(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_embedding: int = 256,
    ):
        super().__init__()
        self.embedding_layer = torch.nn.Conv2d(n_embedding, in_channels*2, 1)
        self.normalization = torch.nn.GroupNorm(1, in_channels, affine=False)
        self.upsampling = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_layer = torch.nn.Conv2d(
            in_channels+out_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            shortcut: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.embedding_layer(
            embedding[..., ::2, ::2]
        ).chunk(2, dim=1)
        normed_tensor = self.normalization(in_tensor) * (scale + 1) + shift
        upsampled_tensor = self.upsampling(normed_tensor)
        cat_tensor = torch.cat([upsampled_tensor, shortcut], dim=1)
        return self.conv_layer(cat_tensor)


class AttentionBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            n_embedding: int = 256,
            n_heads: int = 8,
            mult: int = 1,
            dropout_attn: float = 0.,
            dropout_mlp: float = 0.,
            attn_type: str = None
    ):
        super().__init__()
        self.attention = SelfAttentionLayer(
            in_channels=in_channels,
            n_embedding=n_embedding,
            n_heads=n_heads,
            dropout_rate=dropout_attn,
            attn_type=attn_type
        )
        self.mlp = MLPLayer(
            in_channels=in_channels,
            n_embedding=n_embedding,
            mult=mult,
            dropout_rate=dropout_mlp
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = in_tensor.shape
        in_tensor = rearrange(
            in_tensor, "b c h w -> b (h w) c",
            b=b, c=c, h=h, w=w
        )
        embedding = rearrange(
            embedding, "b n h w -> b (h w) n",
            b=b, h=h, w=w, n=embedding.size(1)
        )
        out_tensor = self.attention(in_tensor, embedding)
        out_tensor = self.mlp(out_tensor, embedding)
        return rearrange(out_tensor, "b (h w) c -> b c h w", b=b, c=c, h=h, w=w)


class ConvNeXtBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            n_embedding: int = 256,
            mult: int = 1,
    ):
        super().__init__()
        self.embedding_layer = torch.nn.Conv2d(n_embedding, in_channels*2, 1)
        self.spatial_conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=7, bias=True,
            groups=in_channels, padding=3
        )
        self.norm_layer = torch.nn.GroupNorm(
            1, in_channels, affine=False
        )
        self.hidden_channels = int(mult * in_channels)
        self.mix_layer = torch.nn.Sequential(
            torch.nn.Linear(in_channels, self.hidden_channels, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_channels, in_channels, bias=True),
        )
        self.gamma = torch.nn.Parameter(torch.ones(in_channels, 1, 1) * 1E-6)

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = in_tensor.shape
        scale, shift = self.embedding_layer(
            embedding
        ).chunk(2, dim=1)
        branch = self.spatial_conv(in_tensor)
        branch = self.norm_layer(branch) * (1 + scale) + shift
        branch = rearrange(branch,
            "b c h w -> b (h w) c",
            b=b, c=c, h=h, w=w
        )
        return in_tensor + self.gamma * rearrange(
            self.mix_layer(branch),
            "b (h w) c -> b c h w",
            b=b, c=c, h=h, w=w
        )


class UViT(torch.nn.Module):
    def __init__(
            self,
            n_input: int = 11,
            n_output: int = 5,
            n_features: int = 64,
            n_bottleneck: int = 3,
            n_down_blocks: Iterable[int] = (3, 3, 3),
            n_up_blocks: Iterable[int] = (3, 3, 3),
            channel_mul: Iterable[int] = (1, 2, 4),
            n_embedding: int = 256,
            n_mesh_hidden: int = 256,
            n_augment_hidden: int = 256,
            n_time_hidden: int = 256,
            n_heads: int = 8,
            mult: int = 1,
            dropout_attn: float = 0.,
            dropout_mlp: float = 0.,
            attn_type: str = None,
            *args, **kwargs
    ):
        super().__init__()
        self.n_depth = len(n_down_blocks)
        self.n_embedding = n_embedding
        if n_time_hidden > 0:
            self.time_embedder = torch.nn.Sequential(
                SinusoidalEmbedding(n_time_hidden),
                torch.nn.Linear(n_time_hidden, n_embedding),
                torch.nn.GELU(),
                torch.nn.Linear(n_embedding, n_embedding),
            )
        else:
            self.register_buffer("time_embedder", None)
        if n_augment_hidden > 0:
            self.augment_embedder = torch.nn.Sequential(
                torch.nn.Linear(
                    n_augment_hidden, n_embedding, bias=True
                ),
            )
        else:
            self.register_buffer("augment_embedder", None)
        if n_mesh_hidden > 0:
            self.mesh_embedder = torch.nn.Sequential(
                RandomFourierLayer(in_features=3, n_neurons=n_mesh_hidden),
                torch.nn.Linear(n_mesh_hidden, n_embedding),
                torch.nn.GELU(),
                torch.nn.Linear(n_embedding, n_embedding),
            )
        else:
            self.register_buffer("mesh_embedder", None)
        self.embedding_activation = torch.nn.GELU()
        self.in_encoder = torch.nn.Conv2d(n_input, n_features, 1)
        self.down_blocks = torch.nn.ModuleList()
        channel_mul = [1] + list(channel_mul)
        out_features = n_features
        for k, n_blocks in enumerate(n_down_blocks):
            curr_features = int(n_features*channel_mul[k])
            out_features = int(n_features*channel_mul[k+1])
            self.down_blocks.append(torch.nn.Module())
            self.down_blocks[-1].blocks = torch.nn.ModuleList([
                ConvNeXtBlock(
                    curr_features,
                    n_embedding=n_embedding,
                    mult=mult,
                )
                for _ in range(n_blocks)
            ])
            self.down_blocks[-1].downscale = DownScale(
                curr_features, out_features, n_embedding=n_embedding
            )
        self.bottleneck = torch.nn.ModuleList([
            AttentionBlock(
                out_features,
                n_embedding=n_embedding,
                n_heads=n_heads,
                mult=mult,
                dropout_attn=dropout_attn,
                dropout_mlp=dropout_mlp,
                attn_type=attn_type
            )
            for _ in range(n_bottleneck)
        ])
        self.up_blocks = torch.nn.ModuleList()
        for k, n_blocks in enumerate(n_up_blocks):
            curr_features = int(n_features*channel_mul[self.n_depth-k])
            out_features = int(n_features*channel_mul[self.n_depth-k-1])
            self.up_blocks.append(torch.nn.Module())
            self.up_blocks[-1].upscale = UpScale(
                curr_features, out_features, n_embedding=n_embedding
            )
            self.up_blocks[-1].blocks = torch.nn.ModuleList([
                ConvNeXtBlock(
                    out_features,
                    n_embedding=n_embedding,
                    mult=mult,
                )
                for _ in range(n_blocks)
            ])
        self.out_decoder = torch.nn.Sequential(
            torch.nn.GroupNorm(1, n_features, affine=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                n_features, n_output, kernel_size=1, padding=0
            )
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            **to_embed
    ) -> torch.Tensor:
        embedding = torch.zeros(
            in_tensor.size(0), self.n_embedding, *in_tensor.shape[-2:],
            device=in_tensor.device, layout=in_tensor.layout,
            dtype=in_tensor.dtype
        )
        if self.time_embedder is not None:
            embedding.add_(
                self.time_embedder(
                    to_embed["normalized_gamma"]*1000.
                )[..., None, None]
            )
        if self.augment_embedder is not None:
            embedding.add_(
                self.augment_embedder(to_embed["labels"])[..., None, None]
            )
        if self.mesh_embedder is not None:
            embedding.add_(
                self.mesh_embedder(
                    to_embed["mesh"].movedim(-3, -1)
                ).movedim(-1, -3)
            )
        embedding = self.embedding_activation(embedding)
        features = self.in_encoder(in_tensor)
        down_tensors = []
        for k, block in enumerate(self.down_blocks):
            for b in block.blocks:
                features = b(
                    features,
                    embedding[..., ::2**k, ::2**k]
                )
            down_tensors.append(features)
            features = block.downscale(
                features,
                embedding[..., ::2**k, ::2**k]
            )
        for block in self.bottleneck:
            features = block(
                features,
                embedding[..., ::2**(k+1), ::2**(k+1)]
            )
        for k, block in enumerate(self.up_blocks):
            idx = self.n_depth-k-1
            features = block.upscale(
                features,
                down_tensors[idx],
                embedding[..., ::2 ** idx, ::2 ** idx]
            )
            for b in block.blocks:
                features = b(
                    features,
                    embedding[..., ::2 ** idx, ::2 ** idx]
                )
        output = self.out_decoder(features)
        return output
