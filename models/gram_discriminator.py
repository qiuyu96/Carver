# python3.8
"""Contains the implementation of discriminator described in GRAM.

Paper: https://arxiv.org/pdf/2112.08867.pdf

Official PyTorch implementation: https://github.com/microsoft/GRAM
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class GRAMDiscriminator(nn.Module):

    def __init__(self,
                 resolution,
                 latent_dim=256,
                 label_dim=0,
                 embedding_dim=256,
                 normalize_embedding=True,
                 **kwargs):  # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.normalize_embedding = normalize_embedding
        # self.register_buffer('lod', torch.zeros(()))
        self.use_embedding = label_dim > 0 and embedding_dim > 0
        if self.use_embedding > 0:
            self.class_embedding = EqualLinear(label_dim,
                                               embedding_dim,
                                               bias=True,
                                               bias_init=0,
                                               lr_mul=1)
            self.norm = PixelNormLayer(dim=1, eps=1e-8)

        self.layers = nn.ModuleList([
            ResidualCCBlock(32, 64),  # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128),  # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256),  # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400),  # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400),  # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400),  # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400),  # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList([
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
        ])
        self.score_conv = nn.Conv2d(
            400, embedding_dim if self.use_embedding else max(label_dim, 3), 2)

        self.img_size_to_layer = {
            2: 7,
            4: 6,
            8: 5,
            16: 4,
            32: 3,
            64: 2,
            128: 1,
            256: 0
        }

    def forward(self,
                input,
                label=None,
                options=None,
                alpha=None,
                enable_amp=False,
                **kwargs):

        if self.label_dim > 0:
            if label is None:
                raise ValueError(
                    f'Model requires an additional label '
                    f'(with dimension {self.label_dim}) as input, '
                    f'but no label is received!')
            if label.ndim != 2 or label.shape != (input.shape[0],
                                                  self.label_dim):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_dim], where '
                                 f'`batch_size` equals to that of '
                                 f'images ({input.shape[0]}) and '
                                 f'`label_dim` equals to {self.label_dim}!\n'
                                 f'But `{label.shape}` is received!')
            label = label.to(dtype=torch.float32)
            if self.use_embedding:
                embed = self.class_embedding(label)
                if self.normalize_embedding:
                    embed = self.norm(embed)

        start = self.img_size_to_layer[input.shape[-1]]

        with autocast(enabled=enable_amp):
            x = self.fromRGB[start](input)

            if kwargs.get('instance_noise', 0) > 0:
                x = x + torch.randn_like(x) * kwargs['instance_noise']

            for i, layer in enumerate(self.layers[start:]):
                x = layer(x)

            # x = self.final_layer(x).reshape(x.shape[0], -1)
            x = self.score_conv(x).reshape(x.shape[0], -1)
            if self.use_embedding:
                score = (score * embed).sum(dim=1, keepdim=True)
                score = score / np.sqrt(self.embedding_dim)
            elif self.label_dim > 0:
                score = (score * label).sum(dim=1, keepdim=True)

            score = x[..., 0:1]
            position = x[..., 1:]

        results = {
            'score': score,
            'camera': position,
        }
        return results


class ResidualCCBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 downsample=True):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(
            CoordConv(inplanes,
                      planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=p), nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True))
        self.proj = nn.Conv2d(inplanes, planes,
                              1) if inplanes != planes else None
        self.downsample = downsample

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight,
                                        a=0.2,
                                        mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, identity):
        y = self.network(identity)

        if self.downsample: y = nn.functional.avg_pool2d(y, 2)
        if self.downsample: identity = nn.functional.avg_pool2d(identity, 2)
        identity = identity if self.proj is None else self.proj(identity)

        y = (y + identity) / math.sqrt(2)

        return y


class AdapterBlock(nn.Module):

    def __init__(self, output_channels):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, output_channels, 1, padding=0),
                                   nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        return self.model(input)


class AddCoords(nn.Module):
    """
    Borrowed from:
        https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)
        ], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) +
                torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    Borrowed from:
        https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class EqualLinear(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        bias_init=0,
        lr_mul=1,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input,
                       self.weight * self.scale,
                       bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, '
                '{self.weight.shape[0]})')


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, dim, eps):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self):
        return f'dim={self.dim}, epsilon={self.eps}'

    def forward(self, x):
        scale = (x.square().mean(dim=self.dim, keepdim=True) +
                 self.eps).rsqrt()
        return x * scale