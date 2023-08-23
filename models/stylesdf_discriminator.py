# python3.8
"""Contains the implementation of discriminator described in StyleSDF.

Paper: https://arxiv.org/pdf/2112.11427.pdf

Official PyTorch implementation: https://github.com/royorel/StyleSDF
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .utils.stylesdf_model_helper import VolumeRenderDiscConv2d
from .utils.stylesdf_model_helper import VolumeRenderResBlock
from .utils.stylesdf_model_helper import ResBlock
from .utils.stylesdf_model_helper import ConvLayer
from .utils.stylesdf_model_helper import EqualLinear_sdf


class StyleSDFDiscriminator(nn.Module):
    def __init__(self,
                 resolution=64,
                 latent_dim=256,
                 label_dim=0,
                 full_pipeline=False,
                 renderer_spatial_output_dim=64,
                 no_viewpoint_loss=False):
        super().__init__()
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        init_size = (renderer_spatial_output_dim
                     if not full_pipeline else resolution)
        self.viewpoint_loss = not no_viewpoint_loss
        final_out_channel = 3 if self.viewpoint_loss else 1
        channels = {
            2: 400,
            4: 400,
            8: 400,
            16: 400,
            32: 256,
            64: 128,
            128: 64,
        }

        self.register_buffer('lod', torch.zeros(()))
        convs = [
            VolumeRenderDiscConv2d(3, channels[init_size], 1, activate=True)
        ]

        log_size = int(math.log(init_size, 2))

        in_channel = channels[init_size]

        for i in range(log_size - 1, 0, -1):
            out_channel = channels[2 ** i]

            convs.append(VolumeRenderResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = VolumeRenderDiscConv2d(in_channel, final_out_channel,
                                                 2)

    def forward(self,
                input,
                label=None,
                options=None,
                alpha=None,
                enable_amp=False,
                **kwargs):

        with autocast(enabled=enable_amp):
            out = self.convs(input)
            out = self.final_conv(out)
            gan_preds = out[:, 0:1].clone()
            gan_preds = gan_preds.view(-1, 1)
            if self.viewpoint_loss:
                viewpoints_preds = out[:, 1:]
                viewpoints_preds = viewpoints_preds.view(-1, 2)
            else:
                viewpoints_preds = None

        results = {
            'score': gan_preds,
            'camera': viewpoints_preds,
        }
        return results


class StyleSDFDiscriminator_full(nn.Module):
    def __init__(self,
                 resolution=1024,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 latent_dim=0.0,
                 label_dim=0.0):
        super().__init__()
        init_size = resolution
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.register_buffer('lod', torch.zeros(()))
        convs = [ConvLayer(3, channels[init_size], 1)]

        log_size = int(math.log(init_size, 2))

        in_channel = channels[init_size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        # # minibatch discrimination
        in_channel += 1

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear_sdf(channels[4] * 4 * 4,
                            channels[4],
                            activation="fused_lrelu"),
            EqualLinear_sdf(channels[4], 1),
        )

    def forward(self,
                input,
                label=None,
                options=None,
                alpha=None,
                enable_amp=False,
                **kwargs):
        with autocast(enabled=enable_amp):
            out = self.convs(input)

            # minibatch discrimination
            batch, channel, height, width = out.shape
            group = min(batch, self.stddev_group)
            if batch % group != 0:
                group = 3 if batch % 3 == 0 else 2

            stddev = out.contiguous().view(group, -1, self.stddev_feat,
                                           channel // self.stddev_feat, height,
                                           width)
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
            stddev = stddev.repeat(group, 1, height, width)
            final_out = torch.cat([out, stddev], 1)

            # final layers
            final_out = self.final_conv(final_out)
            final_out = final_out.contiguous().view(batch, -1)
            final_out = self.final_linear(final_out)
            gan_preds = final_out[:,:1]

        results = {
            'score': gan_preds,
        }
        return results


class ResidualCCBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes,
                      planes,
                      kernel_size=kernel_size,
                      stride=2,
                      padding=p), nn.LeakyReLU(0.2, inplace=True))
        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight,
                                        a=0.2,
                                        mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, input):
        y = self.network(input)

        identity = self.proj(input)

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
        ],
                        dim=1)

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