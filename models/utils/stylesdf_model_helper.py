# python3.8
"""Contains some auxiliary classes and functions utilized in the StyleSDF model.

This file is mostly borrowed from:

https://github.com/royorel/StyleSDF/blob/main/model.py
"""

import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from third_party.stylegan2_official_ops.fused_act import FusedLeakyReLU
from third_party.stylegan2_official_ops.fused_act import fused_leaky_relu
from third_party.stylegan2_official_ops.upfirdn2d import upfirdn2d
from third_party.stylegan2_official_ops.upfirdn2d import setup_filter


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(
            torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class MappingLinear(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 activation=None,
                 is_last=False):
        super().__init__()
        if is_last:
            weight_std = 0.25
        else:
            weight_std = 1

        self.weight = nn.Parameter(
            weight_std * nn.init.kaiming_normal_(torch.empty(out_dim, in_dim),
                                                 a=0.2,
                                                 mode='fan_in',
                                                 nonlinearity='leaky_relu'))

        if bias:
            self.bias = nn.Parameter(
                nn.init.uniform_(torch.empty(out_dim),
                                 a=-np.sqrt(1 / in_dim),
                                 b=np.sqrt(1 / in_dim)))
        else:
            self.bias = None

        self.activation = activation

    def forward(self, input):
        if self.activation != None:
            out = F.linear(input, self.weight)
            out = fused_leaky_relu(out, self.bias, scale=1)
        else:
            out = F.linear(input, self.weight, bias=self.bias)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        filter_kernel = kernel
        self.register_buffer("kernel", setup_filter(filter_kernel))

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input,
                        self.kernel,
                        up=self.factor,
                        down=1,
                        padding=self.pad + self.pad,
                        impl='ref')

        return out


class Downsample(nn.Module):

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input,
                        self.kernel,
                        up=1,
                        down=self.factor,
                        pad=self.pad)

        return out


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)
        filter_kernel = kernel
        self.register_buffer("kernel", setup_filter(filter_kernel))

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input,
                        self.kernel,
                        padding=self.pad + self.pad,
                        impl='ref')

        return out


class EqualConv2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(input,
                       self.weight * self.scale,
                       bias=self.bias,
                       stride=self.stride,
                       padding=self.padding)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'


class EqualLinear_sdf(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 lr_mul=1,
                 activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input,
                           self.weight * self.scale,
                           bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


class ModulatedConv2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 style_dim,
                 demodulate=True,
                 upsample=False,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 name=''):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel,
                             pad=(pad0, pad1),
                             upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        self.modulation = EqualLinear_sdf(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, upsample={self.upsample}, downsample={self.downsample})'

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel,
                             self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel,
                                 self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel,
                                                    self.out_channel,
                                                    self.kernel_size,
                                                    self.kernel_size)
            out = F.conv_transpose2d(input,
                                     weight,
                                     padding=0,
                                     stride=2,
                                     groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):

    def __init__(self, project=False):
        super().__init__()
        self.project = project
        self.weight = nn.Parameter(torch.zeros(1))
        self.prev_noise = None
        self.mesh_fn = None
        self.vert_noise = None

    def forward(self, image, noise=None, transform=None, mesh_path=None):
        batch, _, height, width = image.shape
        if noise is None:
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class StyConv(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 style_dim,
                 upsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 project_noise=False):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )

        self.noise = NoiseInjection(project=project_noise)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self,
                input,
                style,
                noise=None,
                transform=None,
                mesh_path=None):
        out = self.conv(input, style)
        out = self.noise(out,
                         noise=noise,
                         transform=transform,
                         mesh_path=mesh_path)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):

    def __init__(self,
                 in_channel,
                 style_dim,
                 upsample=True,
                 blur_kernel=[1, 3, 3, 1],
                 name=''):
        super().__init__()

        self.upsample = upsample
        out_channels = 3
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel,
                                    out_channels,
                                    1,
                                    style_dim,
                                    demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.upsample:
                out = out + self.upsample(skip)
            else:
                out = out + skip
        return out


class ConvLayer(nn.Sequential):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 bias=True,
                 activate=True):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            ))

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class VolumeRenderDiscConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 activate=False):
        super(VolumeRenderDiscConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              bias=bias and not activate)

        self.activate = activate
        if self.activate:
            self.activation = FusedLeakyReLU(out_channels, bias=bias, scale=1)
            bias_init_coef = np.sqrt(1 /
                                     (in_channels * kernel_size * kernel_size))
            nn.init.uniform_(self.activation.bias,
                             a=-bias_init_coef,
                             b=bias_init_coef)

    def forward(self, input):
        """
        input_tensor_shape: (N, C_in, H, W)
        output_tensor_shape: (N, C_out, H_out, W_out)
        :return: Conv2d + activation Result
        """
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class AddCoords(nn.Module):

    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_channel = torch.arange(dim_x,
                                  dtype=torch.float32,
                                  device=input_tensor.device).repeat(
                                      1, 1, dim_y, 1)
        yy_channel = torch.arange(dim_y,
                                  dtype=torch.float32,
                                  device=input_tensor.device).repeat(
                                      1, 1, dim_x, 1).transpose(2, 3)

        xx_channel = xx_channel / (dim_x - 1)
        yy_channel = yy_channel / (dim_y - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
        out = torch.cat([input_tensor, yy_channel, xx_channel], dim=1)

        return out


class CoordConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super(CoordConv2d, self).__init__()

        self.addcoords = AddCoords()
        self.conv = nn.Conv2d(in_channels + 2,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConvLayer(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 bias=True,
                 activate=True):
        super(CoordConvLayer, self).__init__()
        layers = []
        stride = 1
        self.activate = activate
        self.padding = kernel_size // 2 if kernel_size > 2 else 0

        self.conv = CoordConv2d(in_channel,
                                out_channel,
                                kernel_size,
                                padding=self.padding,
                                stride=stride,
                                bias=bias and not activate)

        if activate:
            self.activation = FusedLeakyReLU(out_channel, bias=bias, scale=1)

        bias_init_coef = np.sqrt(1 / (in_channel * kernel_size * kernel_size))
        nn.init.uniform_(self.activation.bias,
                         a=-bias_init_coef,
                         b=bias_init_coef)

    def forward(self, input):
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class VolumeRenderResBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = CoordConvLayer(in_channel, out_channel, 3)
        self.conv2 = CoordConvLayer(out_channel, out_channel, 3)
        self.pooling = nn.AvgPool2d(2)
        self.downsample = nn.AvgPool2d(2)
        if out_channel != in_channel:
            self.skip = VolumeRenderDiscConv2d(in_channel, out_channel, 1)
        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.pooling(out)

        downsample_in = self.downsample(input)
        if self.skip != None:
            skip_in = self.skip(downsample_in)
        else:
            skip_in = downsample_in

        out = (out + skip_in) / math.sqrt(2)

        return out


class ResBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 blur_kernel=[1, 3, 3, 1],
                 merge=False):
        super().__init__()

        self.conv1 = ConvLayer(2 * in_channel if merge else in_channel,
                               in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(2 * in_channel if merge else in_channel,
                              out_channel,
                              1,
                              downsample=True,
                              activate=False,
                              bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = (out + self.skip(input)) / math.sqrt(2)

        return out
