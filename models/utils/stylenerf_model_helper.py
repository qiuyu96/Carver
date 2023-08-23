# python3.8
"""Contains some auxiliary classes and functions utilized in the StyleNeRF
model.

This file is mostly borrowed from:

https://github.com/facebookresearch/StyleNeRF/blob/main/training/networks.py
"""

import numpy as np
import math
from einops import repeat

import torch
import torch.nn.functional as F

from utils import eg3d_misc as misc
from third_party.stylegan3_official_ops import conv2d_resample
from third_party.stylegan3_official_ops import upfirdn2d
from third_party.stylegan3_official_ops import bias_act
from third_party.stylegan3_official_ops import fma


def modulated_conv2d(
    x,
    weight,
    styles,
    noise           = None,
    up              = 1,
    down            = 1,
    padding         = 0,
    resample_filter = None,
    demodulate      = True,
    flip_weight     = True,
    fused_modconv   = True,
    mode            = '2d',
    **unused,
):
    """Defines the 2D/3D modulated convolution operation.

    Settings:

    (1) x: Input tensor of shape [batch_size, in_channels, in_height, in_width].
    (2) weight: Weight tensor,
        with shape [out_channels, in_channels, kernel_height, kernel_width].
    (3) styles: Modulation coefficients of shape [batch_size, in_channels].
    (4) noise: Optional noise tensor to add to the output activations.
    (5) up: Integer upsampling factor.
    (6) down: Integer downsampling factor.
    (7) padding: Padding with respect to the upsampled image.
    (8) resample_filter: Low-pass filter to apply when resampling activations.
        Must be prepared beforehand by calling `upfirdn2d.setup_filter()`.
    (9) demodulate: Whether to apply weight demodulation.
    (10) flip_weight: False = convolution,
         True = correlation (matches `torch.nn.functional.conv2d`).
    (11) fused_modconv: Whether to perform modulation, convolution,
         and demodulation as a single fused operation.
    (12) mode: Mode of convolution, 2D or 3D.
    (13) unused: Unused keyword arguments.
    """

    batch_size = x.shape[0]
    if mode == '3d':
        _, in_channels, kd, kh, kw = weight.shape
    else:
        _, in_channels, kh, kw = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight_sizes = (in_channels * kh *
                        kw if mode != '3d' else in_channels * kd * kh * kw)
        weight = weight * (1 / np.sqrt(weight_sizes) / weight.norm(
            float('inf'), dim=[1, 2, 3], keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1,
                                      keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if mode != '3d':
        rsizes, ssizes = [-1, 1, 1], [2, 3, 4]
    else:
        rsizes, ssizes = [-1, 1, 1, 1], [2, 3, 4, 5]

    if demodulate or fused_modconv:  # if not fused, skip
        w = weight.unsqueeze(0) * styles.reshape(batch_size, 1, *rsizes)
    if demodulate:
        dcoefs = (w.square().sum(dim=ssizes) + 1e-8).rsqrt()  # [NO]

    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, *rsizes, 1)
        # [NOIkk]
        # (batch_size, out_channels, in_channels, kernel_size, kernel_size)

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, *rsizes)
        if mode == '2d':
            x = conv2d_resample.conv2d_resample(x=x,
                                                w=weight.to(x.dtype),
                                                f=resample_filter,
                                                up=up,
                                                down=down,
                                                padding=padding,
                                                flip_weight=flip_weight)
        elif mode == '3d':
            x = conv3d(x=x,
                       w=weight.to(x.dtype),
                       up=up,
                       down=down,
                       padding=padding)
        else:
            raise NotImplementedError

        if demodulate and noise is not None:
            x = fma.fma(x,
                        dcoefs.to(x.dtype).reshape(batch_size, *rsizes),
                        noise.to(x.dtype))  # fused multiply add
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, *rsizes)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():
        # this value will be treated as a constant
        batch_size = int(batch_size)

    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, *w.shape[2:])
    if mode == '2d':
        x = conv2d_resample.conv2d_resample(x=x,
                                            w=w.to(x.dtype),
                                            f=resample_filter,
                                            up=up,
                                            down=down,
                                            padding=padding,
                                            groups=batch_size,
                                            flip_weight=flip_weight)
    elif mode == '3d':
        x = conv3d(x=x,
                   w=w.to(x.dtype),
                   up=up,
                   down=down,
                   padding=padding,
                   groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])

    if noise is not None:
        x = x.add_(noise)

    return x


class FullyConnectedLayer(torch.nn.Module):
    """Defines the fully-connected layer.

    Settings for the fully-connected layer:

    (1) in_features: Number of input features.
    (2) out_features: Number of output features.
    (3) bias: Whether to apply additive bias before the activation function.
    (4) activation: Activation function: 'relu', 'lrelu', etc.
    (5) lr_multiplier: Learning rate multiplier.
    (6) bias_init: Initial value for the additive bias.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation='linear',
        lr_multiplier=1,
        bias_init=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(
            torch.full([out_features],
                       np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class Conv2dLayer(torch.nn.Module):
    """Defines the 2D/3D convolutional layer.

    Settings for the convolutional layer:

    (1) in_channels: Number of input channels.
    (2) out_channels: Number of output channels.
    (3) kernel_size: Width and height of the convolution kernel.
    (4) bias: Whether to apply additive bias before the activation function.
    (5) activation: Activation function: 'relu', 'lrelu', etc.
    (6) up: Integer upsampling factor.
    (7) down: Integer downsampling factor.
    (8) resample_filter: Low-pass filter to apply when resampling activations.
    (9) conv_clamp: Clamp the output to +-X, None = disable clamping.
    (10) channels_last: Whether to expect the input to have
         memory_format=channels_last.
    (11) trainable: Whether to update the weights of this layer during training.
    (12) mode: Mode of convolution, 2D or 3D.
    (13) unused: Unused keyword arguments.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 activation='linear',
                 up=1,
                 down=1,
                 resample_filter=[1, 3, 3, 1],
                 conv_clamp=None,
                 channels_last=False,
                 trainable=True,
                 mode='2d',
                 **unused):
        """Initializes with basic settings."""

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.mode = mode
        weight_shape = [out_channels, in_channels, kernel_size, kernel_size]
        if mode == '3d':
            weight_shape += [kernel_size]

        memory_format = (torch.channels_last
                         if channels_last else torch.contiguous_format)
        weight = torch.randn(weight_shape).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster

        if self.mode == '2d':
            x = conv2d_resample.conv2d_resample(x=x,
                                                w=w.to(x.dtype),
                                                f=self.resample_filter,
                                                up=self.up,
                                                down=self.down,
                                                padding=self.padding,
                                                flip_weight=flip_weight)
        elif self.mode == '3d':
            x = conv3d(x=x,
                       w=w.to(x.dtype),
                       up=self.up,
                       down=self.down,
                       padding=self.padding)

        act_gain = self.act_gain * gain
        act_clamp = (self.conv_clamp *
                     gain if self.conv_clamp is not None else None)
        x = bias_act.bias_act(x,
                              b,
                              act=self.activation,
                              gain=act_gain,
                              clamp=act_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s}, up={self.up}, down={self.down}'


class Blur(torch.nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        from kornia.filters import filter2d
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)


class MappingNetwork(torch.nn.Module):
    """Defines the mapping network.

    Settings for the mapping network:

    (1) z_dim: Input latent (Z) dimensionality, 0 = no latent.
    (2) c_dim: Conditioning label (C) dimensionality, 0 = no label.
    (3) w_dim: Intermediate latent (W) dimensionality.
    (4) num_ws: Number of intermediate latents to output,
        None = do not broadcast.
    (5) num_layers: Number of mapping layers.
    (6) embed_features: Label embedding dimensionality, None = same as w_dim.
    (7) layer_features: Number of intermediate features in the mapping layers,
        None = same as w_dim.
    (8) activation: Activation function: 'relu', 'lrelu', etc.
    (9) lr_multiplier: Learning rate multiplier for the mapping layers.
    (10) w_avg_beta: Decay for tracking the moving average of W during training,
         None = do not track.
    (11) unused: Unused keyword arguments.
    """

    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        num_ws,
        num_layers=8,
        embed_features=None,
        layer_features=None,
        activation='lrelu',
        lr_multiplier=0.01,
        w_avg_beta=0.995,
        **unused,
    ):
        """Initializes with basic settings."""

        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features
                         ] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:   # project label condition
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features,
                                        out_features,
                                        activation=activation,
                                        lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self,
                z=None,
                c=None,
                truncation_psi=1,
                truncation_cutoff=None,
                skip_w_avg_update=False,
                styles=None,
                **unused_kwargs):
        if styles is not None:
            return styles

        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                # Normalize z to shpere.
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if (self.w_avg_beta is not None and self.training
                and not skip_w_avg_update):
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(
                    self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi)
        return x


class SynthesisLayer(torch.nn.Module):
    """Defines the synthesis layer.

    Settings for the synthesis layer:

    (1) in_channels: Number of input channels.
    (2) out_channels: Number of output channels.
    (3) w_dim: Intermediate latent (W) dimensionality.
    (4) resolution: Resolution of this layer.
    (5) kernel_size: Convolution kernel size.
    (6) up: Integer upsampling factor.
    (7) use_noise: Whether to enable noise input.
    (8) activation: Activation function: 'relu', 'lrelu', etc.
    (9) resample_filter: Low-pass filter to apply when resampling activations.
    (10) conv_clamp: Clamp the output of convolution layers to +-X,
         None = disable clamping.
    (11) channels_last: Whether to use `channels_last` format for the weights.
    (12) upsample_mode: Mode of upsampling, choices:
         [default, bilinear, ray_comm, ray_attn, ray_penc]
    (13) use_group: Whether to use group convolution.
    (14) magnitude_ema_beta: Whether to use magnitude ema.
         `-1` means not using magnitude ema.
    (15) mode: Mode of convolution, choices: [1d, 2d, 3d].
    (16) unused_kwargs: Unused keyword arguments.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 w_dim,
                 resolution,
                 kernel_size=3,
                 up=1,
                 use_noise=True,
                 activation='lrelu',
                 resample_filter=[1, 3, 3, 1],
                 conv_clamp=None,
                 channels_last=False,
                 upsample_mode='default',
                 use_group=False,
                 magnitude_ema_beta=-1,
                 mode='2d',
                 **unused_kwargs):
        """Initializes with basic settings."""

        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.upsample_mode = upsample_mode
        self.mode = mode

        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        if up == 2:
            if 'pixelshuffle' in upsample_mode:
                self.adapter = torch.nn.Sequential(
                    Conv2dLayer(out_channels,
                                out_channels // 4,
                                kernel_size=1,
                                activation=activation),
                    Conv2dLayer(out_channels // 4,
                                out_channels * 4,
                                kernel_size=1,
                                activation='linear'),
                )
            elif 'nn_cat' in upsample_mode:
                self.adapter = torch.nn.Sequential(
                    Conv2dLayer(out_channels * 2,
                                out_channels // 4,
                                kernel_size=1,
                                activation=activation),
                    Conv2dLayer(out_channels // 4,
                                out_channels,
                                kernel_size=1,
                                activation='linear'),
                )
            elif 'ada' in upsample_mode:
                self.adapter = torch.nn.Sequential(
                    Conv2dLayer(out_channels,
                                8,
                                kernel_size=1,
                                activation=activation),
                    Conv2dLayer(8,
                                out_channels,
                                kernel_size=1,
                                activation='linear'))
                self.adapter[1].weight.data.zero_()
                if 'blur' in upsample_mode:
                    self.blur = Blur()
            else:
                raise NotImplementedError

        self.padding = kernel_size // 2
        self.groups = 2 if use_group else 1
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)

        memory_format = (torch.channels_last
                         if channels_last else torch.contiguous_format)
        weight_sizes = [
            out_channels // self.groups, in_channels, kernel_size, kernel_size
        ]
        if self.mode == '3d':
            weight_sizes += [kernel_size]
        weight = torch.randn(weight_sizes).to(memory_format=memory_format)
        self.weight = torch.nn.Parameter(weight)

        if use_noise:
            if self.mode == '2d':
                noise_sizes = [resolution, resolution]
            elif self.mode == '3d':
                noise_sizes = [resolution, resolution, resolution]
            else:
                raise NotImplementedError('not support for MLP')
            self.register_buffer('noise_const', torch.randn(noise_sizes))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

        self.magnitude_ema_beta = magnitude_ema_beta
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))

    def forward(self,
                x,
                w,
                noise_mode='random',
                fused_modconv=True,
                gain=1,
                skip_up=False,
                input_noise=None,
                **unused_kwargs):
        assert noise_mode in ['random', 'const', 'none']
        batch_size = x.size(0)

        if (self.magnitude_ema_beta > 0):
            if self.training:  # updating EMA.
                with torch.autograd.profiler.record_function(
                        'update_magnitude_ema'):
                    magnitude_cur = x.detach().to(
                        torch.float32).square().mean()
                    self.w_avg.copy_(
                        magnitude_cur.lerp(self.w_avg,
                                           self.magnitude_ema_beta))
            input_gain = self.w_avg.rsqrt()
            x = x * input_gain

        styles = self.affine(w)  # Batch x style_dim
        if styles.size(0) < x.size(0):  # for repeating
            assert (x.size(0) // styles.size(0) * styles.size(0) == x.size(0))
            styles = repeat(styles,
                            'b c -> (b s) c',
                            s=x.size(0) // styles.size(0))
        up = self.up if not skip_up else 1
        use_default = (self.upsample_mode == 'default')
        noise = None
        resample_filter = None
        if use_default and (up > 1):
            resample_filter = self.resample_filter

        if self.use_noise:
            if input_noise is not None:
                noise = input_noise * self.noise_strength
            elif noise_mode == 'random':
                noise_sizes = [x.shape[0], 1, up * x.shape[2], up * x.shape[3]]
                if self.mode == '3d':
                    noise_sizes += [up * x.shape[4]]
                noise = torch.randn(noise_sizes,
                                    device=x.device) * self.noise_strength
            elif noise_mode == 'const':
                noise = self.noise_const * self.noise_strength
                if noise.shape[-1] < (up * x.shape[3]):
                    noise = repeat(noise,
                                   'h w -> h (s w)',
                                   s=up * x.shape[3] // noise.shape[-1])

        flip_weight = (up == 1)  # slightly faster
        x = modulated_conv2d(x=x,
                             weight=self.weight,
                             styles=styles,
                             noise=noise if
                             (use_default and not skip_up) else None,
                             up=up if use_default else 1,
                             padding=self.padding,
                             resample_filter=resample_filter,
                             flip_weight=flip_weight,
                             fused_modconv=fused_modconv,
                             groups=self.groups,
                             mode=self.mode)

        if (up == 2) and (not use_default):
            resolution = x.size(-1) * 2
            if 'bilinear' in self.upsample_mode:
                x = F.interpolate(x,
                                  size=(resolution, resolution),
                                  mode='bilinear',
                                  align_corners=True)
            elif 'nearest' in self.upsample_mode:
                x = F.interpolate(x,
                                  size=(resolution, resolution),
                                  mode='nearest')
                x = upfirdn2d.filter2d(x, self.resample_filter)
            elif 'bicubic' in self.upsample_mode:
                x = F.interpolate(x,
                                  size=(resolution, resolution),
                                  mode='bicubic',
                                  align_corners=True)
            elif 'pixelshuffle' in self.upsample_mode:
                # Does not have rotation invariance.
                x = F.interpolate(
                    x, size=(resolution, resolution),
                    mode='nearest') + torch.pixel_shuffle(self.adapter(x), 2)
                if not 'noblur' in self.upsample_mode:
                    x = upfirdn2d.filter2d(x, self.resample_filter)
            elif 'nn_cat' in self.upsample_mode:
                x_pad = x.new_zeros(*x.size()[:2],
                                    x.size(-2) + 2,
                                    x.size(-1) + 2)
                x_pad[..., 1:-1, 1:-1] = x
                xl, xu, xd, xr = x_pad[..., 1:-1, :-2], x_pad[
                    ..., :-2, 1:-1], x_pad[..., 2:, 1:-1], x_pad[..., 1:-1, 2:]
                x1, x2, x3, x4 = xl + xu, xu + xr, xl + xd, xr + xd
                xb = torch.stack([x1, x2, x3, x4], 2) / 2
                xb = torch.pixel_shuffle(
                    xb.view(xb.size(0), -1, xb.size(-2), xb.size(-1)), 2)
                xa = F.interpolate(x,
                                   size=(resolution, resolution),
                                   mode='nearest')
                x = xa + self.adapter(torch.cat([xa, xb], 1))
                if not 'noblur' in self.upsample_mode:
                    x = upfirdn2d.filter2d(x, self.resample_filter)
            else:
                raise NotImplementedError

        if up == 2:
            if 'ada' in self.upsample_mode:
                x = x + self.adapter(x)
                if 'blur' in self.upsample_mode:
                    x = self.blur(x)

        if (noise is not None) and (not use_default) and (not skip_up):
            x = x.add_(noise.type_as(x))

        act_gain = self.act_gain * gain
        act_clamp = (self.conv_clamp * gain if self.conv_clamp is not None
                     else None)
        x = bias_act.bias_act(x,
                              self.bias.to(x.dtype),
                              act=self.activation,
                              gain=act_gain,
                              clamp=act_clamp)

        return x


class ToRGBLayer(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 w_dim=0,
                 kernel_size=1,
                 conv_clamp=None,
                 channels_last=False,
                 mode='2d',
                 **unused):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.mode = mode
        weight_shape = [out_channels, in_channels, kernel_size, kernel_size]
        if mode == '3d':
            weight_shape += [kernel_size]

        if w_dim > 0:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            memory_format = (torch.channels_last
                             if channels_last else torch.contiguous_format)
            self.weight = torch.nn.Parameter(
                torch.randn(weight_shape).to(memory_format=memory_format))
            self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
            self.weight_gain = 1 / np.sqrt(np.prod(weight_shape[1:]))

        else:
            assert kernel_size == 1, 'does not support larger kernel sizes for now. used in NeRF'
            assert mode != '3d', 'does not support 3D convolution for now'

            self.weight = torch.nn.Parameter(
                torch.Tensor(out_channels, in_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.weight_gain = 1.

            # Weight initialization.
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, w=None, fused_modconv=True):
        if w is not None:
            styles = self.affine(w) * self.weight_gain
            if x.size(0) > styles.size(0):
                assert (x.size(0) // styles.size(0) *
                        styles.size(0) == x.size(0))
                styles = repeat(styles,
                                'b c -> (b s) c',
                                s=x.size(0) // styles.size(0))
            x = modulated_conv2d(x=x,
                                 weight=self.weight,
                                 styles=styles,
                                 demodulate=False,
                                 fused_modconv=fused_modconv,
                                 mode=self.mode)
            x = bias_act.bias_act(x,
                                  self.bias.to(x.dtype),
                                  clamp=self.conv_clamp)
        else:
            if x.ndim == 2:
                x = F.linear(x, self.weight, self.bias)
            else:
                x = F.conv2d(x, self.weight[:, :, None, None], self.bias)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'


class SynthesisBlock(torch.nn.Module):
    """Defines the synthesis block.

    Settings for the synthesis block:

    (1) in_channels: Number of input channels.
    (2) out_channels: Number of output channels.
    (3) w_dim: Intermediate latent (W) dimensionality.
    (4) resolution: Resolution of this block.
    (5) img_channels: Number of output color channels.
    (6) is_last: Whether the current block is the last block.
    (7) architecture: Architecture: 'orig', 'skip', 'resnet'.
    (8) resample_filter:  Low-pass filter to apply when resampling activations.
    (9) conv_clamp: Clamp the output of convolution layers to +-X,
        None = disable clamping.
    (10) use_fp16: Whether to use use FP16 for this block.
    (11) fp16_channels_last: Whether to use channels-last memory format with
         FP16.
    (12) use_single_layer: Whether to use only one instead of two synthesis
         layer.
    (13) disable_upsample: Whether to disable upsampling.
    (14) layer_kwargs: Keyword arguments for `SynthesisLayer`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        img_channels,
        is_last,
        architecture='skip',
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        use_single_layer=False,
        disable_upsample=False,
        **layer_kwargs,
    ):
        """Initializes with basic settings."""

        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        self.groups = 1
        self.use_single_layer = use_single_layer
        self.margin = layer_kwargs.get('margin', 0)
        self.upsample_mode = layer_kwargs.get('upsample_mode', 'default')
        self.disable_upsample = disable_upsample
        self.mode = layer_kwargs.get('mode', '2d')

        if in_channels == 0:
            const_sizes = [out_channels, resolution, resolution]
            if self.mode == '3d':
                const_sizes = const_sizes + [resolution]
            self.const = torch.nn.Parameter(torch.randn(const_sizes))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels=in_channels,
                                        out_channels=out_channels,
                                        w_dim=w_dim,
                                        resolution=resolution,
                                        up=2 if (not disable_upsample) else 1,
                                        resample_filter=resample_filter,
                                        conv_clamp=conv_clamp,
                                        channels_last=self.channels_last,
                                        **layer_kwargs)
            self.num_conv += 1

        if not self.use_single_layer:
            self.conv1 = SynthesisLayer(in_channels=out_channels,
                                        out_channels=out_channels,
                                        w_dim=w_dim,
                                        resolution=resolution,
                                        conv_clamp=conv_clamp,
                                        channels_last=self.channels_last,
                                        **layer_kwargs)
            self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels,
                                    img_channels,
                                    w_dim=w_dim,
                                    conv_clamp=conv_clamp,
                                    channels_last=self.channels_last,
                                    groups=self.groups,
                                    mode=self.mode)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels,
                                    out_channels,
                                    kernel_size=1,
                                    bias=False,
                                    up=2,
                                    resample_filter=resample_filter,
                                    channels_last=self.channels_last,
                                    mode=self.mode)

    def forward(self,
                x,
                img,
                ws,
                force_fp32=False,
                fused_modconv=None,
                add_on=None,
                block_noise=None,
                disable_rgb=False,
                **layer_kwargs):
        misc.assert_shape(ws,
                          [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = (torch.float16
                 if self.use_fp16 and not force_fp32 else torch.float32)
        memory_format = (torch.channels_last if self.channels_last
                         and not force_fp32 else torch.contiguous_format)
        if fused_modconv is None:
            with misc.suppress_tracer_warnings():
                # this value will be treated as a constant
                fused_modconv = (not self.training) and (
                    dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).expand(ws.shape[0], *x.size())
        else:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if add_on is not None:
            add_on = add_on.to(dtype=dtype, memory_format=memory_format)

        if self.in_channels == 0:
            if not self.use_single_layer:
                layer_kwargs['input_noise'] = (block_noise[:, 1:2]
                                               if block_noise is not None
                                               else None)
                x = self.conv1(x,
                               next(w_iter),
                               fused_modconv=fused_modconv,
                               **layer_kwargs)

        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            layer_kwargs['input_noise'] = (block_noise[:, 0:1]
                                           if block_noise is not None
                                           else None)
            x = self.conv0(x,
                           next(w_iter),
                           fused_modconv=fused_modconv,
                           **layer_kwargs)
            if not self.use_single_layer:
                layer_kwargs['input_noise'] = (block_noise[:, 1:2]
                                               if block_noise is not None
                                               else None)
                x = self.conv1(x,
                               next(w_iter),
                               fused_modconv=fused_modconv,
                               gain=np.sqrt(0.5),
                               **layer_kwargs)
            x = y.add_(x)
        else:
            layer_kwargs['input_noise'] = (block_noise[:, 0:1]
                                           if block_noise is not None
                                           else None)
            x = self.conv0(x,
                           next(w_iter),
                           fused_modconv=fused_modconv,
                           **layer_kwargs)
            if not self.use_single_layer:
                layer_kwargs['input_noise'] = (block_noise[:, 1:2]
                                               if block_noise is not None
                                               else None)
                x = self.conv1(x,
                               next(w_iter),
                               fused_modconv=fused_modconv,
                               **layer_kwargs)

        # ToRGB.
        if img is not None:
            if img.size(-1) * 2 == x.size(-1):
                if ((self.upsample_mode == 'bilinear_all') or
                    (self.upsample_mode == 'bilinear_ada')):
                    img = F.interpolate(img,
                                        scale_factor=2,
                                        mode='bilinear',
                                        align_corners=True)
                else:
                    img = upfirdn2d.upsample2d(img, self.resample_filter)
            elif img.size(-1) == x.size(-1):
                pass
            else:
                raise NotImplementedError

        if self.is_last or self.architecture == 'skip':
            if not disable_rgb:
                y = x if add_on is None else x + add_on
                y = self.torgb(y, next(w_iter), fused_modconv=fused_modconv)
                y = y.to(dtype=torch.float32,
                         memory_format=torch.contiguous_format)
                img = img.add_(y) if img is not None else y
            else:
                img = None

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


class SynthesisNetwork(torch.nn.Module):
    """Defines the synthesis block.

    Settings for the synthesis block:

    (1) w_dim: Intermediate latent (W) dimensionality.
    (2) img_resolution: Output image resolution.
    (3) img_channels: Number of color channels.
    (4) channel_base: Overall multiplier for the number of channels.
    (5) channel_max: Maximum number of channels in any layer.
    (6) num_fp16_res: Use FP16 for the N highest resolutions.
    (7) block_kwargs: Keyword arguments for `SynthesisBlock`.
    """

    def __init__(
        self,
        w_dim,
        img_resolution,
        img_channels,
        channel_base=1,
        channel_max=512,
        num_fp16_res=0,
        **block_kwargs,
    ):
        """Initializes with basic settings."""

        assert (img_resolution >= 4
                and img_resolution & (img_resolution - 1) == 0)
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(2, self.img_resolution_log2 + 1)
        ]

        channel_base = int(channel_base * 32768)
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions
        }
        fp16_resolution = max(2**(self.img_resolution_log2 + 1 - num_fp16_res),
                              8)
        self.channels_dict = channels_dict

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   w_dim=w_dim,
                                   resolution=res,
                                   img_channels=img_channels,
                                   is_last=is_last,
                                   use_fp16=use_fp16,
                                   **block_kwargs)

            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []

        # This part is to slice the style matrices (W) to each layer (conv/RGB)
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(
                    ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def get_current_resolution(self):
        return [self.img_resolution]

    def extra_repr(self):
        return f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


class Generator(torch.nn.Module):
    """Defines the generator network.

    Settings for the generator network:

    (1) z_dim: Input latent (Z) dimensionality.
    (2) c_dim: Conditioning label (C) dimensionality.
    (3) w_dim: Intermediate latent (W) dimensionality.
    (4) img_resolution: Output resolution.
    (5) img_channels: Number of output color channels.
    (6) mapping_kwargs: Keyword arguments for `MappingNetwork`.
    (7) synthesis_kwargs: Keyword arguments for `SynthesisNetwork`.
    (8) encoder_kwargs(optional): Keyword arguments for Encoder.
    """

    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        mapping_kwargs={},
        synthesis_kwargs={},
        encoder_kwargs={},
    ):
        """Initializes with basic settings."""

        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(class_name=synthesis_kwargs.get(
            'module_name', "training.networks.SynthesisNetwork"),
                                          w_dim=w_dim,
                                          img_resolution=img_resolution,
                                          img_channels=img_channels,
                                          **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = None
        self.encoder = None

        if len(mapping_kwargs) > 0:  # Use mapping network
            self.mapping = SynthesisNetwork(class_name=mapping_kwargs.get(
                'module_name', "training.networks.MappingNetwork"),
                                            z_dim=z_dim,
                                            c_dim=c_dim,
                                            w_dim=w_dim,
                                            num_ws=self.num_ws,
                                            **mapping_kwargs)

        if len(encoder_kwargs) > 0:  # Use Image-Encoder
            encoder_kwargs['model_kwargs'].update({
                'num_ws': self.num_ws,
                'w_dim': self.w_dim
            })
            self.encoder = SynthesisNetwork(img_resolution=img_resolution,
                                            img_channels=img_channels,
                                            **encoder_kwargs)

    def forward(self,
                z=None,
                c=None,
                styles=None,
                truncation_psi=1,
                truncation_cutoff=None,
                img=None,
                **synthesis_kwargs):
        if styles is None:
            assert z is not None
            if (self.encoder is not None) and (img is not None):
                outputs = self.encoder(img)
                ws = outputs['ws']
                if ('camera' in outputs) and ('camera_mode'
                                              not in synthesis_kwargs):
                    synthesis_kwargs['camera_RT'] = outputs['camera']
            else:
                ws = self.mapping(z,
                                  c,
                                  truncation_psi=truncation_psi,
                                  truncation_cutoff=truncation_cutoff,
                                  **synthesis_kwargs)
        else:
            ws = styles

        img = self.synthesis(ws, **synthesis_kwargs)
        return img

    def get_final_output(self, *args, **kwargs):
        img = self.forward(*args, **kwargs)
        if isinstance(img, list):
            return img[-1]
        elif isinstance(img, dict):
            return img['img']
        return img


class DiscriminatorBlock(torch.nn.Module):
    """Defines the discriminator block.

    Settings for the discriminator block:

    (1) in_channels: Number of input channels, 0 = first block.
    (2) tmp_channels: Number of intermediate channels.
    (3) out_channels: Number of output channels.
    (4) resolution: Resolution of this block.
    (5) img_channels: Number of input color channels.
    (6) first_layer_idx: Index of the first layer.
    (7) architecture: Architecture: 'orig', 'skip', 'resnet'.
    (8) activation: Activation function: 'relu', 'lrelu', etc.
    (9) resample_filter: Low-pass filter to apply when resampling activations.
    (10) conv_clamp: Clamp the output of convolution layers to +-X,
         None = disable clamping.
    (11) use_fp16: Whether to use FP16 for this block.
    (12) fp16_channels_last: Whether to use `channels-last` memory format with
         FP16.
    (13) freeze_layers: Number of layers to freeze.
    """

    def __init__(
        self,
        in_channels,
        tmp_channels,
        out_channels,
        resolution,
        img_channels,
        first_layer_idx,
        architecture='resnet',
        activation='lrelu',
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        freeze_layers=0,
    ):
        """Initializes with basic settings."""

        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels,
                                       tmp_channels,
                                       kernel_size=1,
                                       activation=activation,
                                       trainable=next(trainable_iter),
                                       conv_clamp=conv_clamp,
                                       channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels,
                                 tmp_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 trainable=next(trainable_iter),
                                 conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels,
                                 out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 down=2,
                                 trainable=next(trainable_iter),
                                 resample_filter=resample_filter,
                                 conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels,
                                    out_channels,
                                    kernel_size=1,
                                    bias=False,
                                    down=2,
                                    trainable=next(trainable_iter),
                                    resample_filter=resample_filter,
                                    channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False, downsampler=None):
        dtype = (torch.float16
                 if self.use_fp16 and not force_fp32 else torch.float32)
        memory_format = (torch.channels_last if self.channels_last
                         and not force_fp32 else torch.contiguous_format)

        # Input.
        if x is not None:
            misc.assert_shape(
                x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(
                img,
                [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            if self.architecture != 'skip':
                img = None
            elif downsampler is not None:
                img = downsampler(img, 2)
            else:
                img = upfirdn2d.downsample2d(img, self.resample_filter)

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():
            # `as_tensor` results are registered as constants
            G = torch.min(
                torch.as_tensor(self.group_size),
                torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        # Split minibatch N into n groups of size G,
        # and channels C into F groups of size c.
        y = x.reshape(G, -1, F, c, H, W)    # [G, n, F, c, H, W]
        # Subtract mean over group.
        y = y - y.mean(dim=0)               # [G, n, F, c, H, W]
        # Calc variance over group.
        y = y.square().mean(dim=0)          # [n, F, c, H, W]
        # Calc stddev over group.
        y = (y + 1e-8).sqrt()               # [n, F, c, H, W]
        # Take average over channels and pixels.
        y = y.mean(dim=[2,3,4])             # [n, F]
        # Add missing dimensions.
        y = y.reshape(-1, F, 1, 1)          # [n, F, 1, 1]
        # Replicate over group and pixels.
        y = y.repeat(G, 1, H, W)            # [N, F, H, W]
        # Append to input as new channels.
        x = torch.cat([x, y], dim=1)        # [N, C, H, W]
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'


class DiscriminatorEpilogue(torch.nn.Module):
    """Defines the epilogue of discriminator network.

    Settings for the epilogue of discriminator network.

    (1) in_channels: Number of input channels.
    (2) cmap_dim: Dimensionality of mapped conditioning label, 0 = no label.
    (3) resolution: Resolution of this block.
    (4) img_channels: Number of input color channels.
    (5) architecture: Architecture: 'orig', 'skip', 'resnet'.
    (6) mbstd_group_size: Group size for the minibatch standard deviation layer,
        None = entire minibatch.
    (7) mbstd_num_channels: Number of features for the minibatch standard
        deviation layer, 0 = disable.
    (8) activation: Activation function: 'relu', 'lrelu', etc.
    (9) conv_clamp: Clamp the output of convolution layers to +-X,
        None = disable clamping.
    """

    def __init__(
        self,
        in_channels,
        cmap_dim,
        resolution,
        img_channels,
        architecture='resnet',
        mbstd_group_size=4,
        mbstd_num_channels=1,
        activation='lrelu',
        conv_clamp=None,
        final_channels=1,
    ):
        """Initializes with basic settings."""

        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.final_channels = final_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels,
                                       in_channels,
                                       kernel_size=1,
                                       activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size,
                                       num_channels=mbstd_num_channels
                                       ) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels,
                                in_channels,
                                kernel_size=3,
                                activation=activation,
                                conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution**2),
                                      in_channels,
                                      activation=activation)
        self.out = FullyConnectedLayer(
            in_channels, final_channels if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(
            x, [None, self.in_channels, self.resolution, self.resolution
                ])  # [NCHW]
        _ = force_fp32
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(
                img,
                [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1,
                               keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x


class Discriminator(torch.nn.Module):
    """Defines the discriminator network.

    Settings for the discriminator network:

    (1) c_dim: Conditioning label (C) dimensionality.
    (2) img_resolution: Input resolution.
    (3) img_channels: Number of input color channels.
    (4) architecture: Architecture: 'orig', 'skip', 'resnet'.
    (5) channel_base: Overall multiplier for the number of channels.
    (6) channel_max: Maximum number of channels in any layer.
    (7) num_fp16_res: Use FP16 for the N highest resolutions.
    (8) conv_clamp: Clamp the output of convolution layers to +-X,
        None = disable clamping.
    (9) cmap_dim: Dimensionality of mapped conditioning label, None = default.
    (10) block_kwargs: Arguments for `DiscriminatorBlock`.
    (11) mapping_kwargs: Arguments for `MappingNetwork`.
    (12) epilogue_kwargs: Arguments for `DiscriminatorEpilogue`.
    """

    def __init__(
        self,
        c_dim,
        img_resolution,
        img_channels,
        architecture='resnet',
        channel_base=32768,
        channel_max=512,
        num_fp16_res=0,
        conv_clamp=None,
        cmap_dim=None,
        block_kwargs={},
        mapping_kwargs={},
        epilogue_kwargs={},
    ):
        """Initializes with basic settings."""

        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(self.img_resolution_log2, 2, -1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions + [4]
        }
        fp16_resolution = max(2**(self.img_resolution_log2 + 1 - num_fp16_res),
                              8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels,
                             architecture=architecture,
                             conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels,
                                       tmp_channels,
                                       out_channels,
                                       resolution=res,
                                       first_layer_idx=cur_layer_idx,
                                       use_fp16=use_fp16,
                                       **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0,
                                          c_dim=c_dim,
                                          w_dim=cmap_dim,
                                          num_ws=None,
                                          w_avg_beta=None,
                                          **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4],
                                        cmap_dim=cmap_dim,
                                        resolution=4,
                                        **epilogue_kwargs,
                                        **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        if isinstance(img, dict):
            img = img['img']
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


class EqualConv2d(torch.nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()
        new_scale = 1.0
        self.weight = torch.nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size) *
            new_scale)
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def conv3d(x, w, up=1, down=1, padding=0, groups=1):
    if up > 1:
        x = F.interpolate(x,
                          scale_factor=up,
                          mode='trilinear',
                          align_corners=True)
    x = F.conv3d(x, w, padding=padding, groups=groups)
    if down > 1:
        x = F.interpolate(x,
                          scale_factor=1. / float(down),
                          mode='trilinear',
                          align_corners=True)
    return x