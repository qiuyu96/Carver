# python3.8
"""Contains some auxiliary classes and functions utilized in the EpiGRAF model.

This file is mostly borrowed from

https://github.com/universome/epigraf/blob/main/src/training/networks_stylegan2.py
"""

import math
import numpy as np
from einops import repeat

import torch
import torch.nn.functional as F

from utils import eg3d_misc as misc
from third_party.stylegan3_official_ops import conv2d_resample
from third_party.stylegan3_official_ops import upfirdn2d
from third_party.stylegan3_official_ops import bias_act
from third_party.stylegan3_official_ops import fma


class FourierEncoder1d(torch.nn.Module):

    def __init__(
        self,
        # Number of scalars to encode for each sample
        coord_dim: int,
        # Maximum scalar value (influences the amount of fourier features we use)
        max_x_value: float = 100.0,
        # Whether we should use positional embeddings from Transformer
        transformer_pe: bool = False,
        use_cos: bool = True,
        **construct_freqs_kwargs,
    ):
        super().__init__()
        assert coord_dim >= 1, f"Wrong coord_dim: {coord_dim}"
        self.coord_dim = coord_dim
        self.use_cos = use_cos
        if transformer_pe:
            d_model = 512
            fourier_coefs = torch.exp(
                torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        else:
            fourier_coefs = construct_log_spaced_freqs(
                max_x_value, **construct_freqs_kwargs)
        self.register_buffer('fourier_coefs', fourier_coefs)
        self.fourier_dim = self.fourier_coefs.shape[0]

    def get_dim(self) -> int:
        return self.fourier_dim * (2 if self.use_cos else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"
        assert x.shape[1] == self.coord_dim
        fourier_raw_embs = self.fourier_coefs.view(
            1, 1, self.fourier_dim) * x.float().unsqueeze(
                2)  # [batch_size, coord_dim, fourier_dim]
        if self.use_cos:
            fourier_embs = torch.cat(
                [fourier_raw_embs.sin(),
                 fourier_raw_embs.cos()],
                dim=2)  # [batch_size, coord_dim, 2 * fourier_dim]
        else:
            fourier_embs = fourier_raw_embs.sin(
            )  # [batch_size, coord_dim, fourier_dim]
        return fourier_embs


class ScalarEncoder1d(torch.nn.Module):
    """Defines 1-dimensional Fourier Features encoder (i.e. encodes raw
    scalars). Assumes that scalars are in range [0, 1].
    """

    def __init__(self,
                 coord_dim: int,
                 x_multiplier: float,
                 const_emb_dim: int,
                 use_raw: bool = False,
                 **fourier_enc_kwargs):
        super().__init__()
        self.coord_dim = coord_dim
        self.const_emb_dim = const_emb_dim
        self.x_multiplier = x_multiplier
        self.use_raw = use_raw

        if self.const_emb_dim > 0 and self.x_multiplier > 0:
            self.const_embed = torch.nn.Embedding(
                int(np.ceil(x_multiplier)) + 1, self.const_emb_dim)
        else:
            self.const_embed = None

        if self.x_multiplier > 0:
            self.fourier_encoder = FourierEncoder1d(coord_dim,
                                                    max_x_value=x_multiplier,
                                                    **fourier_enc_kwargs)
            self.fourier_dim = self.fourier_encoder.get_dim()
        else:
            self.fourier_encoder = None
            self.fourier_dim = 0

        self.raw_dim = 1 if self.use_raw else 0

    def get_dim(self) -> int:
        return self.coord_dim * (self.const_emb_dim + self.fourier_dim +
                                 self.raw_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes that x is in [0, 1] range

        misc.assert_shape(x, [None, self.coord_dim])
        batch_size, coord_dim = x.shape
        out = torch.empty(batch_size,
                          self.coord_dim,
                          0,
                          device=x.device,
                          dtype=x.dtype)  # [batch_size, coord_dim, 0]
        if self.use_raw:
            out = torch.cat([out, x.unsqueeze(2)],
                            dim=2)  # [batch_size, coord_dim, 1]
        if not self.fourier_encoder is None or not self.const_embed is None:
            # Convert from [0, 1] to the [0, `x_multiplier`] range
            x = x.float() * self.x_multiplier  # [batch_size, coord_dim]
        if not self.fourier_encoder is None:
            fourier_embs = self.fourier_encoder(
                x)  # [batch_size, coord_dim, fourier_dim]
            out = torch.cat(
                [out, fourier_embs],
                dim=2)  # [batch_size, coord_dim, raw_dim + fourier_dim]
        if not self.const_embed is None:
            const_embs = self.const_embed(
                x.round().long())  # [batch_size, coord_dim, const_emb_dim]
            out = torch.cat(
                [out, const_embs], dim=2
            )  # [batch_size, coord_dim, raw_dim + fourier_dim + const_emb_dim]
        out = out.view(
            batch_size,
            coord_dim * (self.raw_dim + self.const_emb_dim + self.fourier_dim)
        )  # [batch_size, coord_dim * (raw_dim + const_emb_dim + fourier_dim)]
        return out


def modulated_conv2d(
    x,
    weight,
    styles,
    noise=None,
    up=1,
    down=1,
    padding=0,
    resample_filter=None,
    demodulate=True,
    flip_weight=True,
    fused_modconv=True,
):
    """Defines the 2D modulated convolution operation.

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
    """

    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(
            float('inf'), dim=[1, 2, 3], keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1,
                                      keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x,
                                            w=weight.to(x.dtype),
                                            f=resample_filter,
                                            up=up,
                                            down=down,
                                            padding=padding,
                                            flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x,
                        dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1),
                        noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():
        # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x,
                                        w=w.to(x.dtype),
                                        f=resample_filter,
                                        up=up,
                                        down=down,
                                        padding=padding,
                                        groups=batch_size,
                                        flip_weight=flip_weight)
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
        activation='linear',
        bias=True,
        lr_multiplier=1,
        weight_init=1,
        bias_init=0,
    ):
        """Initializes with basic settings."""

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) *
            (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32),
                                    [out_features])
        self.bias = torch.nn.Parameter(
            torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        # Selecting the weights
        w = self.weight  # [c_out, c_in]
        b = self.bias if not self.bias is None else None  # [c_out]
        # Adjusting the scales
        # shape of `w`: [c_out, c_in] or [batch_size, c_out, c_in]
        w = w.to(x.dtype) * self.weight_gain
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        # Applying the weights
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class Conv2dLayer(torch.nn.Module):
    """Defines the 2D convolutional layer.

    Settings for the 2D convolutional layer:

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
    (12) c_dim: Whether to pass `c` via re-normalization.
    (13) hyper_mod: Whether to use hypernet-based modulation.
    """

    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        bias              = True,
        activation        = 'linear',
        up                = 1,
        down              = 1,
        resample_filter   = [1,3,3,1],
        conv_clamp        = None,
        channels_last     = False,
        trainable         = True,
        c_dim             = 0,
        hyper_mod            = False,
    ):
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

        memory_format = (torch.channels_last
                         if channels_last else torch.contiguous_format)
        weight = torch.randn(
            [out_channels, in_channels, kernel_size,
             kernel_size]).to(memory_format=memory_format)
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
        if hyper_mod:
            assert c_dim > 0
            self.affine = FullyConnectedLayer(c_dim, in_channels, bias_init=0)
        else:
            self.affine = None

    def forward(self, x, c: torch.Tensor = None, gain=1):
        w = self.weight * self.weight_gain  # [c_out, c_in, k, k]
        flip_weight = (self.up == 1)  # slightly faster
        if not self.affine is None:
            weights = 1.0 + self.affine(c).tanh().unsqueeze(2).unsqueeze(
                3)  # [batch_size, c_in, 1, 1]
            x = (x * weights).to(x.dtype)  # [batch_size, c_out, h, w]
        x = conv2d_resample.conv2d_resample(x=x,
                                            w=w.to(x.dtype),
                                            f=self.resample_filter,
                                            up=self.up,
                                            down=self.down,
                                            padding=self.padding,
                                            flip_weight=flip_weight)
        act_gain = self.act_gain * gain
        act_clamp = (self.conv_clamp *
                     gain if self.conv_clamp is not None else None)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        x = bias_act.bias_act(x,
                              b,
                              act=self.activation,
                              gain=act_gain,
                              clamp=act_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s}, up={self.up}, down={self.down}'


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
    (11) camera_cond: Whether to use camera conditioning.
    (12) camera_raw_scalars: Whether to use raw camera angles as input or
         preprocess them with Fourier Features.
    (13) camera_cond_drop_p: Camera conditioning dropout.
    (14) camera_cond_noise_std: Camera conditioning noise std.
    (15) mean_camera_pose: Average camera pose for use at test time.
    (16) include_cam_input: Whether to include camera conditioning as input.
    """

    def __init__(self,
                 z_dim,
                 c_dim,
                 w_dim,
                 num_ws,
                 num_layers=2,
                 embed_features=None,
                 layer_features=None,
                 activation='lrelu',
                 lr_multiplier=0.01,
                 w_avg_beta=0.998,
                 camera_cond=False,
                 camera_raw_scalars=False,
                 camera_cond_drop_p=0.0,
                 camera_cond_noise_std=0.0,
                 mean_camera_pose=None,
                 include_cam_input=True):
        """Initializes with basic settings."""

        super().__init__()
        if camera_cond:
            if camera_raw_scalars:
                self.camera_scalar_enc = ScalarEncoder1d(coord_dim=2,
                                                         x_multiplier=0.0,
                                                         const_emb_dim=0,
                                                         use_raw=True)
            else:
                self.camera_scalar_enc = ScalarEncoder1d(coord_dim=2,
                                                         x_multiplier=64.0,
                                                         const_emb_dim=0)
            if include_cam_input:
                c_dim = c_dim + self.camera_scalar_enc.get_dim()
            else:
                c_dim = self.camera_scalar_enc.get_dim()
            assert self.camera_scalar_enc.get_dim() > 0
        else:
            self.camera_scalar_enc = None

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.camera_cond_drop_p = camera_cond_drop_p
        self.camera_cond_noise_std = camera_cond_noise_std
        self.include_cam_input = include_cam_input

        if embed_features is None:
            embed_features = w_dim
        if self.c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features
                         ] + [layer_features] * (num_layers - 1) + [w_dim]

        if self.c_dim > 0:
            self.embed = FullyConnectedLayer(self.c_dim, embed_features)
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

        if not mean_camera_pose is None:
            self.register_buffer('mean_camera_pose', mean_camera_pose)
        else:
            self.mean_camera_pose = None

    def forward(self,
                z,
                c,
                camera_angles: torch.Tensor = None,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False):
        if (not self.camera_scalar_enc is None) and (not self.training) and (
                camera_angles is None):
            camera_angles = self.mean_camera_pose.unsqueeze(0).repeat(
                len(z), 1)  # [batch_size, 3]

        if not self.camera_scalar_enc is None:
            # Using only yaw and pitch for conditioning (roll is always zero)
            camera_angles = camera_angles[:, [0, 1]]  # [batch_size, 2]
            if self.training and self.camera_cond_noise_std > 0:
                camera_angles = (camera_angles + self.camera_cond_noise_std *
                                 torch.randn_like(camera_angles) *
                                 camera_angles.std(dim=0, keepdim=True))
                # [batch_size, 2]
            camera_angles = camera_angles.sign() * (
                (camera_angles.abs() %
                 (2.0 * np.pi)) / (2.0 * np.pi))  # [batch_size, 2]
            camera_angles_embs = self.camera_scalar_enc(
                camera_angles)  # [batch_size, fourier_dim]
            camera_angles_embs = F.dropout(
                camera_angles_embs,
                p=self.camera_cond_drop_p,
                training=self.training)  # [batch_size, fourier_dim]
            if self.include_cam_input:
                c = torch.zeros(len(camera_angles_embs),
                                0,
                                device=camera_angles_embs.device
                                ) if c is None else c  # [batch_size, c_dim]
                c = torch.cat([c, camera_angles_embs],
                              dim=1)  # [batch_size, c_dim + angle_emb_dim]
            else:
                c = camera_angles_embs

        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
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
        if update_emas and self.w_avg_beta is not None:
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
    (12) unused_kwargs: Unused keyword arguments.
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
                 **unused_kwargs):
        """Initializes with basic settings."""

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = (torch.channels_last
                         if channels_last else torch.contiguous_format)
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size,
                         kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const',
                                 torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(
            x, [None, self.in_channels, in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution],
                device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster
        x = modulated_conv2d(x=x,
                             weight=self.weight,
                             styles=styles,
                             noise=noise,
                             up=self.up,
                             padding=self.padding,
                             resample_filter=self.resample_filter,
                             flip_weight=flip_weight,
                             fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = (self.conv_clamp *
                     gain if self.conv_clamp is not None else None)
        x = bias_act.bias_act(x,
                              self.bias.to(x.dtype),
                              act=self.activation,
                              gain=act_gain,
                              clamp=act_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}, resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'


class ToRGBLayer(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 w_dim,
                 kernel_size=1,
                 conv_clamp=None,
                 channels_last=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = (torch.channels_last
                         if channels_last else torch.contiguous_format)
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size,
                         kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        if x.size(0) > styles.size(0):
            assert (x.size(0) // styles.size(0) * styles.size(0) == x.size(0))
            styles = repeat(styles,
                            'b c -> (b s) c',
                            s=x.size(0) // styles.size(0))
        x = modulated_conv2d(x=x,
                             weight=self.weight,
                             styles=styles,
                             demodulate=False,
                             fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
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
    (12) fused_modconv_default: Default value of fused_modconv.
    (13) layer_kwargs: Keyword arguments for `SynthesisLayer`.
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
        conv_clamp=256,
        use_fp16=False,
        fp16_channels_last=False,
        fused_modconv_default=True,
        **layer_kwargs,
    ):
        """Initializes with basic settings."""

        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(
                torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels,
                                        out_channels,
                                        w_dim=w_dim,
                                        resolution=resolution,
                                        up=2,
                                        resample_filter=resample_filter,
                                        conv_clamp=conv_clamp,
                                        channels_last=self.channels_last,
                                        **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels,
                                    out_channels,
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
                                    channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels,
                                    out_channels,
                                    kernel_size=1,
                                    bias=False,
                                    up=2,
                                    resample_filter=resample_filter,
                                    channels_last=self.channels_last)

    def forward(self,
                x,
                img,
                ws,
                force_fp32=False,
                fused_modconv=None,
                update_emas=False,
                **layer_kwargs):
        _ = update_emas  # unused
        misc.assert_shape(ws,
                          [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = (torch.float16
                 if self.use_fp16 and not force_fp32 else torch.float32)
        memory_format = (torch.channels_last if self.channels_last
                         and not force_fp32 else torch.contiguous_format)
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [
                None, self.in_channels, self.resolution // 2,
                self.resolution // 2
            ])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x,
                           next(w_iter),
                           fused_modconv=fused_modconv,
                           **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x,
                           next(w_iter),
                           fused_modconv=fused_modconv,
                           **layer_kwargs)
            x = self.conv1(x,
                           next(w_iter),
                           fused_modconv=fused_modconv,
                           gain=np.sqrt(0.5),
                           **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x,
                           next(w_iter),
                           fused_modconv=fused_modconv,
                           **layer_kwargs)
            x = self.conv1(x,
                           next(w_iter),
                           fused_modconv=fused_modconv,
                           **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [
                None, self.img_channels, self.resolution // 2,
                self.resolution // 2
            ])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32,
                     memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


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
        channel_base=32768,
        channel_max=512,
        num_fp16_res=4,
        **block_kwargs,
    ):
        assert (img_resolution >= 4
                and img_resolution & (img_resolution - 1) == 0)
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [
            2**i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions
        }
        fp16_resolution = max(2**(self.img_resolution_log2 + 1 - num_fp16_res),
                              8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels,
                                   out_channels,
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

    def extra_repr(self):
        return f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}, num_fp16_res={self.num_fp16_res:d}'


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
    """

    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        mapping_kwargs={},
        **synthesis_kwargs,
    ):
        """Initializes with basic settings."""
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim,
                                          img_resolution=img_resolution,
                                          img_channels=img_channels,
                                          **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim,
                                      c_dim=c_dim,
                                      w_dim=w_dim,
                                      num_ws=self.num_ws,
                                      **mapping_kwargs)

    def forward(self,
                z,
                c,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False,
                **synthesis_kwargs):
        ws = self.mapping(z,
                          c,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
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
    (7) activation: Activation function: 'relu', 'lrelu', etc.
    (8) resample_filter: Low-pass filter to apply when resampling activations.
    (9) conv_clamp: Clamp the output of convolution layers to +-X,
         None = disable clamping.
    (10) use_fp16: Whether to use FP16 for this block.
    (11) fp16_channels_last: Whether to use `channels-last` memory format with
         FP16.
    (12) freeze_layers: Number of layers to freeze.
    (13) down: Downsampling factor.
    (14) c_dim: Hyper-conditioning dimension.
    (15) hyper_mod: Whether to use hyper-cond in `Conv2dLayer`.
    """

    def __init__(
        self,
        in_channels,
        tmp_channels,
        out_channels,
        resolution,
        img_channels,
        first_layer_idx,
        activation='lrelu',
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        use_fp16=False,
        fp16_channels_last=False,
        freeze_layers=0,
        down=2,
        c_dim=0,
        hyper_mod=False,
    ):
        """Initializes with basic settings."""

        assert in_channels in [0, tmp_channels]
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
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

        self.fromrgb = Conv2dLayer(img_channels,
                                   tmp_channels,
                                   kernel_size=1,
                                   activation=activation,
                                   c_dim=c_dim,
                                   hyper_mod=False,
                                   trainable=next(trainable_iter),
                                   conv_clamp=conv_clamp,
                                   channels_last=self.channels_last)
        self.conv0 = Conv2dLayer(tmp_channels,
                                 tmp_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 c_dim=c_dim,
                                 hyper_mod=False,
                                 trainable=next(trainable_iter),
                                 conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)
        self.conv1 = Conv2dLayer(tmp_channels,
                                 out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 down=down,
                                 c_dim=c_dim,
                                 hyper_mod=hyper_mod,
                                 trainable=next(trainable_iter),
                                 resample_filter=resample_filter,
                                 conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)
        self.skip = Conv2dLayer(tmp_channels,
                                out_channels,
                                kernel_size=1,
                                bias=False,
                                down=down,
                                c_dim=c_dim,
                                hyper_mod=False,
                                trainable=next(trainable_iter),
                                resample_filter=resample_filter,
                                channels_last=self.channels_last)

    def forward(self, x, img, c: torch.Tensor = None, force_fp32=False):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = (torch.float16
                 if self.use_fp16 and not force_fp32 else torch.float32)
        memory_format = (torch.channels_last if self.channels_last
                         and not force_fp32 else torch.contiguous_format)

        # Input.
        if x is not None:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0:
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img, c=c)
            x = x + y if x is not None else y

        # Main layers.
        y = self.skip(x, c=c, gain=np.sqrt(0.5))
        x = self.conv0(x, c=c)
        x = self.conv1(x, c=c, gain=np.sqrt(0.5))
        x = y.add_(x)

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}'


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
        mbstd_group_size=4,
        mbstd_num_channels=1,
        activation='lrelu',
        conv_clamp=None,
    ):
        """Initializes with basic settings."""

        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size,
                                       num_channels=mbstd_num_channels
                                       ) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels,
                                in_channels,
                                kernel_size=3,
                                activation=activation,
                                conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution**2),
                                      out_features=in_channels,
                                      activation=activation)
        self.out = FullyConnectedLayer(
            in_channels, out_features=(1 if cmap_dim == 0 else cmap_dim))

    def forward(self, x, cmap, force_fp32=False):
        misc.assert_shape(x,
            [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]

        _ = force_fp32
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = x.flatten(1) # [batch_size, in_channels * (resolution ** 2)]
        x = self.fc(x) # [batch_size, in_channels]
        x = self.out(x) # [batch_size, 1 or cmap_dim]

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1,
                               keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}'


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
        num_fp16_res=4,
        conv_clamp=256,
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

    def forward(self, img, c, update_emas=False, **block_kwargs):
        _ = update_emas
        x = None
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


def construct_log_spaced_freqs(max_t: int,
                               skip_small_t_freqs: int = 0,
                               skip_large_t_freqs: int = 0):
    time_resolution = 2**np.ceil(np.log2(max_t))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(
        torch.arange(num_fourier_feats))  # [num_fourier_feats]
    powers = powers[skip_large_t_freqs:len(powers) -
                    skip_small_t_freqs]  # [num_fourier_feats]
    fourier_coefs = powers.float() * np.pi  # [1, num_fourier_feats]

    return fourier_coefs / time_resolution


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()