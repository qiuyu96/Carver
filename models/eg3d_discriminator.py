# python3.8
"""Contains the implementation of discriminator described in EG3D.

Paper: https://arxiv.org/pdf/2112.07945.pdf

Official PyTorch implementation: https://github.com/NVlabs/eg3d
"""

import numpy as np
import torch
from third_party.stylegan3_official_ops import upfirdn2d
from models.utils.eg3d_model_helper import DiscriminatorBlock
from models.utils.eg3d_model_helper import MappingNetwork
from models.utils.eg3d_model_helper import DiscriminatorEpilogue


class SingleDiscriminator(torch.nn.Module):
    """Defines the single discriminator network in EG3D.

    Settings for the backbone:

    (1) c_dim: Conditioning label dimensionality.
    (2) img_resolution: Input resolution.
    (3) img_channels: Number of input color channels.
    (4) architecture: Architecture: 'orig', 'skip', 'resnet'.
    (5) channel_base: Overall multiplier for the number of channels.
    (6) channel_max: Maximum number of channels in any layer.
    (7) num_fp16_res: Use FP16 for the N highest resolutions.
    (8) conv_clamp: Clamp the output of convolution layers to +-X.
        None = disable clamping.
    (9) cmap_dim: Dimensionality of mapped conditioning label, None = default.
    (10) disc_c_noise: Corrupt camera parameters with X std dev of noise
         before disc. pose conditioning.
    (11) block_kwargs: Arguments for DiscriminatorBlock.
    (12) mapping_kwargs: Arguments for MappingNetwork.
    (13) epilogue_kwargs: Arguments for DiscriminatorEpilogue.
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
        disc_c_noise=0,
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
        img = img['image']

        _ = update_emas # unused
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
        return (f'c_dim={self.c_dim:d}, '
                f'img_resolution={self.img_resolution:d}, '
                f'img_channels={self.img_channels:d}')


class DualDiscriminator(torch.nn.Module):
    """Defines the dual discriminator network in EG3D.

    Settings for the backbone:

    (1) c_dim: Conditioning label dimensionality.
    (2) img_resolution: Input resolution.
    (3) img_channels: Number of input color channels.
    (4) architecture: Architecture: 'orig', 'skip', 'resnet'.
    (5) channel_base: Overall multiplier for the number of channels.
    (6) channel_max: Maximum number of channels in any layer.
    (7) num_fp16_res: Use FP16 for the N highest resolutions.
    (8) conv_clamp: Clamp the output of convolution layers to +-X.
        None = disable clamping.
    (9) cmap_dim: Dimensionality of mapped conditioning label, None = default.
    (10) disc_c_noise: Corrupt camera parameters with X std dev of noise
         before disc. pose conditioning.
    (11) block_kwargs: Arguments for DiscriminatorBlock.
    (12) mapping_kwargs: Arguments for MappingNetwork.
    (13) epilogue_kwargs: Arguments for DiscriminatorEpilogue.
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
        disc_c_noise=0,
        block_kwargs={},
        mapping_kwargs={},
        epilogue_kwargs={},
    ):
        """Initializes with basic settings."""

        super().__init__()
        img_channels *= 2

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
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter([1, 3, 3, 1]))
        self.disc_c_noise = disc_c_noise

    def forward(self, img, c, update_emas=False, **block_kwargs):
        image_raw = filtered_resizing(img['image_raw'],
                                      size=img['image'].shape[-1],
                                      f=self.resample_filter)
        img = torch.cat([img['image'], image_raw], 1)

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0:
                c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return (f'c_dim={self.c_dim:d}, '
                f'img_resolution={self.img_resolution:d}, '
                f'img_channels={self.img_channels:d}')


class DummyDualDiscriminator(torch.nn.Module):
    """Defines the dummy dual discriminator network in EG3D.

    Settings for the backbone:

    (1) c_dim: Conditioning label dimensionality.
    (2) img_resolution: Input resolution.
    (3) img_channels: Number of input color channels.
    (4) architecture: Architecture: 'orig', 'skip', 'resnet'.
    (5) channel_base: Overall multiplier for the number of channels.
    (6) channel_max: Maximum number of channels in any layer.
    (7) num_fp16_res: Use FP16 for the N highest resolutions.
    (8) conv_clamp: Clamp the output of convolution layers to +-X.
        None = disable clamping.
    (9) cmap_dim: Dimensionality of mapped conditioning label, None = default.
    (10) disc_c_noise: Corrupt camera parameters with X std dev of noise
         before disc. pose conditioning.
    (11) block_kwargs: Arguments for DiscriminatorBlock.
    (12) mapping_kwargs: Arguments for MappingNetwork.
    (13) epilogue_kwargs: Arguments for DiscriminatorEpilogue.
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
        disc_c_noise=0,
        block_kwargs={},
        mapping_kwargs={},
        epilogue_kwargs={},
    ):
        """Initializes with basic settings."""

        super().__init__()
        img_channels *= 2

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
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter([1, 3, 3, 1]))

        self.raw_fade = 1

    def forward(self, img, c, update_emas=False, **block_kwargs):
        self.raw_fade = max(0, self.raw_fade - 1 / (500000 / 32))

        image_raw = filtered_resizing(img['image_raw'],
                                      size=img['image'].shape[-1],
                                      f=self.resample_filter) * self.raw_fade
        img = torch.cat([img['image'], image_raw], 1)

        _ = update_emas  # unused
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
        return (f'c_dim={self.c_dim:d}, '
                f'img_resolution={self.img_resolution:d}, '
                f'img_channels={self.img_channels:d}')


def filtered_resizing(image_orig_tensor, size, f, filter_mode='antialiased'):
    if filter_mode == 'antialiased':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor,
                                                          size=(size, size),
                                                          mode='bilinear',
                                                          align_corners=False,
                                                          antialias=True)
    elif filter_mode == 'classic':
        ada_filtered_64 = upfirdn2d.upsample2d(image_orig_tensor, f, up=2)
        ada_filtered_64 = torch.nn.functional.interpolate(ada_filtered_64,
                                                          size=(size * 2 + 2,
                                                                size * 2 + 2),
                                                          mode='bilinear',
                                                          align_corners=False)
        ada_filtered_64 = upfirdn2d.downsample2d(ada_filtered_64,
                                                 f,
                                                 down=2,
                                                 flip_filter=True,
                                                 padding=-1)
    elif filter_mode == 'none':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor,
                                                          size=(size, size),
                                                          mode='bilinear',
                                                          align_corners=False)
    elif type(filter_mode) == float:
        assert 0 < filter_mode < 1

        filtered = torch.nn.functional.interpolate(image_orig_tensor,
                                                   size=(size, size),
                                                   mode='bilinear',
                                                   align_corners=False,
                                                   antialias=True)
        aliased = torch.nn.functional.interpolate(image_orig_tensor,
                                                  size=(size, size),
                                                  mode='bilinear',
                                                  align_corners=False,
                                                  antialias=False)
        ada_filtered_64 = (1 -
                           filter_mode) * aliased + (filter_mode) * filtered

    return ada_filtered_64
