# python3.8
"""Contains the implementation of discriminator described in EpiGRAF.

Paper: https://arxiv.org/pdf/2206.10535.pdf

Official PyTorch implementation: https://github.com/universome/epigraf
"""

import numpy as np
import torch
import torch.nn as nn
from models.utils.epigraf_model_helper import DiscriminatorBlock
from models.utils.epigraf_model_helper import MappingNetwork
from models.utils.epigraf_model_helper import ScalarEncoder1d
from models.utils.epigraf_model_helper import DiscriminatorEpilogue


class EpiGRAFDiscriminator(torch.nn.Module):
    """Defines the single discriminator network in EpiGRAF.

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
        disc_c_noise= 0,
        block_kwargs={},
        mapping_kwargs={},
        epilogue_kwargs={},
    ):
        """Initializes with basic settings."""

        super().__init__()

        self.num_additional_start_blocks = 3
        self.patch_params_cond = True
        self.camera_cond = True
        self.camera_cond_drop_p = 0.0
        self.camera_cond_noise_std = 0.0
        self.camera_cond_raw = True
        self.hyper_mod = True
        c_dim = 0
        self.c_dim = c_dim
        self.img_resolution = img_resolution * (
            2**self.num_additional_start_blocks)
        self.img_resolution_log2 = int(np.log2(self.img_resolution))
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

        # Concatenating coordinates to the input
        self.img_channels = img_channels

        if self.patch_params_cond > 0:
            self.scalar_enc = ScalarEncoder1d(coord_dim=3,
                                              x_multiplier=1000.0,
                                              const_emb_dim=256)
            assert self.scalar_enc.get_dim() > 0
        else:
            self.scalar_enc = None

        if (self.c_dim
                == 0) and (self.scalar_enc is None) and (not self.camera_cond):
            cmap_dim = 0

        if self.hyper_mod:
            hyper_mod_dim = 512
            self.hyper_mod_mapping = MappingNetwork(
                z_dim=0,
                c_dim=self.scalar_enc.get_dim(),
                camera_cond=False,
                w_dim=hyper_mod_dim,
                num_ws=None,
                w_avg_beta=None,
                **mapping_kwargs)
        else:
            self.hyper_mod_mapping = None
            hyper_mod_dim = 0

        common_kwargs = dict(img_channels=self.img_channels,
                             conv_clamp=conv_clamp)
        total_conditioning_dim = c_dim + (0 if (self.scalar_enc is None
                                                or not self.hyper_mod) else
                                          self.scalar_enc.get_dim())
        cur_layer_idx = 0

        for i, res in enumerate(self.block_resolutions):
            in_channels = channels_dict[res] if res < self.img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            down = 1 if i < self.num_additional_start_blocks else 2
            block = DiscriminatorBlock(in_channels,
                                       tmp_channels,
                                       out_channels,
                                       resolution=res,
                                       first_layer_idx=cur_layer_idx,
                                       use_fp16=use_fp16,
                                       down=down,
                                       c_dim=hyper_mod_dim,
                                       hyper_mod=self.hyper_mod,
                                       **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if self.c_dim > 0 or not self.scalar_enc is None:
            self.head_mapping = MappingNetwork(
                z_dim=0,
                c_dim=total_conditioning_dim,
                camera_cond=self.camera_cond,
                camera_cond_drop_p=self.camera_cond_drop_p,
                camera_cond_noise_std=self.camera_cond_noise_std,
                camera_raw_scalars=self.camera_cond_raw,
                w_dim=cmap_dim,
                num_ws=None,
                w_avg_beta=None,
                **mapping_kwargs)
        else:
            self.head_mapping = None
        self.b4 = DiscriminatorEpilogue(channels_dict[4],
                                        cmap_dim=cmap_dim,
                                        resolution=4,
                                        **epilogue_kwargs,
                                        **common_kwargs)

    def forward(self, img, c, patch_params, update_emas=False, **block_kwargs):

        _ = update_emas  # unused
        batch_size, _, h, w = img.shape

        if self.scalar_enc is not None:
            patch_scales = patch_params['scales']  # [batch_size, 2]
            patch_offsets = patch_params['offsets']  # [batch_size, 2]
            patch_params_cond = torch.cat(
                [patch_scales[:, [0]], patch_offsets],
                dim=1)  # [batch_size, 3]
            patch_scale_embs = self.scalar_enc(
                patch_params_cond)  # [batch_size, fourier_dim]

            if self.hyper_mod:
                patch_c = patch_scale_embs
                hyper_mod_c = self.hyper_mod_mapping(
                    z=None, c=patch_c)  # [batch_size, 512]
            else:
                patch_c = torch.zeros(len(patch_scale_embs),
                                      0,
                                      device=patch_scale_embs.device)
                hyper_mod_c = None

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x = block(x, img, c=hyper_mod_c, **block_kwargs)

        if self.c_dim > 0 or self.hyper_mod:
            assert c.shape[1] > 0
        if not self.head_mapping is None:
            cmap = self.head_mapping(z=None, c=patch_c, camera_angles=c)
        else:
            cmap = None

        x = self.b4(x, cmap)
        x = x.squeeze(1)  # [batch_size]
        return x

    def extra_repr(self):
        return (f'c_dim={self.c_dim:d}, '
                f'img_resolution={self.img_resolution:d}, '
                f'img_channels={self.img_channels:d}')
