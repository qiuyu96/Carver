# python3.8
"""Contains implementation of discriminator described in StyleNeRF.

Paper: https://arxiv.org/pdf/2110.08985.pdf

Official PyTorch implementation: https://github.com/facebookresearch/StyleNeRF
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.ops import upsample
from models.utils.ops import downsample

from models.utils.stylenerf_model_helper import EqualConv2d
from models.utils.stylenerf_model_helper import MappingNetwork
from models.utils.stylenerf_model_helper import DiscriminatorBlock
from models.utils.stylenerf_model_helper import DiscriminatorEpilogue


class StyleNeRFDiscriminator(nn.Module):
    """Defines the single discriminator network in StyleNeRF.

    Settings for the backbone:

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
    (10) lowres_head: Add a low-resolution discriminator head.
    (11) dual_discriminator: Add low-resolution (NeRF) image.
    (12) dual_input_ratio: Optional another low-res image input,
         which will be interpolated to the main input.
    (13) block_kwargs: Arguments for DiscriminatorBlock.
    (14) mapping_kwargs: Arguments for MappingNetwork.
    (15) epilogue_kwargs: Arguments for DiscriminatorEpilogue.
    (16) upsample_type: Type of upsampling.
    (17) progressive: Whether to perform prohressive training.
    (18) resize_real_early: Peform resizing before the training loop.
    (19) enable_ema: Additionally save an EMA checkpoint.
    (20) predict_camera: Learn camera predictor as InfoGAN.
    (21) predict_9d_camera: Use 9D camera distribution.
    (22) predict_3d_camera: Use 3D camera (u, v, r),
         assuming camera is on the unit sphere.
    (23) no_camera_condition: Disable camera conditioning in the discriminator.
    (24) saperate_camera: By default, only works in the lowest resolution.
    (25) unused: Some unused keyword arguments.
    """

    def __init__(
        self,
        c_dim,
        img_resolution,
        img_channels,
        architecture='resnet',
        channel_base=1,
        channel_max=512,
        num_fp16_res=0,
        conv_clamp=None,
        cmap_dim=None,
        lowres_head=None,
        dual_discriminator=False,
        dual_input_ratio=None,
        block_kwargs={},
        mapping_kwargs={},
        epilogue_kwargs={},
        upsample_type='default',
        progressive=False,
        resize_real_early=False,
        enable_ema=False,
        predict_camera=False,
        predict_9d_camera=False,
        predict_3d_camera=False,
        no_camera_condition=False,
        saperate_camera=False,
        **unused):
        """Initializes with basic settings."""

        super().__init__()
        self.img_resolution      = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels        = img_channels
        self.block_resolutions = [
            2**i for i in range(self.img_resolution_log2, 2, -1)
        ]
        self.architecture        = architecture
        self.lowres_head         = lowres_head
        self.dual_input_ratio    = dual_input_ratio
        self.dual_discriminator  = dual_discriminator
        self.upsample_type       = upsample_type
        self.progressive         = progressive
        self.resize_real_early   = resize_real_early
        self.enable_ema          = enable_ema
        self.predict_camera      = predict_camera
        self.predict_9d_camera   = predict_9d_camera
        self.predict_3d_camera   = predict_3d_camera
        self.no_camera_condition = no_camera_condition
        self.separate_camera     = saperate_camera
        if self.progressive:
            assert self.architecture == 'skip', 'not supporting other types for now.'
        if self.dual_input_ratio is not None:
            # similar to EG3d, concat low/high-res images
            self.img_channels = self.img_channels * 2
        if self.predict_camera:
            assert not (self.predict_9d_camera and self.predict_3d_camera), 'cannot achieve at the same time'
        channel_base = int(channel_base * 32768)
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions + [4]
        }
        fp16_resolution = max(2**(self.img_resolution_log2 + 1 - num_fp16_res),
                              8)

        # camera prediction module
        self.c_dim = c_dim
        if predict_camera:
            if not self.no_camera_condition:
                if self.predict_3d_camera:
                    self.c_dim = out_dim = 3     # (u, v) on the sphere
                else:
                    self.c_dim = 16              # extrinsic 4x4 (for now)
                    if self.predict_9d_camera:
                        out_dim = 9
                    else:
                        out_dim = 16
            self.projector = EqualConv2d(channels_dict[4],
                                         out_dim,
                                         4,
                                         padding=0,
                                         bias=False)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if self.c_dim == 0:
            cmap_dim = 0
        if self.c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0,
                                          c_dim=self.c_dim,
                                          w_dim=cmap_dim,
                                          num_ws=None,
                                          w_avg_beta=None,
                                          **mapping_kwargs)

        # main discriminator blocks
        common_kwargs = dict(img_channels=self.img_channels,
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

        # dual discriminator or separate camera predictor
        if self.separate_camera or self.dual_discriminator:
            cur_layer_idx = 0
            for res in [
                    r for r in self.block_resolutions if r <= self.lowres_head
            ]:
                in_channels = channels_dict[res] if res < img_resolution else 0
                tmp_channels = channels_dict[res]
                out_channels = channels_dict[res // 2]
                block = DiscriminatorBlock(in_channels,
                                           tmp_channels,
                                           out_channels,
                                           resolution=res,
                                           first_layer_idx=cur_layer_idx,
                                           use_fp16=False,
                                           **block_kwargs,
                                           **common_kwargs)
                setattr(self, f'c{res}', block)
                cur_layer_idx += block.num_layers

        # final output module
        self.b4 = DiscriminatorEpilogue(channels_dict[4],
                                        cmap_dim=cmap_dim,
                                        resolution=4,
                                        **epilogue_kwargs,
                                        **common_kwargs)
        self.register_buffer("alpha", torch.scalar_tensor(-1))

    def set_alpha(self, alpha):
        if alpha is not None:
            self.alpha = self.alpha * 0 + alpha

    def set_resolution(self, res):
        self.curr_status = res

    def get_estimated_camera(self, img, **block_kwargs):
        if isinstance(img, dict):
            img = img['img']
        img4cam = img.clone()
        if self.progressive and (img.size(-1) != self.lowres_head):
            img4cam = downsample(img, self.lowres_head)

        c, xc = None, None
        for res in [
                r for r in self.block_resolutions
                if r <= self.lowres_head or (not self.progressive)
        ]:
            xc, img4cam = getattr(self, f'c{res}')(xc, img4cam, **block_kwargs)

        if self.separate_camera:
            c = self.projector(xc)[:, :, 0, 0]
            if self.predict_9d_camera:
                c = camera_9d_to_16d(c)
        return c, xc, img4cam

    def get_camera_loss(self, RT=None, UV=None, c=None):
        if UV is not None:
            return F.mse_loss(UV, c)
        elif RT is not None:
            return F.smooth_l1_loss(RT.reshape(RT.size(0), -1), c) * 10
        return None

    def get_block_resolutions(self, input_img):
        block_resolutions = self.block_resolutions
        lowres_head = self.lowres_head
        alpha = self.alpha
        img_res = input_img.size(-1)
        if self.progressive and (self.lowres_head
                                 is not None) and (self.alpha > -1):
            if (self.alpha < 1) and (self.alpha > 0):
                try:
                    n_levels, _, before_res, target_res = self.curr_status
                    alpha, index = math.modf(self.alpha * n_levels)
                    index = int(index)
                except Exception as e:
                    before_res = target_res = img_res
                if before_res == target_res:
                    alpha = 0
                block_resolutions = [
                    res for res in self.block_resolutions if res <= target_res
                ]
                lowres_head = before_res
            elif self.alpha == 0:
                block_resolutions = [
                    res for res in self.block_resolutions if res <= lowres_head
                ]
        return block_resolutions, alpha, lowres_head

    def forward(self,
                inputs,
                c=None,
                aug_pipe=None,
                return_camera=False,
                **block_kwargs):
        if not isinstance(inputs, dict):
            inputs = {'img': inputs}
        img = inputs['img']
        block_resolutions, alpha, lowres_head = self.get_block_resolutions(img)
        if img.size(-1) > block_resolutions[0]:
            img = downsample(img, block_resolutions[0])

        # this is to handle real images to obtain nerf-size image.
        if (self.dual_discriminator or
            (self.dual_input_ratio is not None)) and ('img_nerf'
                                                      not in inputs):
            inputs['img_nerf'] = img
            if self.dual_discriminator and (
                    inputs['img_nerf'].size(-1) >
                    self.lowres_head):  # using Conv to read image.
                inputs['img_nerf'] = downsample(inputs['img_nerf'],
                                                self.lowres_head)
            elif self.dual_input_ratio is not None:  # similar to EG3d
                if inputs['img_nerf'].size(-1) > (img.size(-1) //
                                                  self.dual_input_ratio):
                    inputs['img_nerf'] = downsample(
                        inputs['img_nerf'],
                        img.size(-1) // self.dual_input_ratio)
                img = torch.cat(
                    [img, upsample(inputs['img_nerf'], img.size(-1))], 1)

        camera_loss = None
        RT = inputs['camera_matrices'][1].detach(
        ) if 'camera_matrices' in inputs else None
        UV = inputs['camera_matrices'][2].detach(
        ) if 'camera_matrices' in inputs else None

        # perform separate camera predictor or dual discriminator
        if self.dual_discriminator or self.separate_camera:
            temp_img = img if not self.dual_discriminator else inputs[
                'img_nerf']
            c_nerf, x_nerf, img_nerf = self.get_estimated_camera(
                temp_img, **block_kwargs)
            if c.size(-1) == 0 and self.separate_camera:
                c = c_nerf
                if self.predict_3d_camera:
                    camera_loss = self.get_camera_loss(RT, UV, c)

        # if applied data augmentation for discriminator
        if aug_pipe is not None:
            assert self.separate_camera or (not self.predict_camera), 'ada may break the camera predictor.'
            img = aug_pipe(img)

        # obtain the downsampled image for progressive growing
        if self.progressive and (self.lowres_head is not None) and (
                self.alpha > -1) and (self.alpha < 1) and (alpha > 0):
            img0 = downsample(img, img.size(-1) // 2)

        x = None if (not self.progressive) or (
            block_resolutions[0] == self.img_resolution) else getattr(
                self, f'b{block_resolutions[0]}').fromrgb(img)
        for res in block_resolutions:
            block = getattr(self, f'b{res}')
            if (lowres_head == res) and (self.alpha > -1) and (
                    self.alpha < 1) and (alpha > 0):
                if self.architecture == 'skip':
                    img = img * alpha + img0 * (1 - alpha)
                if self.progressive:
                    x = x * alpha + block.fromrgb(img0) * (1 - alpha)
            x, img = block(x, img, **block_kwargs)

        # predict camera based on discriminator features
        if (c.size(-1)
                == 0) and self.predict_camera and (not self.separate_camera):
            c = self.projector(x)[:,:,0,0]
            if self.predict_9d_camera:
                c = camera_9d_to_16d(c)
            if self.predict_3d_camera:
                camera_loss = self.get_camera_loss(RT, UV, c)

        # camera conditional discriminator
        cmap = None
        if self.c_dim > 0:
            cc = c.clone().detach()
            cmap = self.mapping(None, cc)
        logits  = self.b4(x, img, cmap)
        if self.dual_discriminator:
            logits = torch.cat([logits, self.b4(x_nerf, img_nerf, cmap)], 0)

        outputs = {'score': logits}
        if self.predict_camera and (camera_loss is not None):
            outputs['camera_loss'] = camera_loss
        if return_camera:
            outputs['camera'] = c

        return outputs


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035.
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def camera_9d_to_16d(d9):
    d6, translation = d9[..., :6], d9[..., 6:]
    rotation = rotation_6d_to_matrix(d6)
    RT = torch.eye(4).to(device=d9.device, dtype=d9.dtype).reshape(
        1, 4, 4).repeat(d6.size(0), 1, 1)
    RT[:, :3, :3] = rotation
    RT[:, :3, -1] = translation
    return RT.reshape(-1, 16)