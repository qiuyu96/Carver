# python3.8
"""Contains the runner for EpiGRAF."""

import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F

from .base_runner import BaseRunner
from utils.misc import linear_schedule

__all__ = ['EpiGRAFRunner']


class EpiGRAFRunner(BaseRunner):
    """Defines the runner for EpiGRAF."""

    def __init__(self, config):
        super().__init__(config)

    def build_models(self):
        super().build_models()

        self.g_ema_img = self.config.models['generator'].get(
            'g_ema_img', 10_000)
        self.g_ema_rampup = self.config.models['generator'].get(
            'g_ema_rampup', 0)
        if 'generator_smooth' not in self.models:
            self.models['generator_smooth'] = deepcopy(self.models['generator'])
            self.model_kwargs_init['generator_smooth'] = deepcopy(
                self.model_kwargs_init['generator'])
        if 'generator_smooth' not in self.model_kwargs_val:
            self.model_kwargs_val['generator_smooth'] = deepcopy(
                self.model_kwargs_val['generator'])

    def build_loss(self):
        super().build_loss()
        self.running_stats.add('Misc/Gs Beta',
                               log_name='Gs_beta',
                               log_format='.4f',
                               log_strategy='CURRENT')

    def gen_fake_labels(self):
        training_set = self.train_loader.dataset
        fake_labels = [
            training_set.get_pose(np.random.randint(len(training_set)))
            for _ in range(self.batch_size)
        ]
        fake_labels = torch.from_numpy(
            np.stack(fake_labels)).pin_memory().to(self.device)
        return fake_labels

    def sample_patch_params(self):
        """
        Generates patch scales and patch offsets
        Returns patch offsets for [0, 1] range (i.e. image_size = 1 unit)
        """
        min_scale = self.config.patch_sampling_kwargs.get('min_scale', 0.125)
        max_scale = self.config.patch_sampling_kwargs.get('max_scale', 1.0)
        alpha = self.config.patch_sampling_kwargs.get('alpha', 1.0)
        beta_val_start = self.config.patch_sampling_kwargs.get(
            'beta_val_start', 0.001)
        beta_val_end = self.config.patch_sampling_kwargs.get(
            'beta_val_end', 0.8)
        group_size = self.config.patch_sampling_kwargs.get(
            'mbstd_group_size', 4)
        anneal_kimg = self.config.patch_sampling_kwargs.get(
            'anneal_kimg', 10000)
        beta = linear_schedule(self.seen_img / 1000, beta_val_start,
                               beta_val_end, anneal_kimg)

        assert max_scale <= 1.0, f'Too large max_scale: {max_scale}'
        assert min_scale <= max_scale, f'Incorrect params: min_scale = {min_scale}, max_scale = {max_scale}'

        batch_size = self.batch_size
        num_groups = batch_size // group_size
        patch_scales_x = np.random.beta(a=alpha, b=beta, size=num_groups) * (
            max_scale - min_scale) + min_scale
        patch_scales_x = torch.from_numpy(patch_scales_x).float().to(
            self.device)
        patch_scales = torch.stack([patch_scales_x, patch_scales_x], dim=1)

        # Sample an offset from [0, 1 - patch_size]
        patch_offsets = torch.rand(patch_scales.shape,
                                   device=self.device) * (1.0 - patch_scales)

        # Replicate the groups (needed for the MiniBatchStdLayer)
        patch_scales = patch_scales.repeat_interleave(group_size, dim=0)
        patch_offsets = patch_offsets.repeat_interleave(group_size, dim=0)

        return {'scales': patch_scales, 'offsets': patch_offsets}

    def extract_patches_real(self,
                             x,
                             patch_params,
                             resolution,
                             align_corners=True,
                             for_grid_sample=True):
        """
        Extracts patches from images and interpolates them to a desired
        resolution. Assumes, that scales/offests in patch_params are given for
        the [0, 1] image range (i.e. not [-1, 1]).
        """
        batch_size, _, h, w = x.shape
        assert h == w, 'Can only work on square images for now'
        patch_scales = patch_params['scales']  # [batch_size, 2]
        patch_offsets = patch_params['offsets']  # [batch_size, 2]
        if align_corners:
            row = torch.linspace(-1, 1, resolution,
                                 device=self.device).float()  # [img_size]
        else:
            row = (torch.arange(0, resolution, device=self.device).float() /
                   resolution) * 2 - 1  # [img_size]
        x_coords = row.view(1, -1).repeat(resolution,
                                          1)  # [img_size, img_size]
        y_coords = -x_coords.t()  # [img_size, img_size]

        coords = torch.stack([x_coords, y_coords],
                             dim=2)  # [img_size, img_size, 2]
        coords = coords.view(-1, 2)  # [img_size ** 2, 2]
        coords = coords.t().view(1, 2, resolution, resolution).repeat(
            batch_size, 1, 1, 1)  # [batch_size, 2, img_size, img_size]
        coords = coords.permute(0, 2, 3,
                                1)  # [batch_size, 2, img_size, img_size]

        coords = (coords + 1.0) * patch_scales.view(
            batch_size, 1, 1, 2) - 1.0 + patch_offsets.view(
                batch_size, 1, 1, 2) * 2.0  # [batch_size, out_h, out_w, 2]
        if for_grid_sample:
            coords[:, :, :,
                   1] = -coords[:, :, :, 1]  # [batch_size, out_h, out_w]
        out = F.grid_sample(
            x, coords, mode='bilinear',
            align_corners=True)  # [batch_size, c, resolution, resolution]
        return out

    def train_step(self, data):
        # Some common arguments for various losses.
        blur_sigma = max(
            1 - self.seen_img / (self.loss.blur_fade_kimg * 1e3), 0
        ) * self.loss.blur_init_sigma if self.loss.blur_fade_kimg > 0 else 0

        real_img = {'image': data['image']}
        real_labels = data['pose']

        ### Update generator.
        self.models['discriminator'].requires_grad_(False)
        self.models['generator'].requires_grad_(True)

        fake_labels = self.gen_fake_labels()
        patch_params = self.sample_patch_params()
        g_loss = self.loss.g_loss(self,
                                  fake_labels,
                                  patch_params,
                                  blur_sigma,
                                  sync=True)
        self.zero_grad_optimizer('generator')
        g_loss.backward()
        self.step_optimizer('generator')

        pl_penalty = self.loss.g_reg(self, data, sync=True)
        if pl_penalty is not None:
            self.zero_grad_optimizer('generator')
            pl_penalty.backward()
            self.step_optimizer('generator')

        ### Update discriminator.
        self.models['discriminator'].requires_grad_(True)
        self.models['generator'].requires_grad_(False)

        self.zero_grad_optimizer('discriminator')
        fake_labels = self.gen_fake_labels()

        patch_params = self.sample_patch_params()
        d_fake_loss = self.loss.d_fake_loss(self,
                                            fake_labels,
                                            patch_params,
                                            blur_sigma,
                                            sync=False)
        d_fake_loss.backward()

        patch_resolution = self.config.patch_sampling_kwargs.get(
            'patc_resolution', 64)
        patch_params = self.sample_patch_params()
        real_patch = self.extract_patches_real(real_img['image'], patch_params,
                                               patch_resolution)
        real_patch = {'image': real_patch}
        d_real_loss = self.loss.d_real_loss(self,
                                            real_patch,
                                            real_labels,
                                            patch_params,
                                            blur_sigma,
                                            sync=True)
        d_real_loss.backward()
        self.step_optimizer('discriminator')

        patch_params = self.sample_patch_params()
        real_patch = self.extract_patches_real(real_img['image'], patch_params,
                                               patch_resolution)

        real_patch = {'image': real_patch}
        r1_penalty = self.loss.d_reg(self,
                                     real_patch,
                                     real_labels,
                                     patch_params,
                                     blur_sigma,
                                     sync=True)
        if r1_penalty is not None:
            self.zero_grad_optimizer('discriminator')
            r1_penalty.backward()
            self.step_optimizer('discriminator')

        self.models['generator'].progressive_update(self.seen_img / 1000)

        # Life-long update generator with ema.
        if self.g_ema_rampup is not None and self.g_ema_rampup > 0:
            g_ema_img = min(self.g_ema_img, self.seen_img * self.g_ema_rampup)
        else:
            g_ema_img = self.g_ema_img
        beta = 0.5 ** (self.minibatch / max(g_ema_img, 1e-8))
        self.running_stats.update({'Misc/Gs Beta': beta})
        self.smooth_model(src=self.models['generator'],
                          avg=self.models['generator_smooth'],
                          beta=beta)
