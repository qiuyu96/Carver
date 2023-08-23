# python3.8
"""Contains the runner for GRAF."""

from copy import deepcopy
import math
import torch
import torch.nn.functional as F

from .base_runner import BaseRunner

__all__ = ['GRAFRunner']


class GRAFRunner(BaseRunner):
    """Defines the runner for GRAF."""
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

        self.patch_resolution = self.config.patch_kwargs.get(
            'patch_resolution', 32)
        self.random_shift = self.config.patch_kwargs.get('random_shift', True)
        self.random_scale = self.config.patch_kwargs.get(
            'random_scale', True)
        self.min_scale = self.config.patch_kwargs.get('min_scale', 0.25)
        self.max_scale = self.config.patch_kwargs.get('max_scale', 1.)
        self.scale_anneal = self.config.patch_kwargs.get('scale_anneal', -1)

        self.w, self.h = torch.meshgrid([
            torch.linspace(-1, 1, self.patch_resolution),
            torch.linspace(-1, 1, self.patch_resolution)
        ])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

    def build_loss(self):
        super().build_loss()
        self.running_stats.add('Misc/Gs Beta',
                               log_name='Gs_beta',
                               log_format='.4f',
                               log_strategy='CURRENT')

    def get_patch_grid(self):
        """Get the location of every pixel of the image patch used for
        TRAINING."""
        if self.scale_anneal > 0:
            k_iter = self.iter // 1000 * 3
            min_scale = max(
                self.min_scale,
                self.max_scale * math.exp(-k_iter * self.scale_anneal))
            min_scale = min(0.9, min_scale)
        else:
            min_scale = self.min_scale

        scale = torch.Tensor(1)
        if self.random_scale:
            scale = scale.uniform_(min_scale, self.max_scale)
            h = self.h * scale
            w = self.w * scale

        if self.random_shift:
            max_offset = 1 - scale.item()
            h_offset = torch.Tensor(1).uniform_(
                0, max_offset) * (torch.randint(2, (1, )).float() - 0.5) * 2
            w_offset = torch.Tensor(1).uniform_(
                0, max_offset) * (torch.randint(2, (1, )).float() - 0.5) * 2

            h += h_offset
            w += w_offset

        patch_grid = torch.cat([h, w], dim=2).to(self.device)

        return patch_grid  # [patch_res, patch_res, 2]

    def train_step(self, data):
        # Patchify image for training.
        patch_grid = self.get_patch_grid()
        patch_grid = patch_grid.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        data['image'] = F.grid_sample(data['image'],
                                      patch_grid,
                                      mode='bilinear',
                                      align_corners=True)

        ### Update generator.
        self.models['discriminator'].requires_grad_(False)
        self.models['generator'].requires_grad_(True)

        # Update with adversarial loss.
        g_loss = self.loss.g_loss(self, data, patch_grid=patch_grid, sync=True)
        self.zero_grad_optimizer('generator')
        g_loss.backward()
        self.step_optimizer('generator')

        ### Update discriminator.
        self.models['discriminator'].requires_grad_(True)
        self.models['generator'].requires_grad_(False)

        # Update with adversarial loss.
        self.zero_grad_optimizer('discriminator')
        # Update with fake images (get synchronized together with real loss).
        d_fake_loss = self.loss.d_fake_loss(self,
                                            data,
                                            patch_grid=patch_grid,
                                            sync=False)
        d_fake_loss.backward()
        # Update with real images.
        d_real_loss = self.loss.d_real_loss(self, data, sync=True)
        d_real_loss.backward()
        self.step_optimizer('discriminator')

        # Update with gradient penalty.
        r1_penalty = self.loss.d_reg(self, data, sync=True)
        if r1_penalty is not None:
            self.zero_grad_optimizer('discriminator')
            r1_penalty.backward()
            self.step_optimizer('discriminator')

        # Life-long update generator.
        if self.g_ema_rampup is not None and self.g_ema_rampup > 0:
            g_ema_img = min(self.g_ema_img, self.seen_img * self.g_ema_rampup)
        else:
            g_ema_img = self.g_ema_img
        beta = 0.5 ** (self.minibatch / max(g_ema_img, 1e-8))
        self.running_stats.update({'Misc/Gs Beta': beta})
        self.smooth_model(src=self.models['generator'],
                          avg=self.models['generator_smooth'],
                          beta=beta)
