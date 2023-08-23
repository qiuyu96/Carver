# python3.8
"""Contains the runner for EG3D."""

import numpy as np
from copy import deepcopy
import torch

from .base_runner import BaseRunner
from models.eg3d_discriminator import filtered_resizing
from third_party.stylegan3_official_ops import upfirdn2d

__all__ = ['EG3DRunner']


class EG3DRunner(BaseRunner):
    """Defines the runner for EG3D."""

    def __init__(self, config):
        super().__init__(config)
        self.adjust_lr_flag = True

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

    def gen_fake_labels(self, runner):
        if self.models['generator'].random_pose:
            fake_labels = torch.Tensor(self.batch_size,
                self.models['generator'].label_dim).to(self.device)
        else:
            training_set = self.train_loader.dataset
            fake_labels = [
                training_set.get_pose(np.random.randint(len(training_set)))
                for _ in range(self.batch_size)
            ]
            fake_labels = torch.from_numpy(
                np.stack(fake_labels)).pin_memory().to(self.device)
        return fake_labels

    def train_step(self, data):
        # Adjust learning rate.
        if 'adjust_lr_kimg' in self.config and self.adjust_lr_flag:
            ratio = self.config.adjust_lr_ratio
            if self.seen_img > (self.config.adjust_lr_kimg * 1e3):
                self.adjust_lr(ratio)
                self.adjust_lr_flag = False

        # Set some common arguments for various losses.
        if self.loss.blur_fade_kimg > 0:
            blur_sigma = max(
                1 - self.seen_img / (self.loss.blur_fade_kimg * 1e3), 0)
            blur_sigma = blur_sigma * self.loss.blur_init_sigma
        else:
            blur_sigma = 0

        if self.loss.gpc_reg_fade_kimg > 0:
            alpha = min(self.seen_img / (self.loss.gpc_reg_fade_kimg * 1e3), 1)
        else:
            alpha = 1

        if self.loss.gpc_reg_prob is not None:
            swapping_prob = (1 - alpha) * 1 + alpha * self.loss.gpc_reg_prob
        else:
            swapping_prob = None

        rendering_resolution_initial = self.loss.rendering_resolution_initial
        rendering_resolution_final = self.loss.rendering_resolution_final
        if isinstance(rendering_resolution_final, list):
            rendering_resolution_final = rendering_resolution_final[0]
        if rendering_resolution_final is not None:
            alpha = min(
                self.seen_img /
                (self.loss.rendering_resolution_fade_kimg * 1e3), 1)
            rendering_resolution = int(
                np.rint(rendering_resolution_initial * (1 - alpha) +
                        rendering_resolution_final * alpha))
        else:
            rendering_resolution = rendering_resolution_initial

        # Update neural rendering resolution.
        self.models['generator'].rendering_resolution.data.fill_(
            rendering_resolution)
        self.models['generator_smooth'].rendering_resolution.data.fill_(
            rendering_resolution)

        real_img = data['image']
        if 'pose' in data:
            real_labels = data['pose']
        else:
            real_labels = torch.Tensor(self.batch_size,
                self.models['generator'].label_dim).to(self.device)

        real_img_raw = filtered_resizing(real_img,
                                         size=rendering_resolution,
                                         f=self.loss.resample_filter,
                                         filter_mode=self.loss.filter_mode)

        if self.loss.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size,
                                 blur_size + 1,
                                 device=real_img_raw.device).div(
                                     blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        ### Update generator.
        self.models['discriminator'].requires_grad_(False)
        self.models['generator'].requires_grad_(True)

        # Update with adversarial loss.
        fake_labels = self.gen_fake_labels(self)
        g_loss = self.loss.g_loss(self,
                                  fake_labels,
                                  blur_sigma,
                                  swapping_prob,
                                  sync=True)
        self.zero_grad_optimizer('generator')
        g_loss.backward()
        self.step_optimizer('generator')

        # Update with density regularization.
        fake_labels = self.gen_fake_labels(self)
        density_tvloss = self.loss.g_reg(self,
                                         fake_labels,
                                         swapping_prob,
                                         sync=True)
        if density_tvloss is not None:
            self.zero_grad_optimizer('generator')
            density_tvloss.backward()
            self.step_optimizer('generator')

        ### Update discriminator.
        self.models['discriminator'].requires_grad_(True)
        self.models['generator'].requires_grad_(False)

        # Update with adversarial loss.
        self.zero_grad_optimizer('discriminator')
        # Update with fake images (get synchronized together with real loss).
        fake_labels = self.gen_fake_labels(self)
        d_fake_loss = self.loss.d_fake_loss(self,
                                            fake_labels,
                                            blur_sigma,
                                            swapping_prob,
                                            sync=False)
        d_fake_loss.backward()

        # Update with real images.
        d_real_loss = self.loss.d_real_loss(self,
                                            real_img,
                                            real_labels,
                                            blur_sigma,
                                            sync=True)
        d_real_loss.backward()
        self.step_optimizer('discriminator')

        # Update with gradient penalty.
        r1_penalty = self.loss.d_reg(self,
                                     real_img,
                                     real_labels,
                                     blur_sigma,
                                     sync=True)
        if r1_penalty is not None:
            self.zero_grad_optimizer('discriminator')
            r1_penalty.backward()
            self.step_optimizer('discriminator')

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
