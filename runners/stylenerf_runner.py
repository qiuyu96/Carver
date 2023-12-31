# python3.8
"""Contains the runner for StyleNeRF."""

from copy import deepcopy

from .base_runner import BaseRunner
import torch.nn.functional as F

__all__ = ['StyleNeRFRunner']


class StyleNeRFRunner(BaseRunner):
    """Defines the runner for StyleNeRF."""
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

    def set_alpha(self):
        start = self.config.pg_iter_start    # kimg
        end = self.config.pg_iter_end        # kimg
        alpha = min(1., max(0., (self.seen_img / 1e3 - start) / (end - start)))
        return alpha

    def train_step(self, data):
        # Set `alpha` for progressive training.
        alpha = self.set_alpha()
        self.models['generator'].alpha.data.fill_(alpha)
        self.models['discriminator'].alpha.data.fill_(alpha)
        self.models['generator_smooth'].alpha.data.fill_(alpha)

        # Resize images.
        _, _, target_img_res = self.models['generator'].get_current_resolution()
        data['image'] = F.interpolate(data['image'], target_img_res)

        ### Update generator.
        self.models['discriminator'].requires_grad_(False)
        self.models['generator'].requires_grad_(True)

        # Update with adversarial loss.
        g_loss = self.loss.g_loss(self, data, sync=True)
        self.zero_grad_optimizer('generator')
        g_loss.backward()
        self.step_optimizer('generator')

        # Update with perceptual path length regularization if needed.
        pl_penalty = self.loss.g_reg(self, data, sync=True)
        if pl_penalty is not None:
            self.zero_grad_optimizer('generator')
            pl_penalty.backward()
            self.step_optimizer('generator')

        ### Update discriminator.
        self.models['discriminator'].requires_grad_(True)
        self.models['generator'].requires_grad_(False)

        # Update with adversarial loss.
        self.zero_grad_optimizer('discriminator')
        # Update with fake images (get synchronized together with real loss).
        d_fake_loss = self.loss.d_fake_loss(self, data, sync=False)
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
