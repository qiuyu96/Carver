# python3.8
"""Contains the runner for Pi-GAN."""

from copy import deepcopy

from .base_runner import BaseRunner

import torch

__all__ = ['PiGANRunner']


class PiGANRunner(BaseRunner):
    """Defines the runner for Pi-GAN."""

    def __init__(self, config):
        super().__init__(config)
        self.lod = getattr(self, 'lod', None)
        self.adjust_lr_flag = True
        self.grad_clip = self.config.get('grad_clip', None)
        if self.grad_clip is not None:
            self.running_stats.add(f'g_grad_norm',
                                   log_format='.3f',
                                   log_strategy='AVERAGE')
            self.running_stats.add(f'd_grad_norm',
                                   log_format='.3f',
                                   log_strategy='AVERAGE')

    def build_models(self):
        super().build_models()
        self.g_ema_img = self.config.models['generator'].get(
            'g_ema_img', 10000)
        if 'generator_smooth' not in self.models:
            self.models['generator_smooth'] = deepcopy(
                self.models['generator'])
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

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(params, **self.grad_clip)
        else:
            self.logger.info('There exsits no parameters to clip!')
            raise NotImplementedError

    def train_step(self, data, **train_kwargs):
        # Adjust learning rate.
        if 'adjust_lr_kimg' in self.config and self.adjust_lr_flag:
            ratio = self.config.adjust_lr_ratio
            if self.seen_img > (self.config.adjust_lr_kimg * 1e3):
                self.adjust_lr(ratio)
                self.adjust_lr_flag = False

        if self.amp_scaler.get_scale() < 1:
            self.amp_scaler.update(1.)

        G = self.models['generator']
        D = self.models['discriminator']
        Gs = self.models['generator_smooth']

         # Set level-of-details.
        if self.lod is None: self.lod = 0
        G.mlp.lod.data.fill_(self.lod)
        D.lod.data.fill_(self.lod)
        Gs.mlp.lod.data.fill_(self.lod)

        # Update discriminator.
        self.models['discriminator'].requires_grad_(True)
        self.models['generator'].requires_grad_(False)
        d_loss = self.loss.d_loss(self, data, sync=True)
        self.zero_grad_optimizer('discriminator')
        d_loss.backward()
        self.unscale_optimizer('discriminator')
        if self.grad_clip is not None:
            d_grad_norm = self.clip_grads(
                self.models['discriminator'].parameters())
            if d_grad_norm is not None:
                self.running_stats.update({'d_grad_norm': d_grad_norm.item()})
        self.step_optimizer('discriminator')

        # Life-long update for generator.
        beta = 0.5**(self.minibatch / self.g_ema_img)
        self.running_stats.update({'Misc/Gs Beta': beta})
        self.smooth_model(src=self.models['generator'],
                          avg=self.models['generator_smooth'],
                          beta=beta)

        # Update generator.
        self.models['discriminator'].requires_grad_(False)
        self.models['generator'].requires_grad_(True)
        g_loss = self.loss.g_loss(self, data, sync=True)
        self.zero_grad_optimizer('generator')
        g_loss.backward()
        self.unscale_optimizer('generator')
        if self.grad_clip is not None:
            g_grad_norm = self.clip_grads(
                self.models['generator'].parameters())
            if g_grad_norm is not None:
                self.running_stats.update(
                    {'g_grad_norm': g_grad_norm.item()})
        self.step_optimizer('generator')

        # Update automatic mixed-precision scaler.
        self.amp_scaler.update()
