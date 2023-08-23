# python3.8
"""Contains the runner for GRAM."""

from copy import deepcopy

from .base_runner import BaseRunner

import torch

import torch.nn.functional as F

__all__ = ['GRAMRunner']


class GRAMRunner(BaseRunner):
    """Defines the runner for GRAM."""

    def __init__(self, config):
        super().__init__(config)
        self.D_repeats = self.config.get('D_repeats', 1)
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

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(params, **self.grad_clip)
        else:
            self.logger.info('There exsits no parameters to clip!')
            raise NotImplementedError

    def train_G(self, batch_size=None, split=1):
        G = self.ddp_models['generator']

        G_kwargs = self.model_kwargs_train['generator']
        D = self.ddp_models['discriminator']
        D_kwargs = self.model_kwargs_train['discriminator']

        _G_kwargs = dict()
        _D_kwargs = dict()
        batch_size = batch_size or self.batch_size
        assert batch_size % split == 0
        split_batch_size = batch_size // split
        latent_dim = self.models['generator'].latent_dim
        label_dim = self.models['generator'].label_dim

        latents = torch.randn((batch_size, *latent_dim), device=self.device)
        labels = None

        if label_dim > 0:
            rnd_labels = torch.randint(0,
                                       label_dim, (batch_size, ),
                                       device=self.device)
            labels = F.one_hot(rnd_labels, num_classes=label_dim)

        for split in range(0, batch_size, split_batch_size):
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                latent = latents[split * split_batch_size:(split + 1) *
                                 split_batch_size]
                label = labels[split * split_batch_size:(
                    split +
                    1) * split_batch_size] if labels is not None else labels
                gen_results = G(latent, label, **G_kwargs, **_G_kwargs)
                dis_results = D(
                    self.augment(gen_results['image'], **self.augment_kwargs),
                    label, **D_kwargs, **_D_kwargs)
                position_penalty = F.mse_loss(
                    dis_results['camera'], gen_results['camera']
                ) * self.config.loss.d_loss_kwargs['position_gamma']
                self.running_stats.update(
                    {'Loss/G Fake ID Penalty': position_penalty})
                g_loss = F.softplus(-dis_results['score']).mean()
                self.running_stats.update({'Loss/G': g_loss})
                g_loss = g_loss + position_penalty
                self.amp_scaler.scale(g_loss).backward()
        self.unscale_optimizer('generator')

        if self.grad_clip is not None:
            g_grad_norm = self.clip_grads(
                G.parameters())
            if g_grad_norm is not None:
                self.running_stats.update(
                    {'g_grad_norm': g_grad_norm.item()})
        self.step_optimizer('generator')

        # Update automatic mixed-precision scaler.
        self.amp_scaler.update()
        self.zero_grad_optimizer('generator')

    def train_step(self, data):

        if self.amp_scaler.get_scale() < 1:
            self.amp_scaler.update(1.)

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
        beta = 0.999
        self.running_stats.update({'Misc/Gs Beta': beta})
        self.smooth_model(src=self.models['generator'],
                          avg=self.models['generator_smooth'],
                          beta=beta)

        # Update generator.
        if self.iter % self.D_repeats == 0:
            self.models['discriminator'].requires_grad_(False)
            self.models['generator'].requires_grad_(True)
            self.train_G(split=self.config.loss.d_loss_kwargs.batch_split)
