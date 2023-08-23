# python3.8
"""Defines loss functions for GIRAFFE training."""

import numpy as np

import torch
import torch.nn.functional as F

from third_party.stylegan2_official_ops import conv2d_gradfix
from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss

__all__ = ['GIRAFFELoss']


class GIRAFFELoss(BaseLoss):
    """Contains the class to compute losses for training GIRAFFE."""

    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""

        if runner.enable_amp:
            raise NotImplementedError('GIRAFFE loss does not support automatic '
                                      'mixed precision training yet.')

        self.d_loss_kwargs = d_loss_kwargs or dict()
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)

        assert self.r1_gamma >= 0.0
        runner.running_stats.add('Loss/D Fake',
                                 log_name='loss_d_fake',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/D Real',
                                 log_name='loss_d_real',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.r1_gamma > 0.0:
            runner.running_stats.add('Loss/Real Gradient Penalty',
                                     log_name='loss_gp',
                                     log_format='.1e',
                                     log_strategy='AVERAGE')

        self.g_loss_kwargs = g_loss_kwargs or dict()

        runner.running_stats.add('Loss/G',
                                 log_name='loss_g',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')

        # Log loss settings.
        runner.logger.info('gradient penalty (D regularizer):', indent_level=1)
        runner.logger.info(f'r1_gamma: {self.r1_gamma}', indent_level=2)

    @staticmethod
    def run_G(runner,
              batch_size=None,
              sync=True):
        """Forwards generator."""
        # Prepare latent codes and labels.
        batch_size = batch_size or runner.batch_size
        device = runner.device

        # Forward generator.
        G = runner.ddp_models['generator']
        G_kwargs = runner.model_kwargs_train['generator']
        with ddp_sync(G, sync=sync):
            return G(z=None,
                     label=None,
                     batch_size=batch_size,
                     device=device,
                     **G_kwargs)

    @staticmethod
    def run_D(runner, images, sync=True):
        """Forwards discriminator."""
        # Augment the images.
        images = runner.augment(images, **runner.augment_kwargs)

        # Forward discriminator.
        D = runner.ddp_models['discriminator']
        D_kwargs = runner.model_kwargs_train['discriminator']
        with ddp_sync(D, sync=sync):
            return D(images, **D_kwargs)

    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        with conv2d_gradfix.no_weight_gradients():
            image_grad = torch.autograd.grad(
                outputs=[scores.sum()],
                inputs=[images],
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        grad_penalty = image_grad.square().sum((1, 2, 3))
        return grad_penalty

    def g_loss(self, runner, _data, sync=True):
        """Computes loss for generator."""
        fake_results = self.run_G(runner, sync=sync)
        fake_scores = self.run_D(runner,
                                 images=fake_results['image'],
                                 sync=False)['score']
        g_loss = F.softplus(-fake_scores)
        runner.running_stats.update({'Loss/G': g_loss})

        return g_loss.mean()

    def d_fake_loss(self, runner, _data, sync=True):
        """Computes discriminator loss on generated images."""
        fake_results = self.run_G(runner, sync=False)
        fake_scores = self.run_D(runner,
                                 images=fake_results['image'],
                                 sync=sync)['score']
        d_fake_loss = F.softplus(fake_scores)
        runner.running_stats.update({'Loss/D Fake': d_fake_loss})

        return d_fake_loss.mean()

    def d_real_loss(self, runner, data, sync=True):
        """Computes discriminator loss on real images."""
        real_images = data['image'].detach()
        real_scores = self.run_D(runner,
                                 images=real_images,
                                 sync=sync)['score']
        d_real_loss = F.softplus(-real_scores)
        runner.running_stats.update({'Loss/D Real': d_real_loss})

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return d_real_loss.mean()

    def d_reg(self, runner, data, sync=True):
        """Computes the regularization loss for discriminator."""
        if self.r1_gamma == 0.0:
            return None

        real_images = data['image'].detach().requires_grad_(True)
        real_scores = self.run_D(runner,
                                 images=real_images,
                                 sync=sync)['score']
        r1_penalty = self.compute_grad_penalty(images=real_images,
                                               scores=real_scores)
        runner.running_stats.update({'Loss/Real Gradient Penalty': r1_penalty})
        r1_penalty = r1_penalty * (self.r1_gamma * 0.5)

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return (real_scores * 0 + r1_penalty).mean()
