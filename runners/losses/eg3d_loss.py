# python3.8
"""Defines loss functions for EG3D training."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss

from third_party.stylegan3_official_ops import upfirdn2d
from third_party.stylegan3_official_ops import conv2d_gradfix

__all__ = ['EG3DLoss']


class EG3DLoss(BaseLoss):
    """Contains the class to compute losses for training EG3D."""

    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""
        # Setting for discriminator loss.
        self.device = runner.device
        self.d_loss_kwargs = d_loss_kwargs or dict()
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        self.r1_interval = self.d_loss_kwargs.get('r1_interval', 16)
        if self.r1_gamma is None or self.r1_interval <= 0:
            self.r1_interval = 1
            self.r1_gamma = 0.0
        self.r1_interval = int(self.r1_interval)
        assert self.r1_gamma >= 0.0
        self.blur_init_sigma = self.d_loss_kwargs.get('blur_init_sigma', 10)
        self.blur_fade_kimg = self.d_loss_kwargs.get('blur_fade_kimg', 200)
        self.dual_discrimination = self.d_loss_kwargs.get(
            'dual_discrimination', True)
        self.filter_mode = self.d_loss_kwargs.get('filter_mode', 'antialiased')
        self.blur_raw_target = self.d_loss_kwargs.get('blur_raw_target', True)
        self.resample_filter = upfirdn2d.setup_filter([1, 3, 3, 1],
                                                      device=self.device)

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

        # Setting for generator loss.
        self.g_loss_kwargs = g_loss_kwargs or dict()
        self.pl_batch_shrink = int(self.g_loss_kwargs.get(
            'pl_batch_shrink', 2))
        self.pl_weight = self.g_loss_kwargs.get('pl_weight', 2.0)
        self.pl_decay = self.g_loss_kwargs.get('pl_decay', 0.01)
        self.pl_interval = self.g_loss_kwargs.get('pl_interval', 4)
        if self.pl_interval is None or self.pl_interval <= 0:
            self.pl_interval = 1
            self.pl_weight = 0.0
        self.pl_interval = int(self.pl_interval)
        assert self.pl_batch_shrink >= 1
        assert self.pl_weight >= 0.0
        assert 0.0 <= self.pl_decay <= 1.0
        self.rendering_resolution_initial = self.g_loss_kwargs.get(
            'rendering_resolution_initial', 64)
        self.rendering_resolution_final = self.g_loss_kwargs.get(
            'rendering_resolution_final', None)
        self.rendering_resolution_fade_kimg = self.g_loss_kwargs.get(
            'rendering_resolution_fade_kimg', 1000)
        self.gpc_reg_fade_kimg = self.g_loss_kwargs.get(
            'gpc_reg_fade_kimg', 1000)
        self.gpc_reg_prob = self.g_loss_kwargs.get('gpc_reg_prob', 0.5)
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        self.density_reg_interval = self.g_loss_kwargs.get(
            'density_reg_interval', 1)
        self.density_reg = self.g_loss_kwargs.get('density_reg', 0.0)
        self.density_reg_p_dist = self.g_loss_kwargs.get(
            'density_reg_p_dist', 0.0)

        runner.running_stats.add('Loss/G',
                                 log_name='loss_g',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/G Reg',
                                 log_name='loss_g_reg',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.pl_weight > 0.0:
            runner.running_stats.add('Loss/Path Length Penalty',
                                     log_name='loss_pl',
                                     log_format='.1e',
                                     log_strategy='AVERAGE')
            self.pl_mean = torch.zeros((), device=runner.device)

        # Settings for eikonal loss.
        self.eikonal_lambda = self.g_loss_kwargs.get('eikonal_lambda', 0.0)
        self.min_surf_lambda = self.g_loss_kwargs.get('min_surf_lambda', 0.0)
        self.min_surf_beta = self.g_loss_kwargs.get('min_surf_beta', 100.0)
        if self.eikonal_lambda > 0.0:
            runner.running_stats.add('Loss/Eikonal Loss',
                                     log_name='loss_eikonal',
                                     log_format='.2e',
                                     log_strategy='AVERAGE')
        if self.min_surf_lambda > 0.0:
            runner.running_stats.add('Loss/Min Surf Loss',
                                     log_name='loss_min_surf',
                                     log_format='.2e',
                                     log_strategy='AVERAGE')

    @staticmethod
    def run_G(runner,
              label,
              swapping_prob,
              batch_size=None,
              update_emas=False,
              requires_grad=False,
              sync=True):
        """Forwards generator."""
        batch_size = batch_size or runner.batch_size
        latent_dim = runner.models['generator'].z_dim
        latents = torch.randn((batch_size, latent_dim),
                              device=runner.device,
                              requires_grad=requires_grad)
        G = runner.ddp_models['generator']
        if swapping_prob is not None:
            # Here `label` means pose information.
            label_swapped = torch.roll(label.clone(), 1, 0)
            label_gen_conditioning = torch.where(
                torch.rand(
                    (label.shape[0], 1), device=label.device) < swapping_prob,
                label_swapped, label)
        else:
            label_gen_conditioning = torch.zeros_like(label)

        with ddp_sync(G, sync=sync):
            results = G(latents,
                        label,
                        label_swapped=label_gen_conditioning,
                        style_mixing_prob=runner.config.style_mixing_prob,
                        update_emas=update_emas)

        return results

    @staticmethod
    def run_D(runner, img, label, blur_sigma=0, update_emas=False, sync=True):
        D = runner.ddp_models['discriminator']
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size,
                             blur_size + 1,
                             device=img['image'].device).div(
                                 blur_sigma).square().neg().exp2()
            img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
        # Data augmentation (ada), optional.
        if runner.config.use_ada:
            img['image'] = runner.augment(img['image'], **runner.augment_kwargs)
            img['image_raw'] = torch.nn.functional.interpolate(
                img['image_raw'],
                size=img['image'].shape[2:],
                mode='bilinear',
                antialias=True)
            img['image_raw'] = runner.augment(img['image_raw'],
                                              **runner.augment_kwargs)
            img['image_raw'] = torch.nn.functional.interpolate(
                img['image_raw'],
                size=img['image_raw'].shape[2:],
                mode='bilinear',
                antialias=True)

        with ddp_sync(D, sync=sync):
            scores = D(img, label, update_emas=update_emas)

        return scores

    @staticmethod
    def compute_grad_penalty(images, scores, dual=False):
        """Computes gradient penalty."""
        # Scales the scores for autograd.grad's backward pass.
        # If disable amp, the scaler will always be 1.
        if not dual:
            with conv2d_gradfix.no_weight_gradients():
                image_grad = torch.autograd.grad(outputs=[scores.sum()],
                                                 inputs=[images['image']],
                                                 create_graph=True,
                                                 retain_graph=True,
                                                 only_inputs=True)[0]
            penalty = image_grad.square().sum((1, 2, 3))
        else:
            with conv2d_gradfix.no_weight_gradients():
                image_grad = torch.autograd.grad(
                    outputs=[scores.sum()],
                    inputs=[images['image'], images['image_raw']],
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)
            image_grad_0 = image_grad[0]
            image_grad_1 = image_grad[1]
            penalty = image_grad_0.square().sum(
                (1, 2, 3)) + image_grad_1.square().sum((1, 2, 3))

        return penalty

    def d_fake_loss(self,
                    runner,
                    fake_labels,
                    blur_sigma,
                    swapping_prob,
                    sync=True):
        """Computes discriminator loss on fake/generated images."""
        # Train with fake/generated samples.
        fake_imgs = self.run_G(runner,
                               fake_labels,
                               swapping_prob,
                               update_emas=True,
                               sync=False)
        fake_scores = self.run_D(runner,
                                 fake_imgs,
                                 fake_labels,
                                 blur_sigma,
                                 update_emas=True,
                                 sync=sync)
        d_fake_loss = F.softplus(fake_scores)
        runner.running_stats.update({'Loss/D Fake': d_fake_loss})
        return d_fake_loss.mean()

    def d_real_loss(self, runner, real_img, real_labels, blur_sigma, sync=True):
        # Train with real samples.
        real_img_tmp_image = real_img['image'].detach().requires_grad_(False)
        real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(
            False)
        real_img_tmp = {
            'image': real_img_tmp_image,
            'image_raw': real_img_tmp_image_raw
        }
        real_scores = self.run_D(runner,
                                 real_img_tmp,
                                 real_labels,
                                 blur_sigma,
                                 sync=sync)
        d_real_loss = F.softplus(-real_scores)
        runner.running_stats.update({'Loss/D Real': d_real_loss})

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return d_real_loss.mean()

    def d_reg(self, runner, real_img, real_labels, blur_sigma, sync=True):
        """Compute the regularization loss for discriminator."""
        if runner.iter % self.r1_interval != 1 or self.r1_gamma == 0.0:
            return None

        real_img_tmp_image = real_img['image'].detach().requires_grad_(True)
        real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(
            True)
        real_img_tmp = {
            'image': real_img_tmp_image,
            'image_raw': real_img_tmp_image_raw
        }
        real_scores = self.run_D(runner,
                                 real_img_tmp,
                                 real_labels,
                                 blur_sigma,
                                 sync=sync)
        r1_penalty = self.compute_grad_penalty(real_img_tmp,
                                               real_scores,
                                               dual=self.dual_discrimination)
        runner.running_stats.update({'Loss/Real Gradient Penalty': r1_penalty})
        r1_penalty = r1_penalty * (self.r1_gamma * 0.5) * self.r1_interval

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return (real_scores * 0 + r1_penalty).mean()

    def g_loss(self,
               runner,
               fake_labels,
               blur_sigma,
               swapping_prob,
               sync=True):
        """Computes loss for generator."""
        fake_results = self.run_G(runner,
                                  fake_labels,
                                  swapping_prob,
                                  sync=sync)
        fake_scores = self.run_D(runner,
                                 fake_results,
                                 fake_labels,
                                 blur_sigma,
                                 sync=False)

        g_loss = F.softplus(-fake_scores)
        runner.running_stats.update({'Loss/G': g_loss})

        if ('use_sdf' in runner.config and runner.config.use_sdf
                and self.eikonal_lambda > 0):
            eikonal_term = fake_results['eikonal_term']
            g_eikonal, g_minimal_surface = self.eikonal_loss(
                eikonal_term,
                sdf=fake_results['sdf'] if self.min_surf_lambda > 0 else None,
                beta=self.min_surf_beta)
            g_eikonal = self.eikonal_lambda * g_eikonal
            if self.min_surf_lambda > 0:
                g_minimal_surface = self.min_surf_lambda * g_minimal_surface
            runner.running_stats.update({'Loss/Eikonal Loss': g_eikonal})
            runner.running_stats.update(
                {'Loss/Min Surf Loss': g_minimal_surface})
        else:
            g_eikonal = 0.0
            g_minimal_surface = 0.0

        return (g_loss + g_eikonal + g_minimal_surface).mean()

    def g_reg(self,
              runner,
              fake_labels,
              swapping_prob=0.5,
              batch_size=None,
              sync=True):
        """Compute the density regularization loss."""
        if (runner.iter % self.density_reg_interval != 1
                or self.density_reg == 0):
            return None

        batch_size = batch_size or runner.batch_size
        latent_dim = runner.models['generator'].z_dim
        latents = torch.randn((batch_size, latent_dim), device=runner.device)

        if swapping_prob is not None:
            fake_labels_swapped = torch.roll(fake_labels, 1, 0)
            labels_conditioning = torch.where(
                torch.rand([], device=runner.device) < swapping_prob,
                fake_labels_swapped, fake_labels)

        initial_coordinates = torch.rand(
            (batch_size, 1000, 3), device=runner.device) * 2 - 1
        perturbed_coordinates = initial_coordinates + torch.randn_like(
            initial_coordinates) * self.density_reg_p_dist
        all_coordinates = torch.cat(
            [initial_coordinates, perturbed_coordinates], dim=1)

        G = runner.ddp_models['generator']
        with ddp_sync(G, sync=sync):
            density = G(latents,
                        labels_conditioning,
                        update_emas=False,
                        density_reg=True,
                        coordinates=all_coordinates)['sample_density']

        density_initial = density[:, :density.shape[1] // 2]
        density_perturbed = density[:, density.shape[1] // 2:]
        TVloss = torch.nn.functional.l1_loss(
            density_initial,
            density_perturbed) * self.density_reg * self.density_reg_interval

        runner.running_stats.update({'Loss/G Reg': TVloss})

        return TVloss

    def eikonal_loss(self, eikonal_term, sdf=None, beta=100.0):
        if eikonal_term == None:
            eikonal_loss = 0
        else:
            eikonal_loss = ((eikonal_term.norm(dim=-1) - 1) ** 2).mean()

        if sdf == None:
            minimal_surface_loss = torch.tensor(0.0, device=eikonal_term.device)
        else:
            minimal_surface_loss = torch.exp(-beta * torch.abs(sdf)).mean()

        return eikonal_loss, minimal_surface_loss