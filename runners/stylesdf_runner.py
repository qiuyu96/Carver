# python3.8
"""Contains the runner for StyleSDF."""

from copy import deepcopy
import os
from .base_runner import BaseRunner

import torch

__all__ = ['StyleSDFRunner']


class StyleSDFRunner(BaseRunner):
    """Defines the runner for StyleSDF."""

    def __init__(self, config):
        super().__init__(config)
        self.lod = getattr(self, 'lod', None)
        self.D_repeats = self.config.get('D_repeats', 1)
        self.grad_clip = self.config.get('grad_clip', None)
        self.full_pipline = self.config.get('full_pipeline', None)
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
        self.models['generator'].requires_grad_(False)
        self.models['discriminator'].requires_grad_(True)
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

        if self.full_pipline:
            # Update with gradient penalty.
            r1_penalty = self.loss.d_reg(self, data, sync=True)
            if r1_penalty is not None:
                self.zero_grad_optimizer('discriminator')
                r1_penalty.backward()
                self.step_optimizer('discriminator')

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


        if self.full_pipline:
            # Update with perceptual path length regularization if needed.
            pl_penalty = self.loss.g_reg(self, data, sync=True)
            if pl_penalty is not None:
                self.zero_grad_optimizer('generator')
                pl_penalty.backward()
                self.step_optimizer('generator')


        # Life-long update for generator.
        beta = 0.5**(self.minibatch / self.g_ema_img)
        self.running_stats.update({'Misc/Gs Beta': beta})
        self.smooth_model(src=self.models['generator'],
                          avg=self.models['generator_smooth'],
                          beta=beta)
        # Update automatic mixed-precision scaler.
        self.amp_scaler.update()

    def load(self,
             filepath,
             running_metadata=True,
             optimizer=True,
             learning_rate=True,
             loss=True,
             augment=True,
             running_stats=False,
             map_location='cpu'):
        """Loads previous running status.

        Args:
            filepath: File path to load the checkpoint.
            running_metadata: Whether to load the running metadata, such as
                batch size, current iteration, etc. (default: True)
            optimizer: Whether to load the optimizer. (default: True)
            learning_rate: Whether to load the learning rate. (default: True)
            loss: Whether to load the loss. (default: True)
            augment: Whether to load the augmentation, especially the adaptive
                augmentation probability. (default: True)
            running_stats: Whether to load the running stats. (default: False)
            map_location: Map location used for model loading. (default: `cpu`)
        """
        self.logger.info(f'Resuming from checkpoint `{filepath}` ...')
        if not os.path.isfile(filepath):
            raise IOError(f'Checkpoint `{filepath}` does not exist!')
        map_location = map_location.lower()
        assert map_location in ['cpu', 'gpu']
        if map_location == 'gpu':
            map_location = lambda storage, location: storage.cuda(self.device)
        checkpoint = torch.load(filepath, map_location=map_location)
        # Load models.
        if 'models' not in checkpoint:
            checkpoint = {'models': checkpoint}
        for model_name, model in self.models.items():
            if model_name not in checkpoint['models']:
                self.logger.warning(f'Model `{model_name}` is not included in '
                                    f'the checkpoint, and hence will NOT be '
                                    f'loaded!', indent_level=1)
                continue

            if 'generator' in model_name:
                state_dict = checkpoint['models']['generator_smooth']
            else:
                state_dict = checkpoint['models'][model_name]

            missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                                  strict=False)
            self.logger.info(f'Successfully loaded model `{model_name}`.',
                             indent_level=1)
            self.logger.info(f'Missing keys are `{missing_keys}` and '
                             f'unexpected keys are `{unexpected_keys}`.',
                             indent_level=1)
        # Load running metadata.
        if running_metadata:
            if 'running_metadata' not in checkpoint:
                self.logger.warning('Running metadata is not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                self._iter = checkpoint['running_metadata']['iter']
                self._start_iter = self._iter
                self.seen_img = checkpoint['running_metadata']['seen_img']
        # Load optimizers.
        if optimizer:
            if 'optimizers' not in checkpoint:
                self.logger.warning('Optimizers are not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                for opt_name, opt in self.optimizers.items():
                    if opt_name not in checkpoint['optimizers']:
                        self.logger.warning(f'Optimizer `{opt_name}` is not '
                                            f'included in the checkpoint, and '
                                            f'hence will NOT be loaded!',
                                            indent_level=1)
                        continue
                    opt.load_state_dict(checkpoint['optimizers'][opt_name])
                    self.logger.info(f'Successfully loaded optimizer '
                                     f'`{opt_name}`.', indent_level=1)
        # Load learning rates.
        if learning_rate:
            if 'learning_rates' not in checkpoint:
                self.logger.warning('Learning rates are not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                for lr_name, lr in self.lr_schedulers.items():
                    if lr_name not in checkpoint['learning_rates']:
                        self.logger.warning(f'Learning rate `{lr_name}` is not '
                                            f'included in the checkpoint, and '
                                            f'hence will NOT be loaded!',
                                            indent_level=1)
                        continue
                    lr.load_state_dict(checkpoint['learning_rates'][lr_name])
                    self.logger.info(f'Successfully loaded learning rate '
                                     f'`{lr_name}`.', indent_level=1)
        # Load loss.
        if loss:
            if 'loss' not in checkpoint:
                self.logger.warning('Loss is not included in the checkpoint, '
                                    'and hence will NOT be loaded!',
                                    indent_level=1)
            else:
                self.loss.load_state_dict(checkpoint['loss'])
                self.logger.info('Successfully loaded loss.', indent_level=1)
        # Load augmentation.
        if augment:
            if 'augment' not in checkpoint:
                self.logger.warning('Augmentation is not included in '
                                    'the checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                self.augment.load_state_dict(checkpoint['augment'])
                self.logger.info('Successfully loaded augmentation.',
                                 indent_level=1)
        # Load running stats.
        #  Only resume `stats_pool` from checkpoint.
        if running_stats:
            if 'running_stats' not in checkpoint:
                self.logger.warning('Running stats is not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                self.running_stats.stats_pool = deepcopy(
                    checkpoint['running_stats'])
                self.running_stats.is_resumed = True  # avoid conflicts when add
                self.logger.info('Successfully loaded running stats.',
                                 indent_level=1)
        # Log message.
        tailing_message = ''
        if running_metadata and 'running_metadata' in checkpoint:
            tailing_message = f' (iteration {self.iter})'
        self.logger.info(f'Successfully loaded from checkpoint `{filepath}`.'
                         f'{tailing_message}')