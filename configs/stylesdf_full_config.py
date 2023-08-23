# python3.8
"""Configurations for training StyleSDF in the second stage."""

import math
import click
from .base_config import BaseConfig

__all__ = ['StyleSDFFullConfig']

RUNNER = 'StyleSDFRunner'
DATASET = 'ImageDataset'
DISCRIMINATOR = 'StyleSDFFullDiscriminator'
GENERATOR = 'StyleSDFGenerator'
LOSS = 'StyleSDFLoss'

PI = math.pi


class StyleSDFFullConfig(BaseConfig):
    """Defines the configuration for training StyleSDF."""

    name = 'stylesdf_full'
    hint = 'Train a StyleSDF full model.'
    info = '''
To train a StyleSDF model, the recommended settings are as follows:

\b
- batch_size: 4 (for FF-HQ dataset, 8 GPU)
- val_batch_size: 16 (for FF-HQ dataset, 8 GPU)
- data_repeat: 200 (for FF-HQ dataset)
- total_img: 25_000_000 (for FF-HQ dataset)
- train_data_mirror: True (for FF-HQ dataset)
'''

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.config.runner_type = RUNNER

    @classmethod
    def get_options(cls):
        options = super().get_options()

        options['Data transformation settings'].extend([
            cls.command_option(
                '--resolution',
                type=cls.int_type,
                default=1024,
                help='Resolution of the training images.'),
            cls.command_option(
                '--render_resolution',
                type=cls.int_type,
                default=64,
                help='Resolution of the rendering images.'),
            cls.command_option(
                '--image_channels',
                type=cls.int_type,
                default=3,
                help='Number of channels of the training images.'),
            cls.command_option(
                '--min_val',
                type=cls.float_type,
                default=-1.0,
                help='Minimum pixel value of the training images.'),
            cls.command_option(
                '--max_val',
                type=cls.float_type,
                default=1.0,
                help='Maximum pixel value of the training images.'),
        ])

        options['Network settings'].extend([
            cls.command_option(
                '--g_init_res',
                type=cls.int_type,
                default=4,
                help='The initial resolution to start convolution with in '
                'generator.'),
            cls.command_option(
                '--latent_dim',
                type=cls.int_type,
                default=256,
                help='The dimension of the latent space.'),
            cls.command_option(
                '--label_dim',
                type=cls.int_type,
                default=0,
                help='Number of classes in conditioning training. Set to `0` '
                'to disable conditional training.'),
            cls.command_option(
                '--d_fmaps_factor',
                type=cls.float_type,
                default=1.0,
                help='A factor to control the number of feature maps of '
                'discriminator, which will be `factor * 16384`.'),
            cls.command_option(
                '--d_mbstd_groups',
                type=cls.int_type,
                default=4,
                help='Number of groups for MiniBatchSTD layer of '
                'discriminator.'),
            cls.command_option(
                '--g_fmaps_factor',
                type=cls.float_type,
                default=1.0,
                help='A factor to control the number of feature maps of '
                'generator, which will be `factor * 16384`.'),
            cls.command_option(
                '--g_num_mappings',
                type=cls.int_type,
                default=3,
                help='Number of mapping layers of generator.'),
            cls.command_option(
                '--d_architecture',
                type=str,
                default='resnet',
                help='Architecture type of discriminator.'),
            cls.command_option(
                '--num_fp16_res',
                type=cls.int_type,
                default=0,
                help='Number of (highest) resolutions that use `float16` '
                'precision for training, which speeds up the training yet '
                'barely affects the performance. The official '
                'StyleGAN-ADA uses 4 by default.')
        ])

        options['Training settings'].extend([
            cls.command_option(
                '--d_lr',
                type=cls.float_type,
                default=0.002,
                help='The learning rate of discriminator.'),
            cls.command_option(
                '--d_beta_1',
                type=cls.float_type,
                default=0.0,
                help='The Adam hyper-parameter `beta_1` for discriminator '
                'optimizer.'),
            cls.command_option(
                '--d_beta_2',
                type=cls.float_type,
                default=0.99,
                help='The Adam hyper-parameter `beta_2` for discriminator '
                'optimizer.'),
            cls.command_option(
                '--g_lr',
                type=cls.float_type,
                default=0.002,
                help='The learning rate of generator.'),
            cls.command_option(
                '--g_beta_1',
                type=cls.float_type,
                default=0.0,
                help='The Adam hyper-parameter `beta_1` for generator '
                'optimizer.'),
            cls.command_option(
                '--g_beta_2',
                type=cls.float_type,
                default=0.99,
                help='The Adam hyper-parameter `beta_2` for generator '
                'optimizer.'),
            cls.command_option(
                '--style_mixing_prob',
                type=cls.float_type,
                default=0.9,
                help='Probability to perform style mixing as a training '
                'regularization.'),
            cls.command_option(
                '--r1_gamma',
                type=cls.float_type,
                default=10.0,
                help='Factor to control the strength of gradient penalty.'),
            cls.command_option(
                '--r1_interval', type=cls.int_type, default=16,
                help='Interval (in iterations) to perform gradient penalty.'),
            cls.command_option(
                '--pl_interval', type=cls.int_type, default=4,
                help='Interval (in iterations) to perform perceptual path '
                     'length regularization.'),
            cls.command_option(
                '--g_ema_img',
                type=cls.int_type,
                default=10_000,
                help='Factor for updating the smoothed generator, which is '
                'particularly used for inference.'),
            cls.command_option(
                '--use_ada',
                type=cls.bool_type,
                default=False,
                help='Whether to use adaptive augmentation pipeline.')
        ])

        options['Rendering options'].extend([
            cls.command_option(
                '--clamp_mode',
                type=click.Choice(['softplus', 'relu', 'mipnerf']),
                default='relu',
                help='clamp mode of `sigmas` in intergration process.'),
            cls.command_option(
                '--num_points',
                type=cls.int_type,
                default=24,
                help='Number of uniform samples to take per ray '
                'in coarse pass.'),
            cls.command_option(
                '--ray_start',
                type=cls.float_type,
                default=0.88,
                help='Near point along each ray to start taking samples.'),
            cls.command_option(
                '--ray_end',
                type=cls.float_type,
                default=1.12,
                help='Far point along each ray to start taking samples.'),
            cls.command_option(
                '--radius_fix',
                type=cls.float_type,
                default=1.0,
                help='Radius of sphere for sampling camera position.'),
            cls.command_option(
                '--polar_mean',
                type=cls.float_type,
                default=PI / 2,
                help='Mean of polar (vertical) angle for sampling camera '
                'position.'),
            cls.command_option(
                '--polar_stddev',
                type=cls.float_type,
                default=0.155,
                help='Standard deviation of polar (vertical) angle of sphere '
                'for sampling camera position.'),
            cls.command_option(
                '--azimuthal_mean',
                type=cls.float_type,
                default=PI / 2,
                help='Mean of azimuthal (horizontal) angle for sampling camera '
                'position.'),
            cls.command_option(
                '--azimuthal_stddev',
                type=cls.float_type,
                default=0.3,
                help='Standard deviation of azimuthal (horizontal) angle of '
                'sphere for sampling camera position.'),
            cls.command_option(
                '--fov',
                type=cls.float_type,
                default=12,
                help='Field of view of the camera.'),
            cls.command_option(
                '--perturbation_strategy',
                type=click.Choice(
                    ['no', 'middle_uniform', 'uniform', 'self_uniform']),
                default='self_uniform',
                help='clamp mode of `sigmas` in intergration process.'),
        ])

        return options

    def parse_options(self):
        super().parse_options()

        resolution = self.args.pop('resolution')
        render_resolution = self.args.pop('render_resolution')
        image_channels = self.args.pop('image_channels')
        min_val = self.args.pop('min_val')
        max_val = self.args.pop('max_val')

        # Parse data transformation settings.
        data_transform_kwargs = dict(
            image_size=resolution,
            image_channels=image_channels,
            min_val=min_val,
            max_val=max_val
        )
        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_kwargs = data_transform_kwargs
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_kwargs = data_transform_kwargs

        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_kwargs = data_transform_kwargs
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_kwargs = data_transform_kwargs


        d_lr = self.args.pop('d_lr')
        d_beta_1 = self.args.pop('d_beta_1')
        d_beta_2 = self.args.pop('d_beta_2')
        g_lr = self.args.pop('g_lr')
        g_beta_1 = self.args.pop('g_beta_1')
        g_beta_2 = self.args.pop('g_beta_2')
        r1_interval = self.args.pop('r1_interval')
        pl_interval = self.args.pop('pl_interval')

        if r1_interval is not None and r1_interval > 0:
            d_mb_ratio = r1_interval / (r1_interval + 1)
            d_lr = d_lr * d_mb_ratio
            d_beta_1 = d_beta_1 ** d_mb_ratio
            d_beta_2 = d_beta_2 ** d_mb_ratio
        if pl_interval is not None and pl_interval > 0:
            g_mb_ratio = pl_interval / (pl_interval + 1)
            g_lr = g_lr * g_mb_ratio
            g_beta_1 = g_beta_1 ** g_mb_ratio
            g_beta_2 = g_beta_2 ** g_mb_ratio

        latent_dim = self.args.pop('latent_dim')
        label_dim = self.args.pop('label_dim')
        self.args.pop('g_fmaps_factor')
        self.args.pop('d_architecture')
        self.args.pop('d_mbstd_groups')

        point_sampling_kwargs = dict(
            image_boundary_value=1.0,
            x_axis_right=True,
            y_axis_up=True,
            z_axis_out=True,
            radius_strategy='fix',
            radius_fix=self.args.pop('radius_fix'),
            polar_strategy='normal',
            polar_mean=self.args.pop('polar_mean'),
            polar_stddev=self.args.pop('polar_stddev'),
            azimuthal_strategy='normal',
            azimuthal_mean=self.args.pop('azimuthal_mean'),
            azimuthal_stddev=self.args.pop('azimuthal_stddev'),
            fov=self.args.pop('fov'),
            perturbation_strategy=self.args.pop('perturbation_strategy'),
            dis_min=self.args.pop('ray_start'),
            dis_max=self.args.pop('ray_end'),
            num_points=self.args.pop('num_points'))

        ray_marching_kwargs = dict(
            use_mid_point=False,
            density_clamp_mode=self.args.pop('clamp_mode'),
            normalize_radial_dist=False,
            clip_radial_dist=False)
        self.config.full_pipeline=True
        self.config.models.update(
            discriminator=dict(
                model=dict(model_type=DISCRIMINATOR, resolution=resolution),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=d_lr,
                         betas=(d_beta_1, d_beta_2)),
                kwargs_train=dict(enable_amp=self.config.enable_amp),
                kwargs_val=dict(enable_amp=False),
                has_unused_parameters=True,
                broadcast_buffers=False),
            generator=dict(
                model=dict(
                    model_type=GENERATOR,
                    image_resolution=resolution,
                    render_resolution=render_resolution,
                    z_dim=latent_dim,
                    w_dim=latent_dim,
                    mapping_layers=self.args.pop('g_num_mappings'),
                    synthesis_input_dim=3,
                    synthesis_output_dim=256,
                    synthesis_layers=8,
                    grid_scale=0.24,
                    freeze_renderer=True,
                    full_pipeline=True,
                    num_importance=0,
                    smooth_weights=False,
                    point_sampling_kwargs=point_sampling_kwargs,
                    ray_marching_kwargs=ray_marching_kwargs,
                    sphere_init_path='pretrained_renderer/sphere_init.pt'),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=g_lr,
                         betas=(g_beta_1, g_beta_2)),
                kwargs_train=dict(
                    style_mixing_prob=self.args.pop('style_mixing_prob'),
                    enable_amp=self.config.enable_amp),
                kwargs_val=dict(enable_amp=False),
                g_ema_img=self.args.pop('g_ema_img'),
                has_unused_parameters=True,
                broadcast_buffers=True,
                freeze_keywords='*',
                freeze_exclusive_keywords='post_decoder',
            ),
        )

        self.config.loss.update(
            loss_type=LOSS,
            d_loss_kwargs=dict(r1_gamma=self.args.pop('r1_gamma'),
                               latent_gamma=0,
                               camera_gamma=0,
                               batch_split=1,
                               fade_steps=10000,
                               full_pipeline=True),
            g_loss_kwargs=dict(top_k_interval=2000,
                               top_v=0.6),
        )

        # TODO: merge the followings
        self.config.grad_clip = dict(max_norm=10,
                                     norm_type=2)

        self.config.controllers.update(
            DatasetVisualizer=dict(viz_keys='raw_image',
                                   viz_num=(32 if label_dim == 0 else 8),
                                   viz_name='Real Data',
                                   viz_groups=(4 if label_dim == 0 else 1),
                                   viz_classes=min(10, label_dim),
                                   row_major=True,
                                   min_val=min_val,
                                   max_val=max_val,
                                   shuffle=False))

        if self.args.pop('use_ada'):
            self.config.aug.update(
                aug_type='AdaAug',
                # Default augmentation strategy adopted by StyleGAN2-ADA.
                xflip=1,
                rotate90=1,
                xint=1,
                scale=1,
                rotate=1,
                aniso=1,
                xfrac=1,
                brightness=1,
                contrast=1,
                lumaflip=1,
                hue=1,
                saturation=1,
                imgfilter=0,
                noise=0,
                cutout=0)
            self.config.aug_kwargs.update(impl='cuda')
            self.config.controllers.update(
                AdaAugController=dict(every_n_iters=4,
                                      init_p=0.0,
                                      target_p=0.6,
                                      speed_img=500_000,
                                      strategy='adaptive'))

        self.config.metrics.update(
            FID50KFull_256resolution=dict(init_kwargs=dict(name='FID50KFull_256resolution',
                                          latent_dim=latent_dim,
                                          batch_size=16,
                                          label_dim=label_dim,
                                          down_resolution=256),
                            eval_kwargs=dict(
                                generator_smooth=dict(enable_amp=False), ),
                            interval=None,
                            first_iter=False,
                            save_best=True),
            GANSnapshot=dict(init_kwargs=dict(name='snapshot',
                                              latent_dim=latent_dim,
                                              latent_num=32,
                                              label_dim=label_dim,
                                              min_val=min_val,
                                              max_val=max_val),
                             eval_kwargs=dict(
                                 generator_smooth=dict(enable_amp=False), ),
                             interval=None,
                             first_iter=False,
                             save_best=False))
