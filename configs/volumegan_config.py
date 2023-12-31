# python3.8
"""Configuration for training VolumeGAN."""

from .base_config import BaseConfig
import math
import click
import torch

__all__ = ['VolumeGANConfig']

RUNNER = 'VolumeGANRunner'
DATASET = 'ImageDataset'
DISCRIMINATOR = 'VolumeGANDiscriminator'
GENERATOR = 'VolumeGANGenerator'
LOSS = 'VolumeGANLoss'
PI = math.pi


class VolumeGANConfig(BaseConfig):
    """Defines the configuration for training VolumeGAN."""

    name = 'volumegan'
    hint = 'Train a VolumeGAN model.'
    info = '''
To train a VolumeGAN model, the recommend settings are as follows:

\b
- batch_size: 8 (for FF-HQ dataset, 8 GPU)
- val_batch_size: 8 (for FF-HQ dataset, 8 GPU)
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
            cls.command_option('--resolution',
                               type=cls.int_type,
                               default=256,
                               help='Resolution of the training images.'),
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
                help='Maximum pixel value of the training images.')
        ])

        options['Network settings'].extend([
            cls.command_option(
                '--g_init_res',
                type=cls.int_type,
                default=32,
                help='The initial resolution to start convolution with in '
                'generator.'),
            cls.command_option(
                '--g_rgb_init_res_out',
                type=cls.bool_type,
                default=False,
                help='Whether to use rgb head to output initial resolution. '),
            cls.command_option('--latent_dim',
                               type=cls.int_type,
                               default=512,
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
                'discriminator, which will be `factor * 32768`.'),
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
                'generator, which will be `factor * 32768`.'),
            cls.command_option('--g_num_mappings',
                               type=cls.int_type,
                               default=8,
                               help='Number of mapping layers of generator.'),
            cls.command_option('--d_architecture',
                               type=str,
                               default='resnet',
                               help='Architecture type of discriminator.'),
            cls.command_option(
                '--impl',
                type=str,
                default='cuda',
                help='Control the implementation of some neural operations.'),
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
            cls.command_option('--d_lr',
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
            cls.command_option('--g_lr',
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
            cls.command_option('--w_moving_decay',
                               type=cls.float_type,
                               default=0.995,
                               help='Decay factor for updating `w_avg`.'),
            cls.command_option(
                '--sync_w_avg',
                type=cls.bool_type,
                default=False,
                help='Synchronizing the update of `w_avg` across replicas.'),
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
                '--r1_interval',
                type=cls.int_type,
                default=16,
                help='Interval (in iterations) to perform gradient penalty.'),
            cls.command_option(
                '--pl_batch_shrink',
                type=cls.int_type,
                default=2,
                help='Factor to reduce the batch size for perceptual path '
                'length regularization.'),
            cls.command_option(
                '--pl_weight',
                type=cls.float_type,
                default=2.0,
                help='Factor to control the strength of perceptual path length '
                'regularization.'),
            cls.command_option(
                '--pl_decay',
                type=cls.float_type,
                default=0.01,
                help='Decay factor for perceptual path length regularization.'
            ),
            cls.command_option(
                '--pl_interval',
                type=cls.int_type,
                default=4,
                help='Interval (in iterations) to perform perceptual path '
                'length regularization.'),
            cls.command_option(
                '--g_ema_img',
                type=cls.int_type,
                default=10_000,
                help='Factor for updating the smoothed generator, which is '
                'particularly used for inference.'),
            cls.command_option(
                '--g_ema_rampup',
                type=cls.float_type,
                default=0.0,
                help='Rampup factor for updating the smoothed generator, which '
                'is particularly used for inference. Set as `0` to '
                'disable warming up.'),
            cls.command_option(
                '--use_ada',
                type=cls.bool_type,
                default=False,
                help='Whether to use adaptive augmentation pipeline.'),
            cls.command_option(
                '--use_pg',
                type=cls.bool_type,
                default=False,
                help='Whether to use adaptive augmentation pipeline.'),
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
                default=12,
                help='Number of uniform samples to take per ray '
                'in coarse pass.'),
            cls.command_option(
                '--num_importance',
                type=cls.int_type,
                default=12,
                help='Number of importance samples to take per ray '
                'in fine pass.'),
            cls.command_option(
                '--ray_start',
                type=cls.float_type,
                default=0.8,
                help='Near point along each ray to start taking samples.'),
            cls.command_option(
                '--ray_end',
                type=cls.float_type,
                default=1.2,
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
            cls.command_option('--fov',
                               type=cls.float_type,
                               default=12,
                               help='Field of view of the camera.'),
            cls.command_option(
                '--perturbation_strategy',
                type=click.Choice(
                    ['no', 'middle_uniform', 'uniform', 'self_uniform']),
                default='self_uniform',
                help='clamp mode of `sigmas` in intergration process.')
        ])

        return options

    def parse_options(self):
        super().parse_options()

        resolution = self.args.pop('resolution')
        image_channels = self.args.pop('image_channels')
        min_val = self.args.pop('min_val')
        max_val = self.args.pop('max_val')

        # Parse data transformation settings.
        data_transform_kwargs = dict(image_size=resolution,
                                     image_channels=image_channels,
                                     min_val=min_val,
                                     max_val=max_val)
        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_kwargs = data_transform_kwargs
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_kwargs = data_transform_kwargs

        g_init_res = self.args.pop('g_init_res')
        d_init_res = 4  # This should be fixed as 4.
        latent_dim = self.args.pop('latent_dim')
        label_dim = self.args.pop('label_dim')
        d_fmaps_base = int(self.args.pop('d_fmaps_factor') * (32 << 10))
        g_fmaps_base = int(self.args.pop('g_fmaps_factor') * (32 << 10))
        impl = self.args.pop('impl')
        num_fp16_res = self.args.pop('num_fp16_res')

        # Parse network settings and training settings.
        if not isinstance(num_fp16_res, int) or num_fp16_res <= 0:
            d_fp16_res = None
            g_fp16_res = None
            conv_clamp = None
        else:
            d_fp16_res = max(resolution // (2**(num_fp16_res - 1)),
                             d_init_res * 2)
            g_fp16_res = max(resolution // (2**(num_fp16_res - 1)),
                             g_init_res * 2)
            conv_clamp = 256

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
            d_beta_1 = d_beta_1**d_mb_ratio
            d_beta_2 = d_beta_2**d_mb_ratio
        if pl_interval is not None and pl_interval > 0:
            g_mb_ratio = pl_interval / (pl_interval + 1)
            g_lr = g_lr * g_mb_ratio
            g_beta_1 = g_beta_1**g_mb_ratio
            g_beta_2 = g_beta_2**g_mb_ratio

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

        self.config.models.update(
            discriminator=dict(model=dict(
                model_type=DISCRIMINATOR,
                resolution=resolution,
                image_channels=image_channels,
                init_res=d_init_res,
                label_dim=label_dim,
                architecture=self.args.pop('d_architecture'),
                fmaps_base=d_fmaps_base,
                conv_clamp=conv_clamp,
                mbstd_groups=self.args.pop('d_mbstd_groups')),
                               lr=dict(lr_type='FIXED'),
                               opt=dict(opt_type='Adam',
                                        base_lr=d_lr,
                                        betas=(d_beta_1, d_beta_2)),
                               kwargs_train=dict(fp16_res=d_fp16_res,
                                                 impl=impl),
                               kwargs_val=dict(fp16_res=None, impl=impl),
                               has_unused_parameters=True),
            generator=dict(
                model=dict(
                    model_type=GENERATOR,
                    z_dim=latent_dim,
                    mapping_layers=self.args.pop('g_num_mappings'),
                    fmaps_base=g_fmaps_base,
                    conv_clamp=conv_clamp,
                    rgb_init_res_out=self.args.pop('g_rgb_init_res_out'),
                    nerf_res=g_init_res,
                    resolution=resolution,
                    num_importance=self.args.pop('num_importance'),
                    point_sampling_kwargs=point_sampling_kwargs,
                    ray_marching_kwargs=ray_marching_kwargs),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=g_lr,
                         betas=(g_beta_1, g_beta_2)),
                # Please turn off `fused_modulate` during training, which is
                # because the customized gradient computation omits weights, and
                # the fused operation will introduce division by 0.
                kwargs_train=dict(
                    w_moving_decay=self.args.pop('w_moving_decay'),
                    sync_w_avg=self.args.pop('sync_w_avg'),
                    style_mixing_prob=self.args.pop('style_mixing_prob'),
                    noise_mode='random',
                    fused_modulate=False,
                    fp16_res=g_fp16_res,
                    impl=impl,
                ),
                kwargs_val=dict(
                    noise_mode='const',
                    fused_modulate=False,
                    fp16_res=None,
                    impl=impl,
                ),
                g_ema_img=self.args.pop('g_ema_img'),
                g_ema_rampup=self.args.pop('g_ema_rampup'),
                has_unused_parameters=True))

        self.config.loss.update(
            loss_type=LOSS,
            d_loss_kwargs=dict(r1_gamma=self.args.pop('r1_gamma'),
                               r1_interval=r1_interval),
            g_loss_kwargs=dict(
                pl_batch_shrink=self.args.pop('pl_batch_shrink'),
                pl_weight=self.args.pop('pl_weight'),
                pl_decay=self.args.pop('pl_decay'),
                pl_interval=pl_interval))

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

        use_pg = self.args.pop('use_pg')
        if use_pg:
            self.config.controllers.update(ProgressScheduler=dict(
                init_res=g_init_res,
                final_res=resolution,
                minibatch_repeats=4,
                lod_training_img=750_000,
                lod_transition_img=750_000,
                batch_size_schedule=dict(res32=8, res64=8, res128=8),
                lr_schedule=dict(res32=1, res64=1, res128=1, res256=1),
            ))

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
            FID50KFull=dict(init_kwargs=dict(name='fid50k',
                                             latent_dim=latent_dim,
                                             label_dim=label_dim),
                            eval_kwargs=dict(generator_smooth=dict(
                                noise_mode='random',
                                fused_modulate=False,
                                fp16_res=None,
                                impl=impl,
                            ), ),
                            interval=None,
                            first_iter=None,
                            save_best=True),
            GANSnapshot=dict(init_kwargs=dict(name='snapshot',
                                              latent_dim=latent_dim,
                                              latent_num=32,
                                              label_dim=label_dim,
                                              min_val=min_val,
                                              max_val=max_val),
                             eval_kwargs=dict(generator_smooth=dict(
                                 noise_mode='const',
                                 fused_modulate=False,
                                 fp16_res=None,
                                 impl=impl,
                             ), ),
                             interval=None,
                             first_iter=None,
                             save_best=False))
