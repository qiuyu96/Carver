# python3.8
"""Configurations for training GRAM."""

import math
import click
from .base_config import BaseConfig

__all__ = ['GRAMConfig']

RUNNER = 'GRAMRunner'
DATASET = 'GRAMDataset'
DISCRIMINATOR = 'GRAMDiscriminator'
GENERATOR = 'GRAMGenerator'
LOSS = 'GRAMLoss'

PI = math.pi


class GRAMConfig(BaseConfig):
    """Defines the configuration for training GRAM."""

    name = 'gram'
    hint = 'Train a GRAM model.'
    info = '''
To train a GRAM model, the recommended settings are as follows:

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
                help='Maximum pixel value of the training images.'),
            cls.command_option(
                '--use_square', type=cls.bool_type, default=False,
                help='Whether to use square image for training.'),
            cls.command_option(
                '--center_crop', type=cls.bool_type, default=False,
                help='Whether to centrally crop non-square images. This field '
                     'only takes effect when `use_square` is set as `True`.')
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
                '--hidden_dim_sample',
                type=cls.int_type,
                default=128,
                help='Number of hidden dim for MPIPredictor.'),
            cls.command_option(
                '--center_z',
                type=cls.float_type,
                default=-1.5,
                help='the z coordinates of center point for MPIPredictor '),
            cls.command_option(
                '--label_dim',
                type=cls.int_type,
                default=0,
                help='Number of classes in conditioning training. Set to `0` '
                'to disable conditional training.'),
            cls.command_option(
                '--grid_scale',
                type=cls.float_type,
                default=0.24,
                help='box_warp'),
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
                help='Number of mapping layers of generator.')
        ])

        options['Training settings'].extend([
            cls.command_option(
                '--d_lr',
                type=cls.float_type,
                default=0.0002,
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
                default=0.9,
                help='The Adam hyper-parameter `beta_2` for discriminator '
                'optimizer.'),
            cls.command_option(
                '--g_lr',
                type=cls.float_type,
                default=0.00002,
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
                default=0.9,
                help='The Adam hyper-parameter `beta_2` for generator '
                'optimizer.'),
            cls.command_option(
                '--w_moving_decay',
                type=cls.float_type,
                default=0.995,
                help='Decay factor for updating `w_avg`.'),
            cls.command_option(
                '--lr_ratio',
                type=cls.float_type,
                default=1.0,
                help='Ratio for adjusting learning rate of different modules.'),
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
                '--position_gamma',
                type=cls.float_type,
                default=15.0,
                help='Factor to control the strength of camera pose penalty.'),
            cls.command_option(
                '--batch_split',
                type=cls.int_type,
                default=4,
                help='batch split to backwards.'),
            cls.command_option(
                '--grad_clip',
                type=cls.float_type,
                default=1.0,
                help='grad_clip for gradients.'),
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
                '--rendering_resolution',
                type=cls.int_type,
                default=256,
                help='Resolution of volume rendering images.'),
            cls.command_option(
                '--density_clamp_mode',
                type=click.Choice(['softplus', 'relu', 'mipnerf']),
                default='relu',
                help='clamp mode of `sigmas` in intergration process.'),
            cls.command_option(
                '--num_points',
                type=cls.int_type,
                default=64,
                help='Number of uniform samples to take per ray '
                'in coarse pass.'),
            cls.command_option(
                '--num_planes',
                type=cls.int_type,
                default=24,
                help='Number of manifolds to take per ray'),
            cls.command_option(
                '--levels_start',
                type=cls.float_type,
                default=23,
                help='Near level along each ray to start taking samples.'),
            cls.command_option(
                '--levels_end',
                type=cls.float_type,
                default=3,
                help='Far level along each ray to start taking samples.'),
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
                '--use_spherical_uniform_position',
                type=cls.bool_type,
                default=False,
                help='Whether to use spherical uniform position for camera '
                     'pose sampling.'),
            cls.command_option(
                '--fov',
                type=cls.float_type,
                default=12,
                help='Field of view of the camera.'),
            cls.command_option(
                '--delta_modulate_scalar',
                type=cls.float_type,
                default=0.04,
                help='delta_modulate_scalar for delta'),
            cls.command_option(
                '--perturbation_strategy',
                type=click.Choice(
                    ['no', 'middle_uniform', 'uniform', 'self_uniform']),
                default='self_uniform',
                help='clamp mode of `sigmas` in intergration process.'),
            cls.command_option(
                '--white_back', type=cls.bool_type, default=False,
                help='Controls the color of rays that pass through the volume '
                     'without encountering any solid objects. Set to `True` if '
                     'your background is white and set to `False` if the '
                     'background is black.'),
            cls.command_option(
                '--mpi_back', type=cls.bool_type, default=True,
                help='control if use mpi background')

        ])

        return options

    def parse_options(self):
        super().parse_options()

        resolution = self.args.pop('resolution')
        image_channels = self.args.pop('image_channels')
        min_val = self.args.pop('min_val')
        max_val = self.args.pop('max_val')
        use_square = self.args.pop('use_square')
        center_crop = self.args.pop('center_crop')

        # Parse data transformation settings.
        data_transform_kwargs = dict(
            image_size=resolution,
            image_channels=image_channels,
            min_val=min_val,
            max_val=max_val,
            use_square=use_square,
            center_crop=center_crop,
            resize_size_pre=320,
            crop_size_pre=256
        )
        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_kwargs = data_transform_kwargs
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_kwargs = data_transform_kwargs


        latent_dim = self.args.pop('latent_dim')
        label_dim = self.args.pop('label_dim')
        self.args.pop('g_init_res')
        self.args.pop('d_mbstd_groups')
        self.args.pop('d_fmaps_factor')
        self.args.pop('g_fmaps_factor')
        rendering_resolution = self.args.pop(
            'rendering_resolution')

        point_sampling_kwargs = dict(
            image_boundary_value=1.0,
            x_axis_right=True,
            y_axis_up=True,
            z_axis_out=True,
            dis_min=self.args.pop('ray_start'),
            dis_max=self.args.pop('ray_end'),
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
            num_points=self.args.pop('num_points'),
            use_spherical_uniform_position=self.args.pop(
                'use_spherical_uniform_position'))

        ray_marching_kwargs = dict(
            use_mid_point=False,
            density_clamp_mode=self.args.pop('density_clamp_mode'),
            color_clamp_mode='widen_sigmoid',
            use_white_background=self.args.pop('white_back'),
            scale_color=True,
            delta_modulate_scalar=self.args.pop('delta_modulate_scalar'),
            normalize_radial_dist=True)

        lr_ratio = self.args.pop('lr_ratio')
        if lr_ratio == 1.0:
            opt_dict = dict(opt_type='Adam',
                            base_lr=self.args.pop('g_lr'),
                            betas=(self.args.pop('g_beta_1'),
                                   self.args.pop('g_beta_2')))
        else:
            opt_dict = dict(
                opt_type='Adam',
                base_lr=self.args.pop('g_lr'),
                betas=(self.args.pop('g_beta_1'), self.args.pop('g_beta_2')),
                lr_ratio_dict=dict(ref_representation_generator=lr_ratio))

        self.config.models.update(
            discriminator=dict(
                model=dict(model_type=DISCRIMINATOR,
                           resolution=resolution,
                           latent_dim=latent_dim,
                           label_dim=label_dim),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=self.args.pop('d_lr'),
                         betas=(self.args.pop('d_beta_1'),
                                self.args.pop('d_beta_2'))),
                kwargs_train=dict(enable_amp=self.config.enable_amp),
                kwargs_val=dict(enable_amp=False),
                has_unused_parameters=True),
            generator=dict(
                model=dict(
                    model_type=GENERATOR,
                    resolution=resolution,
                    z_dim=latent_dim,
                    w_dim=latent_dim,
                    repeat_w=True,
                    mapping_layers=self.args.pop('g_num_mappings'),
                    synthesis_input_dim=3,
                    synthesis_output_dim=256,
                    synthesis_layers=8,
                    hidden_dim_sample=self.args.pop('hidden_dim_sample'),
                    center_z=self.args.pop('center_z'),
                    point_sampling_kwargs=point_sampling_kwargs,
                    ray_marching_kwargs=ray_marching_kwargs,
                    rendering_resolution=rendering_resolution,
                    num_planes=self.args.pop('num_planes'),
                    levels_start=self.args.pop('levels_start'),
                    levels_end=self.args.pop('levels_end'),
                    grid_scale=self.args.pop('grid_scale'),
                    use_mpi_background=self.args.pop('mpi_back')),
                lr=dict(lr_type='FIXED'),
                opt=opt_dict,
                kwargs_train=dict(
                    w_moving_decay=self.args.pop('w_moving_decay'),
                    sync_w_avg=self.args.pop('sync_w_avg'),
                    style_mixing_prob=self.args.pop('style_mixing_prob'),
                    enable_amp=self.config.enable_amp),
                kwargs_val=dict(enable_amp=False),
                g_ema_img=self.args.pop('g_ema_img'),
                has_unused_parameters=True))

        self.config.loss.update(
            loss_type=LOSS,
            d_loss_kwargs=dict(r1_gamma=self.args.pop('r1_gamma'),
                               latent_gamma=0,
                               position_gamma=self.args.pop('position_gamma'),
                               batch_split=self.args.pop('batch_split'),
                               fade_steps=10000),
            g_loss_kwargs=dict(top_k_interval=2000, top_v=0.6),
        )

        self.config.grad_clip = dict(max_norm=self.args.pop('grad_clip'),
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
            FID50KFull=dict(init_kwargs=dict(name='fid50k',
                                             latent_dim=latent_dim,
                                             label_dim=label_dim,
                                             image_size = resolution),
                            eval_kwargs=dict(
                                generator_smooth=dict(enable_amp=False), ),
                            interval=None,
                            first_iter=True,
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
                             first_iter=None,
                             save_best=False))
