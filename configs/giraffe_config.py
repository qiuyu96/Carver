# python3.8
"""Configuration for training GIRAFFE."""

from .base_config import BaseConfig
import math
import click
import torch

__all__ = ['GIRAFFEConfig']

RUNNER = 'GIRAFFERunner'
DATASET = 'ImageDataset'
DISCRIMINATOR = 'GIRAFFEDiscriminator'
GENERATOR = 'GIRAFFEGenerator'
LOSS = 'GIRAFFELoss'
PI = math.pi


class GIRAFFEConfig(BaseConfig):
    """Defines the configuration for training GIRAFFE."""

    name = 'giraffe'
    hint = 'Train a GIRAFFE model.'
    info = '''
To train a GIRAFFE model, the recommend settings are as follows:

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
                               default=128,
                               help='Resolution of the training images.'),
            cls.command_option(
                '--image_channels',
                type=cls.int_type,
                default=3,
                help='Number of channels of the training images.'),
            cls.command_option(
                '--min_val',
                type=cls.float_type,
                default=0.0,
                help='Minimum pixel value of the training images.'),
            cls.command_option(
                '--max_val',
                type=cls.float_type,
                default=1.0,
                help='Maximum pixel value of the training images.'),
            cls.command_option(
                '--resize_size', type=cls.int_type, default=0,
                help='Size for resizing images before cropping. `0` means no '
                     'cropping.'),
            cls.command_option(
                '--crop_size', type=cls.int_type, default=0,
                help='Size for cropping images. `0` means no cropping.')
        ])

        options['Network settings'].extend([
            cls.command_option('--latent_dim',
                               type=cls.int_type,
                               default=256,
                               help='The dimension of the latent space Z.'),
            cls.command_option('--latent_dim_bg',
                               type=cls.int_type,
                               default=128,
                               help='The dimension of the latent space Z for '
                                    'the background.'),
            cls.command_option('--label_dim',
                type=cls.int_type,
                default=0,
                help='Number of classes in conditioning training. Set to `0` '
                     'to disable conditional training.'),
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
                '--r1_gamma',
                type=cls.float_type,
                default=10.0,
                help='Factor to control the strength of gradient penalty.'),
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
        ])

        options['Rendering options'].extend([
            cls.command_option(
                '--rendering_resolution',
                type=cls.int_type,
                default=32,
                help='Resolution of volume rendering images.'),
            cls.command_option(
                '--num_points',
                type=cls.int_type,
                default=64,
                help='Number of uniform samples to take per ray '
                'in coarse pass.'),
            cls.command_option(
                '--depth_min',
                type=cls.float_type,
                default=0.5,
                help='Near point along each ray to start taking samples.'),
            cls.command_option(
                '--depth_max',
                type=cls.float_type,
                default=6.0,
                help='Far point along each ray to start taking samples.'),
            cls.command_option(
                '--radius_fix',
                type=cls.float_type,
                default=2.732,
                help='Radius of sphere for sampling camera position.'),
            cls.command_option(
                '--polar_min',
                type=cls.float_type,
                default=0.4167,
                help='Minimum value of polar (vertical) angle for sampling '
                'camera position.'),
            cls.command_option(
                '--polar_max',
                type=cls.float_type,
                default=0.5,
                help='Maximum value of polar (vertical) angle for sampling '
                'camera position.'),
            cls.command_option(
                '--polar_mean',
                type=cls.float_type,
                default=PI / 2,
                help='Mean of polar (vertical) angle for sampling camera '
                'position.'),
            cls.command_option(
                '--polar_stddev',
                type=cls.float_type,
                default=PI / 2,
                help='Standard deviation of polar (vertical) angle of sphere '
                'for sampling camera position.'),
            cls.command_option(
                '--azimuthal_min',
                type=cls.float_type,
                default=0,
                help='Minimum value of azimuthal (horizontal) angle for '
                'sampling camera position.'),
            cls.command_option(
                '--azimuthal_max',
                type=cls.float_type,
                default=0,
                help='Maximum value of azimuthal (horizontal) angle for '
                'sampling camera position.'),
            cls.command_option(
                '--azimuthal_mean',
                type=cls.float_type,
                default=PI,
                help='Mean of azimuthal (horizontal) angle for sampling camera '
                'position.'),
            cls.command_option(
                '--azimuthal_stddev',
                type=cls.float_type,
                default=PI,
                help='Standard deviation of azimuthal (horizontal) angle of '
                'sphere for sampling camera position.'),
            cls.command_option(
                '--fov',
                type=cls.float_type,
                default=10,
                help='Field of view of the camera.'),
            cls.command_option(
                '--scale_range_min',
                type=cls.list_type,
                default='0.21, 0.21, 0.21',
                help='Minimum value of scale transformation. '
                'Each value in the list represents for x, y, z, respectively.'),
            cls.command_option(
                '--scale_range_max',
                type=cls.list_type,
                default='0.21, 0.21, 0.21',
                help='Maximum value of scale transformation. '
                'Each value in the list represents for x, y, z, respectively.'),
            cls.command_option(
                '--translation_range_min',
                type=cls.list_type,
                default='0., 0., 0.',
                help='Minimum value of translation. '
                'Each value in the list represents for x, y, z, respectively.'),
            cls.command_option(
                '--translation_range_max',
                type=cls.list_type,
                default='0., 0., 0.',
                help='Maximum value of translation. '
                'Each value in the list represents for x, y, z, respectively.'),
            cls.command_option(
                '--rotation_range',
                type=cls.list_type,
                default='0.40278, 0.59722',
                help='Range of rotation.')
        ])

        return options

    def parse_options(self):
        super().parse_options()

        resolution = self.args.pop('resolution')
        image_channels = self.args.pop('image_channels')
        min_val = self.args.pop('min_val')
        max_val = self.args.pop('max_val')

        # Parse data transformation settings.
        data_transform_kwargs = dict(
            image_size=resolution,
            image_channels=image_channels,
            min_val=min_val,
            max_val=max_val,
            resize_size_pre=self.args.pop('resize_size'),
            crop_size_pre=self.args.pop('crop_size'))
        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_kwargs = data_transform_kwargs
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_kwargs = data_transform_kwargs

        latent_dim = self.args.pop('latent_dim')
        latent_dim_bg = self.args.pop('latent_dim_bg')
        label_dim = self.args.pop('label_dim')

        d_lr = self.args.pop('d_lr')
        d_beta_1 = self.args.pop('d_beta_1')
        d_beta_2 = self.args.pop('d_beta_2')
        g_lr = self.args.pop('g_lr')
        g_beta_1 = self.args.pop('g_beta_1')
        g_beta_2 = self.args.pop('g_beta_2')

        rendering_resolution = self.args.pop('rendering_resolution')

        ray_marching_kwargs = dict(
            use_mid_point=False,
            density_clamp_mode='relu',
            normalize_radial_dist=False,
            clip_radial_dist=False,
            scale_color=False)

        bbox_generator_kwargs = dict(
            scale_range_min=self.args.pop('scale_range_min'),
            scale_range_max=self.args.pop('scale_range_max'),
            translation_range_min=self.args.pop('translation_range_min'),
            translation_range_max=self.args.pop('translation_range_max'),
            rotation_range=self.args.pop('rotation_range'))

        self.config.models.update(
            discriminator=dict(
                model=dict(
                    model_type=DISCRIMINATOR,
                    img_size=resolution),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                        base_lr=d_lr,
                        betas=(d_beta_1, d_beta_2)),
                kwargs_train=dict(),
                kwargs_val=dict(),
                has_unused_parameters=True),
            generator=dict(
                model=dict(
                    model_type=GENERATOR,
                    z_dim=latent_dim,
                    z_dim_bg=latent_dim_bg,
                    fov=self.args.pop('fov'),
                    radius=self.args.pop('radius_fix'),
                    polar_min=self.args.pop('polar_min'),
                    polar_max=self.args.pop('polar_max'),
                    polar_mean=self.args.pop('polar_mean'),
                    polar_stddev=self.args.pop('polar_stddev'),
                    azimuthal_min=self.args.pop('azimuthal_min'),
                    azimuthal_max=self.args.pop('azimuthal_max'),
                    azimuthal_mean=self.args.pop('azimuthal_mean'),
                    azimuthal_stddev=self.args.pop('azimuthal_stddev'),
                    depth_min=self.args.pop('depth_min'),
                    depth_max=self.args.pop('depth_max'),
                    num_points_per_ray=self.args.pop('num_points'),
                    rendering_resolution=rendering_resolution,
                    full_resolution=resolution,
                    ray_marching_kwargs=ray_marching_kwargs,
                    bounding_box_generator_kwargs=bbox_generator_kwargs),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=g_lr,
                         betas=(g_beta_1, g_beta_2)),
                kwargs_train=dict(),
                kwargs_val=dict(),
                g_ema_img=self.args.pop('g_ema_img'),
                g_ema_rampup=self.args.pop('g_ema_rampup'),
                has_unused_parameters=True))

        self.config.loss.update(
            loss_type=LOSS,
            d_loss_kwargs=dict(r1_gamma=self.args.pop('r1_gamma')),
            g_loss_kwargs=dict())

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
            FID=dict(init_kwargs=dict(name='fid',
                                      latent_dim=latent_dim,
                                      label_dim=label_dim,
                                      real_num=20000,
                                      fake_num=20000,
                                      image_size=resolution),
                     eval_kwargs=dict(generator_smooth=dict(
                         batch_size=self.config.val_batch_size,
                         device=torch.cuda.current_device()), ),
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
                                 batch_size=self.config.val_batch_size,
                                 device=torch.cuda.current_device()), ),
                             interval=None,
                             first_iter=None,
                             save_best=False))
