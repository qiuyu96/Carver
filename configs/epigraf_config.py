# python3.8
"Configuration for training EpiGRAF."

import click
from .base_config import BaseConfig

__all__ = ['EpiGRAFConfig']

RUNNER = 'EpiGRAFRunner'
DATASET = 'EpiGRAFDataset'
DISCRIMINATOR = 'EpiGRAFDiscriminator'
GENERATOR = 'EpiGRAFGenerator'
LOSS = 'EpiGRAFLoss'

class EpiGRAFConfig(BaseConfig):
    """Defines the configuration for training EpiGRAF."""

    name = 'epigraf'
    hint = 'Train a EpiGRAF model.'
    info = '''
To train a EpiGRAF model, the recommended settings are as follows:

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
                '--resolution', type=cls.int_type, default=256,
                help='Resolution of the training images.'),
            cls.command_option(
                '--image_channels', type=cls.int_type, default=3,
                help='Number of channels of the training images.'),
            cls.command_option(
                '--min_val', type=cls.float_type, default=-1.0,
                help='Minimum pixel value of the training images.'),
            cls.command_option(
                '--max_val', type=cls.float_type, default=1.0,
                help='Maximum pixel value of the training images.'),
            cls.command_option(
                '--use_square', type=cls.bool_type, default=False,
                help='Whether to use square image for training.'),
            cls.command_option(
                '--center_crop', type=cls.bool_type, default=False,
                help='Whether to centrally crop non-square images. This field '
                     'only takes effect when `use_square` is set as `True`.'),
            cls.command_option(
                '--pose_meta', type=str, default=None,
                help='Name of the pose meta file.')
        ])

        options['Network settings'].extend([
            cls.command_option(
                '--g_init_res', type=cls.int_type, default=4,
                help='The initial resolution to start convolution with in '
                     'generator.'),
            cls.command_option(
                '--latent_dim', type=cls.int_type, default=512,
                help='The dimension of the latent space.'),
            cls.command_option(
                '--label_dim', type=cls.int_type, default=0,
                help='Number of classes in conditioning training. Set to `0` '
                     'to disable conditional training.'),
            cls.command_option(
                '--pose_dim', type=cls.int_type, default=3,
                help='Dimension of conditoned pose information, default: 3.'),
            cls.command_option(
                '--d_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'discriminator, which will be `factor * 32768`.'),
            cls.command_option(
                '--d_mbstd_groups', type=cls.int_type, default=4,
                help='Number of groups for MiniBatchSTD layer of '
                     'discriminator.'),
            cls.command_option(
                '--g_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'generator, which will be `factor * g_channel_base`.'),
            cls.command_option(
                '--d_channel_base', type=cls.int_type, default=16384,
                help='Capacity multiplier of discriminator.'),
            cls.command_option(
                '--g_channel_base', type=cls.int_type, default=32768,
                help='Capacity multiplier of generator.'),
            cls.command_option(
                '--channel_max', type=cls.int_type, default=512,
                help='Maximum feature maps.'),
            cls.command_option(
                '--freezed', type=cls.int_type, default=0,
                help='Freeze first `n` layers of D.'),
            cls.command_option(
                '--impl', type=str, default='cuda',
                help='Control the implementation of some neural operations.'),
        ])

        options['Training settings'].extend([
            cls.command_option(
                '--d_lr', type=cls.float_type, default=0.002,
                help='The learning rate of discriminator.'),
            cls.command_option(
                '--d_beta_1', type=cls.float_type, default=0.0,
                help='The Adam hyper-parameter `beta_1` for discriminator '
                     'optimizer.'),
            cls.command_option(
                '--d_beta_2', type=cls.float_type, default=0.99,
                help='The Adam hyper-parameter `beta_2` for discriminator '
                     'optimizer.'),
            cls.command_option(
                '--g_lr', type=cls.float_type, default=0.0025,
                help='The learning rate of generator.'),
            cls.command_option(
                '--g_beta_1', type=cls.float_type, default=0.0,
                help='The Adam hyper-parameter `beta_1` for generator '
                     'optimizer.'),
            cls.command_option(
                '--g_beta_2', type=cls.float_type, default=0.99,
                help='The Adam hyper-parameter `beta_2` for generator '
                     'optimizer.'),
            cls.command_option(
                '--num_mapping_layers', type=cls.int_type, default=2,
                help='Number of mapping network layers.'),
            cls.command_option(
                '--sync_w_avg', type=cls.bool_type, default=False,
                help='Synchronizing the update of `w_avg` across replicas.'),
            cls.command_option(
                '--style_mixing_prob', type=cls.float_type, default=0.0,
                help='Probability to perform style mixing as a training '
                     'regularization.'),
            cls.command_option(
                '--r1_gamma', type=cls.float_type, default=10.0,
                help='Factor to control the strength of gradient penalty.'),
            cls.command_option(
                '--r1_interval', type=cls.int_type, default=16,
                help='Interval (in iterations) to perform gradient penalty.'),
            cls.command_option(
                '--pl_batch_shrink', type=cls.int_type, default=2,
                help='Factor to reduce the batch size for perceptual path '
                     'length regularization.'),
            cls.command_option(
                '--pl_weight', type=cls.float_type, default=2.0,
                help='Factor to control the strength of perceptual path length '
                     'regularization.'),
            cls.command_option(
                '--pl_decay', type=cls.float_type, default=0.01,
                help='Decay factor for perceptual path length regularization.'),
            cls.command_option(
                '--pl_interval', type=cls.int_type, default=4,
                help='Interval (in iterations) to perform perceptual path '
                     'length regularization.'),
            cls.command_option(
                '--g_ema_img', type=cls.int_type, default=20_000,
                help='Factor for updating the smoothed generator, which is '
                     'particularly used for inference.'),
            cls.command_option(
                '--use_ada', type=cls.bool_type, default=False,
                help='Whether to use adaptive augmentation pipeline.'),
            cls.command_option(
                '--g_ema_rampup', type=cls.float_type, default=0.05,
                help='Rampup factor for updating the smoothed generator, which '
                     'is particularly used for inference. Set as `0` to '
                     'disable warming up.'),
            cls.command_option(
                '--rendering_resolution', type=cls.int_type,
                default=64,
                help='Resolution of neural rendering.'),
            cls.command_option(
                '--blur_fade_kimg', type=cls.int_type, default=200,
                help='Blur over how many.'),
            cls.command_option(
                '--blur_init_sigma', type=cls.float_type, default=10.0,
                help='Blur the images seen by the discriminator.'),
            cls.command_option(
                '--filter_mode', type=str, default='antialiased',
                help='Filter mode for raw images '
                     '[antialiased, none, float [0-1]]'),
            cls.command_option(
                '--blur_raw_target', type=cls.bool_type, default=True,
                help='Whether blur the raw target or not when send to D.'),
            cls.command_option(
                '--g_num_fp16_res', type=cls.int_type, default=0,
                help='Number of fp16 layers in generator.'),
            cls.command_option(
                '--d_num_fp16_res', type=cls.int_type, default=4,
                help='Number of fp16 layers in discriminator.'),
            cls.command_option(
                '--test_fid50k', type=cls.bool_type, default=True,
                help='Whethter to test fid50k.'),
            cls.command_option(
                '--test_fid2k', type=cls.bool_type, default=False,
                help='Whethter to test fid2k.'),
        ])

        options['Rendering options'].extend([
            cls.command_option(
                '--disc_c_noise', type=cls.float_type, default=0.0,
                help='Strength of discriminator pose conditioning '
                     'regularization, in standard deviations.'),
            cls.command_option(
                '--num_points', type=cls.int_type, default=48,
                help='Number of uniform samples to take per ray '
                     'in coarse pass.'),
            cls.command_option(
                '--num_importance', type=cls.int_type, default=48,
                help='Number of importance samples to take per ray '
                     'in fine pass.'),
            cls.command_option(
                '--ray_start', type=cls.float_type, default=0.88,
                help='Near point along each ray to start taking samples.'),
            cls.command_option(
                '--ray_end', type=cls.float_type, default=1.12,
                help='Far point along each ray to start taking samples.'),
            cls.command_option(
                '--camera_radius', type=cls.float_type, default=1.0,
                help='Radius of camera orbit.'),
            cls.command_option(
                '--coordinate_scale', type=cls.float_type, default=1.0,
                help='Scale factor to modulate coordinates before retrieving '
                     'features from triplanes.'),
            cls.command_option('--fov',
                               type=cls.float_type,
                               default=12,
                               help='Field of view of the camera.'),
            cls.command_option('--scale_color',
                               type=cls.bool_type,
                               default=False,
                               help='Whether to scale color value to [-1, 1].')
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
        impl = self.args.pop('impl')

        # Parse data transformation settings.
        data_transform_kwargs = dict(
            image_size=512,
            image_channels=image_channels,
            min_val=min_val,
            max_val=max_val,
            use_square=use_square,
            center_crop=center_crop
        )

        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_kwargs = data_transform_kwargs
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_kwargs = data_transform_kwargs
        pose_meta = self.args.pop('pose_meta')
        if pose_meta is not None:
            self.config.data.train.pose_meta = pose_meta
            self.config.data.val.pose_meta = pose_meta

        latent_dim = self.args.pop('latent_dim')
        label_dim = self.args.pop('label_dim')
        pose_dim = self.args.pop('pose_dim')
        self.args.pop('g_init_res')
        self.args.pop('d_fmaps_factor')
        self.args.pop('g_fmaps_factor')
        self.args.pop('sync_w_avg')
        disc_c_noise = self.args.pop('disc_c_noise')
        d_channel_base = self.args.pop('d_channel_base')
        g_channel_base = self.args.pop('g_channel_base')
        channel_max = self.args.pop('channel_max')
        freezed = self.args.pop('freezed')
        d_mbstd_groups = self.args.pop('d_mbstd_groups')
        d_num_fp16_res = self.args.pop('d_num_fp16_res')
        g_num_fp16_res = self.args.pop('g_num_fp16_res')
        d_conv_clamp = 256 if d_num_fp16_res > 0 else None

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

        rendering_resolution = self.args.pop('rendering_resolution')
        patch_sampling_kwargs = dict(
            patch_resolution=rendering_resolution)
        self.config.patch_sampling_kwargs = patch_sampling_kwargs

        point_sampling_kwargs = dict(image_boundary_value=1.0,
                                     x_axis_right=True,
                                     y_axis_up=True,
                                     z_axis_out=True,
                                     dis_min=self.args.pop('ray_start'),
                                     dis_max=self.args.pop('ray_end'),
                                     fov=self.args.pop('fov'),
                                     num_points=self.args.pop('num_points'),
                                     perturbation_strategy='middle_uniform')
        ray_marching_kwargs = dict(
            use_mid_point=False,
            density_clamp_mode='softplus',
            use_white_background=False,
            scale_color=self.args.pop('scale_color'))

        self.config.style_mixing_prob = self.args.pop('style_mixing_prob')
        self.config.models.update(
            discriminator=dict(model=dict(
                model_type=DISCRIMINATOR,
                c_dim=pose_dim,
                img_resolution=rendering_resolution,
                img_channels=image_channels,
                channel_base=d_channel_base,
                channel_max=channel_max,
                block_kwargs=dict(freeze_layers=freezed),
                epilogue_kwargs=dict(mbstd_group_size=d_mbstd_groups),
                disc_c_noise=disc_c_noise,
                num_fp16_res=d_num_fp16_res,
                conv_clamp=d_conv_clamp,
            ),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=d_lr,
                         betas=(d_beta_1, d_beta_2)),
                has_unused_parameters=True),
            generator=dict(
                model=dict(
                    model_type=GENERATOR,
                    z_dim=512,
                    label_dim=pose_dim,
                    w_dim=512,
                    image_channels=image_channels,
                    mapping_layers=self.args.pop('num_mapping_layers'),
                    camera_cond=True,
                    camera_raw_scalars=True,
                    include_cam_input=False,
                    coordinate_scale=self.args.pop('coordinate_scale'),
                    triplane_resolution=512,
                    triplane_channels=32 * 3,
                    channel_base=g_channel_base,
                    channel_max=channel_max,
                    fused_modconv_default='inference_only',
                    num_fp16_res=g_num_fp16_res,
                    point_sampling_kwargs=point_sampling_kwargs,
                    ray_marching_kwargs=ray_marching_kwargs,
                    resolution=resolution,
                    rendering_resolution=rendering_resolution,
                    num_importance=self.args.pop('num_importance'),
                    density_noise_std=0.0,
                    gpc_spoof_p=1.0,
                    camera_radius=self.args.pop('camera_radius')),
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
            d_loss_kwargs=dict(
                r1_gamma=self.args.pop('r1_gamma'),
                r1_interval=r1_interval,
                blur_init_sigma=self.args.pop('blur_init_sigma'),
                blur_fade_kimg=self.args.pop('blur_fade_kimg'),
                filter_mode=self.args.pop('filter_mode'),
                blur_raw_target=self.args.pop('blur_raw_target'),
            ),
            g_loss_kwargs=dict(
                pl_batch_shrink=self.args.pop('pl_batch_shrink'),
                pl_weight=self.args.pop('pl_weight'),
                pl_decay=self.args.pop('pl_decay'),
                pl_interval=pl_interval
            )
        )

        self.config.controllers.update(
            DatasetVisualizer=dict(
                viz_keys='raw_image',
                viz_num=(32 if label_dim == 0 else 8),
                viz_name='Real Data',
                viz_groups=(4 if label_dim == 0 else 1),
                viz_classes=min(10, label_dim),
                row_major=True,
                min_val=min_val,
                max_val=max_val,
                shuffle=False
            )
        )

        self.config.use_ada = self.args.pop('use_ada')
        if self.config.use_ada:
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
                cutout=0
            )
            self.config.aug_kwargs.update(impl='cuda')
            self.config.controllers.update(
                AdaAugController=dict(
                    every_n_iters=4,
                    init_p=0.0,
                    target_p=0.6,
                    speed_img=500_000,
                    strategy='adaptive'
                )
            )

        self.config.metrics.update(
            GANSnapshot_EG3D_Image=dict(
                init_kwargs=dict(name='snapshot_eg3d_image',
                                 latent_dim=latent_dim,
                                 latent_num=32,
                                 label_dim=label_dim,
                                 min_val=min_val,
                                 max_val=max_val),
                eval_kwargs=dict(
                    generator_smooth=dict(),
                ),
                interval=None,
                first_iter=None,
                save_best=False
            ),
            GANSnapshot_EG3D_Depth=dict(
                init_kwargs=dict(name='snapshot_eg3d_depth',
                                 latent_dim=latent_dim,
                                 latent_num=32,
                                 label_dim=label_dim,
                                 min_val=min_val,
                                 max_val=max_val),
                eval_kwargs=dict(
                    generator_smooth=dict(),
                ),
                interval=None,
                first_iter=None,
                save_best=False
            )
        )

        test_fid50k = self.args.pop('test_fid50k')
        test_fid2k = self.args.pop('test_fid2k')
        assert (test_fid50k and test_fid2k) == False
        assert (test_fid50k or test_fid2k) == True

        if test_fid50k:
            self.config.metrics.update(
                FID50KFullEG3D=dict(
                init_kwargs=dict(name='fid50k_full',
                                 latent_dim=latent_dim,
                                 label_dim=label_dim,
                                 image_size=512),
                eval_kwargs=dict(
                    generator_smooth=dict(),
                ),
                interval=None,
                first_iter=None,
                save_best=True
                )
            )

        if test_fid2k:
            self.config.metrics.update(
                FID2KFullEG3D=dict(
                init_kwargs=dict(name='fid2k_full',
                                 latent_dim=latent_dim,
                                 label_dim=label_dim,
                                 image_size=512),
                eval_kwargs=dict(
                    generator_smooth=dict(),
                ),
                interval=None,
                first_iter=None,
                save_best=True
                )
            )
