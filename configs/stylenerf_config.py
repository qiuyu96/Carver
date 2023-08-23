# python3.8
"""Configuration for training StyleNeRF."""

from .base_config import BaseConfig
import math
import click

__all__ = ['StyleNeRFConfig']

RUNNER = 'StyleNeRFRunner'
DATASET = 'ImageDataset'
DISCRIMINATOR = 'StyleNeRFDiscriminator'
GENERATOR = 'StyleNeRFGenerator'
LOSS = 'StyleNeRFLoss'
PI = math.pi


class StyleNeRFConfig(BaseConfig):
    """Defines the configuration for training StyleNeRF."""

    name = 'stylenerf'
    hint = 'Train a StyleNeRF model.'
    info = '''
To train a StyleNeRF model, the recommend settings are as follows:

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
            # Generator.
            cls.command_option(
                '--g_init_res',
                type=cls.int_type,
                default=32,
                help='The initial resolution to start convolution with in '
                     'generator.'),
            cls.command_option('--latent_dim',
                               type=cls.int_type,
                               default=512,
                               help='The dimension of the latent space Z.'),
            cls.command_option('--w_dim',
                               type=cls.int_type,
                               default=512,
                               help='The dimension of the latent space W.'),
            cls.command_option('--label_dim',
                type=cls.int_type,
                default=0,
                help='Number of classes in conditioning training. Set to `0` '
                     'to disable conditional training.'),
            cls.command_option('--g_num_mappings',
                               type=cls.int_type,
                               default=8,
                               help='Number of mapping layers of generator.'),
            cls.command_option('--impl',
                               type=str,
                               default='cuda',
                               help='Control the implementation of some neural '
                                    'operations.'),
            cls.command_option(
                '--num_fp16_res',
                type=cls.int_type,
                default=4,
                help='Number of (highest) resolutions that use `float16` '
                'precision for training, which speeds up the training yet '
                'barely affects the performance. The official '
                'StyleGAN-ADA uses 4 by default.'),
            cls.command_option('--color_out_dim',
                               type=cls.int_type,
                               default=256,
                               help='Number of channels of feature maps output '
                                    'by the fg_mlp'),
            cls.command_option('--mlp_z_dim',
                               type=cls.int_type,
                               default=0,
                               help='Dimensions of input `z` fed to `fg_mlp` '
                                    'instead of `w`.'),
            cls.command_option('--mlp_z_dim_bg',
                               type=cls.int_type,
                               default=32,
                               help='Dimensions of input `z` fed to `bg_mlp` '
                                    'instead of `w`.'),
            cls.command_option('--mlp_hidden_size_bg',
                               type=cls.int_type,
                               default=64,
                               help='Dimensions of the hidden layer in '
                                    '`bg_mlp`.'),
            cls.command_option('--mlp_n_blocks_bg',
                               type=cls.int_type,
                               default=4,
                               help='Number of blocks in `bg_mlp`.'),
            cls.command_option('--predict_color',
                              type=cls.bool_type,
                              default=True,
                              help='Whether to predict color value along with '
                                   'color feature maps when forwarding the mlp.'),
            cls.command_option('--g_activation',
                              type=str,
                              default='lrelu',
                              help='Activation mode of generator.'),
            cls.command_option('--g_channel_base',
                               type=cls.float_type,
                               default=1,
                               help='Overall multiplier for the number of '
                                    'channels in generator.'),
            cls.command_option('--g_channel_max',
                               type=cls.int_type,
                               default=1024,
                               help='Maximum number of channels in generator.'),
            cls.command_option('--g_conv_clamp',
                               type=cls.int_type,
                               default=256,
                               help='Clamp the output to `[-X, +X]`, `None` '
                                    'means disable clamping.'),
            cls.command_option('--kernel_size',
                               type=cls.int_type,
                               default=1,
                               help='Kernel size for convolutions of '
                                    'upsamplers.'),
            cls.command_option('--g_architecture',
                               type=str,
                               default='skip',
                               help='Architecture type of generator.'),
            cls.command_option('--g_upsample_mode',
                               type=str,
                               default='nn_cat',
                               help='Upsampling mode of upsampler.'),
            cls.command_option('--use_noise',
                               type=cls.bool_type,
                               default=False,
                               help='Whether to use noise injection in the '
                                    'upsampler.'),
            # Discriminator.
            cls.command_option('--d_architecture',
                               type=str,
                               default='skip',
                               help='Architecture type of discriminator.'),
            cls.command_option('--d_channel_base',
                               type=cls.float_type,
                               default=1,
                               help='Overall multiplier for the number of '
                                    'channels in discriminator.'),
            cls.command_option('--d_channel_max',
                               type=cls.int_type,
                               default=512,
                               help='Maximum number of channels in '
                                    'discriminator.'),
            cls.command_option('--d_conv_clamp',
                               type=cls.int_type,
                               default=256,
                               help='Clamp the output to `[-X, +X]`, `None` '
                                    'means disable clamping.'),
            cls.command_option('--d_upsample_mode',
                               type=str,
                               default='bilinear',
                               help='Upsampling mode of upsampler.'),
            cls.command_option('--resize_real_early',
                               type=cls.bool_type,
                               default=True,
                               help='Whether to peform resizing before the '
                                    'training loop.'),
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
                '--w_moving_decay',
                type=cls.float_type,
                default=0.999,
                help='Decay factor for updating `w_avg`.'),
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
                '--nerf_path_reg_weight',
                type=cls.float_type,
                default=1.0,
                help='Weight for NeRF path regularization loss.'),
            cls.command_option(
                '--g_ema_img',
                type=cls.int_type,
                default=20_000,
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
                default=True,
                help='Whether to use progressive training.'),
            cls.command_option(
                '--pg_iter_start',
                type=cls.int_type,
                default=500,
                help='Progressive training start iterations (kimg).'),
            cls.command_option(
                '--pg_iter_end',
                type=cls.int_type,
                default=5000,
                help='Progressive training end iterations (kimg).'),
        ])

        options['Rendering options'].extend([
            cls.command_option(
                '--rendering_resolution',
                type=cls.int_type,
                default=32,
                help='Resolution of volume rendering images.'),
            cls.command_option(
                '--clamp_mode',
                type=click.Choice(['softplus', 'relu', 'mipnerf']),
                default='relu',
                help='clamp mode of `sigmas` in intergration process.'),
            cls.command_option(
                '--num_points',
                type=cls.int_type,
                default=14,
                help='Number of uniform samples to take per ray '
                     'in coarse pass.'),
            cls.command_option(
                '--num_importance',
                type=cls.int_type,
                default=14,
                help='Number of importance samples to take per ray '
                    'in fine pass.'),
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
            cls.command_option(
                '--use_background',
                type=cls.bool_type,
                default=True,
                help='Whether to use background NeRF modeling.'),
            cls.command_option(
                '--bg_ray_start',
                type=cls.float_type,
                default=0.5,
                help='Near point along each ray of background to start taking '
                     'samples.'),
            cls.command_option(
                '--bg_num_points',
                type=cls.int_type,
                default=4,
                help='Number of uniform samples to take per ray '
                     'in pass of background.'),
            cls.command_option(
                '--nerf_path_regularization',
                type=cls.bool_type,
                default=True,
                help='Whether to use NeRF path regularization.'),
            cls.command_option(
                '--reg_res',
                type=cls.int_type,
                default=16,
                help='Resolution of pixels for NeRF path regularization.')
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

        # `g_init_res`: Initial resolution of upsampler.
        g_init_res = self.args.pop('g_init_res')
        latent_dim = self.args.pop('latent_dim')
        w_dim = self.args.pop('w_dim')
        label_dim = self.args.pop('label_dim')
        impl = self.args.pop('impl')
        num_fp16_res = self.args.pop('num_fp16_res')

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
        assert rendering_resolution == g_init_res

        radius = self.args.pop('radius_fix')
        point_sampling_kwargs = dict(
            image_boundary_value=1.0,
            x_axis_right=True,
            y_axis_up=True,
            z_axis_out=True,
            radius_strategy='fix',
            radius_fix=radius,
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
            scale_color=False,
            density_clamp_mode=self.args.pop('clamp_mode'),
            normalize_radial_dist=False,
            clip_radial_dist=False)

        use_pg = self.args.pop('use_pg')
        self.config.pg_iter_start =  self.args.pop('pg_iter_start')  # kimg
        self.config.pg_iter_end =self.args.pop('pg_iter_end')  # kimg

        self.config.models.update(
            discriminator=dict(
                model=dict(
                    model_type=DISCRIMINATOR,
                    c_dim=label_dim,
                    img_channels=image_channels,
                    img_resolution=resolution,
                    channel_base=self.args.pop('d_channel_base'),
                    channel_max=self.args.pop('d_channel_max'),
                    conv_clamp=self.args.pop('d_conv_clamp'),
                    progressive=use_pg,
                    architecture=self.args.pop('d_architecture'),
                    num_fp16_res=num_fp16_res,
                    lowres_head=g_init_res,
                    upsample_type=self.args.pop('d_upsample_mode'),
                    resize_real_early=self.args.pop('resize_real_early')),
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
                    w_dim=w_dim,
                    label_dim=label_dim,
                    mapping_layers=self.args.pop('g_num_mappings'),
                    color_out_dim=self.args.pop('color_out_dim'),
                    color_out_dim_bg=None,
                    mlp_z_dim=self.args.pop('mlp_z_dim'),
                    mlp_z_dim_bg=self.args.pop('mlp_z_dim_bg'),
                    mlp_hidden_size_bg=self.args.pop('mlp_hidden_size_bg'),
                    mlp_n_blocks_bg=self.args.pop('mlp_n_blocks_bg'),
                    predict_color=self.args.pop('predict_color'),
                    progressive_training=use_pg,
                    activation=self.args.pop('g_activation'),
                    channel_base=self.args.pop('g_channel_base'),
                    channel_max=self.args.pop('g_channel_max'),
                    img_channels=image_channels,
                    num_fp16_res=num_fp16_res,
                    conv_clamp=self.args.pop('g_conv_clamp'),
                    kernel_size=self.args.pop('kernel_size'),
                    architecture=self.args.pop('g_architecture'),
                    upsample_mode=self.args.pop('g_upsample_mode'),
                    use_noise=self.args.pop('use_noise'),
                    magnitude_ema_beta=self.args.pop('w_moving_decay'),
                    nerf_res=rendering_resolution,
                    resolution=resolution,
                    num_importance=self.args.pop('num_importance'),
                    use_background=self.args.pop('use_background'),
                    bg_ray_start=self.args.pop('bg_ray_start'),
                    bg_num_points=self.args.pop('bg_num_points'),
                    nerf_path_reg=self.args.pop('nerf_path_regularization'),
                    reg_res=self.args.pop('reg_res'),
                    point_sampling_kwargs=point_sampling_kwargs,
                    ray_marching_kwargs=ray_marching_kwargs),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=g_lr,
                         betas=(g_beta_1, g_beta_2)),
                kwargs_train=dict(
                    style_mixing_prob=self.args.pop('style_mixing_prob')),
                kwargs_val=dict(),
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
                pl_interval=pl_interval,
                nerf_path_reg_weight=self.args.pop('nerf_path_reg_weight')))

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
                                             label_dim=label_dim),
                            eval_kwargs=dict(generator_smooth=dict(),),
                            interval=None,
                            first_iter=None,
                            save_best=True),
            GANSnapshot=dict(init_kwargs=dict(name='snapshot',
                                              latent_dim=latent_dim,
                                              latent_num=32,
                                              label_dim=label_dim,
                                              min_val=min_val,
                                              max_val=max_val),
                             eval_kwargs=dict(generator_smooth=dict(),),
                             interval=None,
                             first_iter=None,
                             save_best=False),
            GANSnapshotMultiView=dict(
                init_kwargs=dict(name='snapshot_multi_view',
                                 latent_dim=latent_dim,
                                 latent_num=8,
                                 label_dim=label_dim,
                                 min_val=min_val,
                                 max_val=max_val,
                                 radius=radius,
                                 azimuthal_start=math.pi/2-0.6,
                                 azimuthal_end=math.pi/2+0.6),
                eval_kwargs=dict(generator_smooth=dict(),),
                interval=None,
                first_iter=None,
                save_best=False))
