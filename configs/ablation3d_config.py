# python3.8
"""Configuration for ablating modules of 3D-aware GANs."""

import click
from .base_config import BaseConfig

__all__ = ['Ablation3DConfig']

RUNNER = 'EG3DRunner'
DATASET = 'EG3DDataset'
DISCRIMINATOR = 'EG3DDiscriminator'
GENERATOR = 'Ablation3DGenerator'
LOSS = 'EG3DLoss'

class Ablation3DConfig(BaseConfig):
    """Defines the configuration for ablating modules of 3D-aware GANs."""

    name = 'ablation3d'
    hint = 'Train a Ablation3D model.'
    info = '''
To train the model, the recommended settings are as follows:

\b
- batch_size: 4 (for FF-HQ dataset, 8 GPU)
- val_batch_size: 4 (for FF-HQ dataset, 8 GPU)
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
                '--pose_dim', type=cls.int_type, default=25,
                help='Dimension of conditoned pose information, default: 25.'),
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
                     'generator, which will be `factor * 32768`.'),
            cls.command_option(
                '--g_num_mappings', type=cls.int_type, default=8,
                help='Number of mapping layers of generator.'),
            cls.command_option(
                '--channel_base', type=cls.int_type, default=32768,
                help='Capacity multiplier.'),
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
                '--d_beta_2', type=cls.float_type, default=0.9,
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
                '--g_beta_2', type=cls.float_type, default=0.9,
                help='The Adam hyper-parameter `beta_2` for generator '
                     'optimizer.'),
            cls.command_option(
                '--adjust_lr_kimg', type=cls.int_type, default=0,
                help='Learning rate adjustment kimg.'),
            cls.command_option(
                '--adjust_lr_ratio', type=cls.float_type, default=1.0,
                help='Learning rate adjustment ratio.'),
            cls.command_option(
                '--map_depth', type=cls.int_type, default=2,
                help='Mapping network depth.'),
            cls.command_option(
                '--mlp_type',
                type=click.Choice(['eg3d', 'pigan', 'stylenerf']),
                default='eg3d',
                help='Type of MLP network.'),
            cls.command_option(
                '--mlp_depth', type=cls.int_type, default=2,
                help='MLP network depth.'),
            cls.command_option(
                '--mlp_hidden_dim', type=cls.int_type, default=64,
                help='Hidden dimension of the mlp network.'),
            cls.command_option(
                '--mlp_output_dim', type=cls.int_type, default=32,
                help='Output dimension of the mlp network.'),
            cls.command_option(
                '--mlp_lr_mul', type=cls.float_type, default=1.0,
                help='Learning rate multiplier of the mlp network.'),
            cls.command_option(
                '--w_moving_decay', type=cls.float_type, default=0.995,
                help='Decay factor for updating `w_avg`.'),
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
                '--use_ada', type=cls.bool_type, default=False,
                help='Whether to use adaptive augmentation pipeline.'),
            cls.command_option(
                '--ada_speed_img', type=cls.int_type, default=500_000,
                help='ADA adjustment speed, measured in how many img it takes '
                     'for `p` to increase/decrease by one unit.'),
            cls.command_option(
                '--g_ema_img', type=cls.int_type, default=10_000,
                help='Factor for updating the smoothed generator, which is '
                     'particularly used for inference.'),
            cls.command_option(
                '--g_ema_rampup', type=cls.float_type, default=0.05,
                help='Rampup factor for updating the smoothed generator, which '
                     'is particularly used for inference. Set as `0` to '
                     'disable warming up.'),
            cls.command_option(
                '--fv_feat_res', type=cls.int_type, default=32,
                help='Feature volume final resolution.'),
            cls.command_option(
                '--fv_init_res', type=cls.int_type, default=4,
                help='Feature volume initial resolution.'),
            cls.command_option(
                '--rendering_resolution_initial', type=cls.int_type,
                default=64,
                help='Resolution to render at.'),
            cls.command_option(
                '--rendering_resolution_final', type=cls.int_type,
                default=None,
                help='Final resolution to render at, if blending.'),
            cls.command_option(
                '--rendering_resolution_fade_kimg', type=cls.int_type,
                default=1000,
                help='Kimg to blend resolution over.'),
            cls.command_option(
                '--blur_fade_kimg', type=cls.int_type, default=200,
                help='Blur over how many.'),
            cls.command_option(
                '--gen_pose_cond', type=cls.bool_type, default=True,
                help='Whether enable generator pose conditioning.'),
            cls.command_option(
                '--label_scale', type=cls.float_type, default=1.0,
                help='Scale factor for generator pose conditioning.'),
            cls.command_option(
                '--gpc_reg_prob', type=cls.float_type, default=0.5,
                help='Strength of swapping regularization. '
                     'None means no generator pose conditioning, '
                     'i.e. condition with zeros.'),
            cls.command_option(
                '--gpc_reg_fade_kimg', type=cls.int_type, default=1000,
                help='Length of swapping prob fade.'),
            cls.command_option(
                '--sr_noise_mode',
                type=click.Choice(['random', 'const', 'none']),
                default='none',
                help='Strength of discriminator pose conditioning '
                     'regularization, in standard deviations.'),
            cls.command_option(
                '--resume_blur', type=cls.bool_type, default=False,
                help='Enable to blur even on resume.'),
            cls.command_option(
                '--blur_init_sigma', type=cls.float_type, default=10.0,
                help='Blur the images seen by the discriminator.'),
            cls.command_option(
                '--sr_num_fp16_res', type=cls.int_type, default=4,
                help='Number of fp16 layers in superresolution.'),
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
                '--sr_first_cutoff', type=cls.int_type, default=2,
                help='First cutoff for AF superresolution.'),
            cls.command_option(
                '--sr_first_stopband', type=cls.float_type, default=2**2.1,
                help='First stopband for AF superresolution.'),
            cls.command_option(
                '--dual_discrimination', type=cls.bool_type, default=True,
                help='Whether to use the dual discrimination or not.'),
            cls.command_option(
                '--use_upsampler', type=cls.bool_type, default=True,
                help='Whether to use upsampler or not.'),
        ])

        options['Rendering options'].extend([
            cls.command_option(
                '--density_reg', type=cls.float_type, default=0.25,
                help='Density regularization strength.'),
            cls.command_option(
                '--density_reg_interval', type=cls.int_type, default=4,
                help='Interval (in iterations) to perform density '
                     'regularization.'),
            cls.command_option(
                '--density_reg_p_dist', type=cls.float_type, default=0.004,
                help='Density regularization strength.'),
            cls.command_option(
                '--use_sdf', type=cls.bool_type, default=False,
                help='Whether to use SDF as geometry representation.'),
            cls.command_option(
                '--return_eikonal', type=cls.bool_type, default=False,
                help='Whether to use eikonal loss for regularization.'),
            cls.command_option(
                '--eikonal_lambda', type=cls.float_type, default=0.1,
                help='Factor to control the strength of eikonal loss.'),
            cls.command_option(
                '--min_surf_lambda', type=cls.float_type, default=0.05,
                help='Factor to control the strength of minimal surface loss.'),
            cls.command_option(
                '--min_surf_beta', type=cls.float_type, default=100.0,
                help='Beta value in minimal surface loss.'),
            cls.command_option(
                '--disc_c_noise', type=cls.float_type, default=0.0,
                help='Strength of discriminator pose conditioning '
                     'regularization, in standard deviations.'),
            cls.command_option(
                '--sr_antialias', type=cls.bool_type, default=True,
                help='Whether do antialising in SR module.'),
            cls.command_option(
                '--num_points', type=cls.int_type, default=48,
                help='Number of uniform samples to take per ray '
                     'in coarse pass.'),
            cls.command_option(
                '--num_importance', type=cls.int_type, default=48,
                help='Number of importance samples to take per ray '
                     'in fine pass.'),
            cls.command_option(
                '--ray_start', type=cls.float_type, default=2.25,
                help='Near point along each ray to start taking samples.'),
            cls.command_option(
                '--ray_end', type=cls.float_type, default=3.3,
                help='Far point along each ray to start taking samples.'),
            cls.command_option(
                '--focal', type=cls.float_type, default=4.2647,
                help='Normalized focal length of the camera.'),
            cls.command_option(
                '--coordinate_scale', type=cls.float_type, default=1.0,
                help='Scale factor to modulate coordinates before retrieving '
                     'features from triplanes.'),
            cls.command_option(
                '--white_back', type=cls.bool_type, default=False,
                help='Controls the color of rays that pass through the volume '
                     'without encountering any solid objects. Set to `True` if '
                     'your background is white and set to `False` if the '
                     'background is black.'),
            cls.command_option(
                '--avg_camera_radius', type=cls.float_type, default=2.7,
                help='The average radius of camera orbit, which is only used '
                     'in evaluation.'),
            cls.command_option(
                '--avg_camera_pivot_x', type=cls.float_type, default=0.0,
                help='X-coordinate of average pivot of camera rotation, which '
                     'is only used in evaluation.'),
            cls.command_option(
                '--avg_camera_pivot_y', type=cls.float_type, default=0.0,
                help='Y-coordinate of average pivot of camera rotation, which '
                     'is only used in evaluation.'),
            cls.command_option(
                '--avg_camera_pivot_z', type=cls.float_type, default=0.2,
                help='Z-coordinate of average pivot of camera rotation, which '
                     'is only used in evaluation.'),
            cls.command_option(
                '--use_raw_triplane_axes', type=cls.bool_type, default=False,
                help='Whether to use raw triplane axes as the official code.'),
            cls.command_option(
                '--ref_mode', type=click.Choice(
                    ['coordinate', 'volume', 'triplane', 'hybrid']),
                default='triplane',
                help='Mode of the reference representation.'),
            cls.command_option(
                '--use_positional_encoding', type=cls.bool_type,
                default=True, help='Whether to use positional encoding.'),
            cls.command_option(
                '--pe_num_freqs', type=cls.int_type, default=10,
                help='Number of frequency used for positional encoding.'),
            cls.command_option(
                '--pe_include_input', type=cls.bool_type, default=False,
                help='Whether to include raw coordinates in positional '
                     'encoding.'),
            cls.command_option(
                '--x_min', type=cls.float_type, default=-0.5,
                help='Minimum x value of the bounding box to normalize the '
                     'point coordinates.'),
            cls.command_option(
                '--x_max', type=cls.float_type, default=0.5,
                help='Maximum x value of the bounding box to normalize the '
                     'point coordinates.'),
            cls.command_option(
                '--y_min', type=cls.float_type, default=-0.5,
                help='Minimum y value of the bounding box to normalize the '
                     'point coordinates.'),
            cls.command_option(
                '--y_max', type=cls.float_type, default=0.5,
                help='Maximum y value of the bounding box to normalize the '
                     'point coordinates.'),
            cls.command_option(
                '--z_min', type=cls.float_type, default=-0.5,
                help='Minimum z value of the bounding box to normalize the '
                     'point coordinates.'),
            cls.command_option(
                '--z_max', type=cls.float_type, default=0.5,
                help='Maximum z value of the bounding box to normalize the '
                     'point coordinates.'),
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
            image_size=resolution,
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
        channel_base = self.args.pop('channel_base')
        channel_max = self.args.pop('channel_max')
        freezed = self.args.pop('freezed')
        d_mbstd_groups = self.args.pop('d_mbstd_groups')
        d_num_fp16_res = self.args.pop('d_num_fp16_res')
        g_num_fp16_res = self.args.pop('g_num_fp16_res')
        d_conv_clamp = 256 if d_num_fp16_res > 0 else None
        g_conv_clamp = 256 if g_num_fp16_res > 0 else None

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

        adjust_lr_kimg = self.args.pop('adjust_lr_kimg')
        adjust_lr_ratio = self.args.pop('adjust_lr_ratio')
        if adjust_lr_kimg > 0:
            self.config.adjust_lr_kimg = adjust_lr_kimg
            self.config.adjust_lr_ratio = adjust_lr_ratio

        gen_pose_cond = self.args.pop('gen_pose_cond')
        gpc_reg_prob = self.args.pop('gpc_reg_prob')
        label_scale = self.args.pop('label_scale')
        sr_noise_mode = self.args.pop('sr_noise_mode')
        density_reg = self.args.pop('density_reg')
        density_reg_interval = self.args.pop('density_reg_interval')
        density_reg_p_dist = self.args.pop('density_reg_p_dist')

        use_sdf = self.args.pop('use_sdf')
        self.config.use_sdf = use_sdf

        use_upsampler = self.args.pop('use_upsampler')
        dual_discrimination = self.args.pop('dual_discrimination')
        if not use_upsampler:
            assert dual_discrimination == False
            DISCRIMINATOR = 'EG3DSingleDiscriminator'
        else:
            DISCRIMINATOR = 'EG3DDiscriminator'

        self.args.pop('g_num_mappings')
        self.args.pop('w_moving_decay')
        self.args.pop('resume_blur')
        self.args.pop('sr_first_cutoff')
        self.args.pop('sr_first_stopband')
        rendering_resolution_initial = self.args.pop(
            'rendering_resolution_initial')
        rendering_resolution_final = self.args.pop(
            'rendering_resolution_final')
        rendering_resolution_fade_kimg = self.args.pop(
            'rendering_resolution_fade_kimg')

        image_boundary_value = 0.5 * (1 - 1 / rendering_resolution_initial)
        point_sampling_kwargs = dict(image_boundary_value=image_boundary_value,
                                     x_axis_right=True,
                                     y_axis_up=False,
                                     z_axis_out=False,
                                     dis_min=self.args.pop('ray_start'),
                                     dis_max=self.args.pop('ray_end'),
                                     focal=self.args.pop('focal'),
                                     num_points=self.args.pop('num_points'),
                                     perturbation_strategy='uniform')
        ray_marching_kwargs = dict(
            use_mid_point=True,
            density_clamp_mode='mipnerf',
            use_white_background=self.args.pop('white_back'),
            scale_color=True)

        avg_camera_pivot = [
            self.args.pop('avg_camera_pivot_x'),
            self.args.pop('avg_camera_pivot_y'),
            self.args.pop('avg_camera_pivot_z')]
        x_min = self.args.pop('x_min')
        x_max = self.args.pop('x_max')
        y_min = self.args.pop('y_min')
        y_max = self.args.pop('y_max')
        z_min = self.args.pop('z_min')
        z_max = self.args.pop('z_max')
        bound = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        use_positional_encoding = self.args.pop('use_positional_encoding')
        self.config.style_mixing_prob = self.args.pop('style_mixing_prob')
        self.config.models.update(
            discriminator=dict(model=dict(
                model_type=DISCRIMINATOR,
                c_dim=pose_dim,
                img_resolution=resolution,
                img_channels=image_channels,
                channel_base=channel_base,
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
                    mapping_layers=self.args.pop('map_depth'),
                    label_gen_conditioning_zero=(not gen_pose_cond),
                    label_scale=label_scale,
                    triplane_resolution=256,
                    triplane_channels=32 * 3,
                    fv_feat_res=self.args.pop('fv_feat_res'),
                    fv_init_res=self.args.pop('fv_init_res'),
                    num_fp16_res=g_num_fp16_res,
                    conv_clamp=g_conv_clamp,
                    coordinate_scale=self.args.pop('coordinate_scale'),
                    use_upsampler=use_upsampler,
                    sr_num_fp16_res=self.args.pop('sr_num_fp16_res'),
                    sr_channel_base=channel_base,
                    sr_channel_max=channel_max,
                    sr_antialias=self.args.pop('sr_antialias'),
                    sr_fused_modconv_default='inference_only',
                    sr_noise_mode=sr_noise_mode,
                    mlp_type=self.args.pop('mlp_type'),
                    mlp_depth=self.args.pop('mlp_depth'),
                    mlp_hidden_dim=self.args.pop('mlp_hidden_dim'),
                    mlp_output_dim=self.args.pop('mlp_output_dim'),
                    mlp_lr_mul=self.args.pop('mlp_lr_mul'),
                    pe_num_freqs=self.args.pop('pe_num_freqs'),
                    include_input=self.args.pop('pe_include_input'),
                    point_sampling_kwargs=point_sampling_kwargs,
                    ray_marching_kwargs=ray_marching_kwargs,
                    resolution=resolution,
                    rendering_resolution=rendering_resolution_initial,
                    num_importance=self.args.pop('num_importance'),
                    avg_camera_radius=self.args.pop('avg_camera_radius'),
                    avg_camera_pivot=avg_camera_pivot,
                    use_raw_triplane_axes=self.args.pop(
                        'use_raw_triplane_axes'),
                    ref_mode=self.args.pop('ref_mode'),
                    use_positional_encoding=use_positional_encoding,
                    bound=bound,
                    use_sdf=use_sdf,
                    return_eikonal=self.args.pop('return_eikonal')),
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
                dual_discrimination=dual_discrimination,
                filter_mode=self.args.pop('filter_mode'),
                blur_raw_target=self.args.pop('blur_raw_target')),
            g_loss_kwargs=dict(
                pl_batch_shrink=self.args.pop('pl_batch_shrink'),
                pl_weight=self.args.pop('pl_weight'),
                pl_decay=self.args.pop('pl_decay'),
                pl_interval=pl_interval,
                rendering_resolution_initial=rendering_resolution_initial,
                rendering_resolution_final=rendering_resolution_final,
                rendering_resolution_fade_kimg=rendering_resolution_fade_kimg,
                gpc_reg_fade_kimg=self.args.pop('gpc_reg_fade_kimg'),
                gpc_reg_prob=gpc_reg_prob,
                density_reg=density_reg,
                density_reg_interval=density_reg_interval,
                density_reg_p_dist=density_reg_p_dist,
                eikonal_lambda=self.args.pop('eikonal_lambda'),
                min_surf_lambda=self.args.pop('min_surf_lambda'),
                min_surf_beta=self.args.pop('min_surf_beta'))
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
        ada_speed_img = self.args.pop('ada_speed_img')
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
                    speed_img=ada_speed_img,
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
            ),
            FID50KFullEG3D=dict(
                init_kwargs=dict(name='fid50k_eg3d_full',
                                 latent_dim=latent_dim,
                                 label_dim=label_dim,
                                 image_size=resolution),
                eval_kwargs=dict(
                    generator_smooth=dict(),
                ),
                interval=None,
                first_iter=None,
                save_best=True
            )
        )