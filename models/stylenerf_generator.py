# python3.8
"""Contains the implementation of generator described in StyleNeRF.

Paper: https://arxiv.org/pdf/2110.08985.pdf

Official PyTorch implementation: https://github.com/facebookresearch/StyleNeRF
"""

import math
import numpy as np
from einops import repeat
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.stylenerf_model_helper import MappingNetwork
from models.utils.stylenerf_model_helper import FullyConnectedLayer
from models.utils.stylenerf_model_helper import SynthesisBlock
from models.utils.stylenerf_model_helper import ToRGBLayer
from models.utils.stylenerf_model_helper import Conv2dLayer
from models.utils.stylenerf_model_helper import modulated_conv2d

from models.rendering import PointSampler
from models.rendering import PointRepresenter
from models.rendering import PointIntegrator
from models.rendering.utils import PositionEncoder

from utils import eg3d_misc as misc
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan2_official_ops import bias_act

from models.rendering.utils import sample_importance
from models.rendering.utils import unify_attributes
from models.rendering.utils import depth2pts_outside
from models.rendering.point_sampler import perturb_points_per_ray


class StyleNeRFGenerator(nn.Module):
    """Defines the generator network in StyleNeRF."""

    def __init__(
            self,
            # Settings for the mapping network.
            z_dim=512,
            w_dim=512,
            label_dim=0,
            image_channels=3,
            mapping_layers=8,
            # Settiings for the mlp.
            color_out_dim=256,
            color_out_dim_bg=None,
            mlp_z_dim=0,
            mlp_z_dim_bg=32,
            mlp_hidden_size_bg=64,
            mlp_n_blocks_bg=4,
            activation='lrelu',
            predict_color=True,
            progressive_training=True,
            # Settings for progressive training.
            pe_input_dim=3,
            pe_num_freqs=10,
            pe_factor=np.pi,
            include_input=False,
            # Settings for the upsampler.
            channel_base=1,
            channel_max=1024,
            img_channels=3,
            num_fp16_res=4,
            conv_clamp=256,
            kernel_size=1,
            architecture='skip',
            upsample_mode='nn_cat',
            use_noise=False,
            magnitude_ema_beta=-1,
            # Setitings for rendering.
            nerf_res=32,
            resolution=256,
            num_importance=12,
            use_background=True,
            bg_ray_start=0.5,
            bg_num_points=4,
            nerf_path_reg=True,
            reg_res=16,
            point_sampling_kwargs=None,
            ray_marching_kwargs=None):

        super().__init__()

        self.z_dim = (z_dim,)
        self.latent_dim = z_dim
        self.label_dim = label_dim
        self.image_channels = image_channels
        self.num_wp = 0

        # Set up the rendering related module.
        if point_sampling_kwargs is None:
            point_sampling_kwargs = {}
        if ray_marching_kwargs is None:
            ray_marching_kwargs = {}
        self.point_sampler = PointSampler(**point_sampling_kwargs)
        self.point_integrator = PointIntegrator(**ray_marching_kwargs)

        self.point_representer = PointRepresenter(
            representation_type='coordinate')

        # Set up the position encoder.
        self.position_encoder = PositionEncoder(input_dim=pe_input_dim,
                                                max_freq_log2=pe_num_freqs - 1,
                                                num_freqs=pe_num_freqs,
                                                factor=pe_factor,
                                                include_input=include_input)


        # Set up the mlp (foreground and background).
        self.fg_mlp = MLPNetwork(input_dim=self.position_encoder.out_dim,
                                 z_dim=mlp_z_dim,
                                 w_dim=w_dim,
                                 color_out_dim=color_out_dim,
                                 predict_color=predict_color,
                                 activation=activation)

        self.num_wp += self.fg_mlp.num_wp
        self.bg_mlp = None
        if use_background:
            bg_position_encoder = PositionEncoder(input_dim=4,
                                                  max_freq_log2=pe_num_freqs -
                                                  1,
                                                  num_freqs=pe_num_freqs,
                                                  factor=pe_factor,
                                                  include_input=include_input)
            self.bg_mlp = MLPNetwork(
                input_dim=bg_position_encoder.out_dim,
                z_dim=mlp_z_dim_bg,
                w_dim=w_dim,
                hidden_size=mlp_hidden_size_bg,
                n_blocks=mlp_n_blocks_bg,
                color_out_dim=color_out_dim_bg
                if color_out_dim_bg is not None else color_out_dim,
                predict_color=predict_color,
                activation=activation)
            self.num_wp += self.bg_mlp.num_wp

        # Set up the post cnn.
        self.nerf_res = nerf_res
        self.post_cnn = PostNeuralRendererNetwork(
            w_dim=w_dim,
            input_dim=color_out_dim,
            img_channels=img_channels,
            in_res=self.nerf_res,
            out_res=resolution,
            channel_base=channel_base,
            channel_max=channel_max,
            num_fp16_res=num_fp16_res,
            conv_clamp=conv_clamp,
            kernel_size=kernel_size,
            architecture=architecture,
            upsample_mode=upsample_mode,
            use_noise=use_noise,
            magnitude_ema_beta=magnitude_ema_beta,
            progressive_training=progressive_training)

        upsamplers = self.post_cnn.networks
        if len(upsamplers) > 0:
            self.block_names = [u['name'] for u in upsamplers]
            self.num_wp += sum([u['num_wp'] for u in upsamplers])
            for u in upsamplers:
                setattr(self, u['name'], u['block'])

        # Set up the mapping network.
        self.mapping = MappingNetwork(z_dim=z_dim,
                                      c_dim=label_dim,
                                      w_dim=w_dim,
                                      num_ws=self.num_wp,
                                      num_layers=mapping_layers)

        # Set up `alpha` to control progressive training.
        # alpha = -1, temporarily not progressive training.
        # alpha ~ [0, 1], degrees of progressive resolution.
        self.register_buffer('alpha', torch.scalar_tensor(-1))

        # Other settings.
        self.progressive_training = progressive_training
        self.block_resolutions = self.post_cnn.block_resolutions
        self.in_res = self.post_cnn.in_res
        self.num_importance = num_importance
        self.use_background = use_background
        self.bg_ray_start = bg_ray_start
        self.bg_num_points = bg_num_points
        self.nerf_path_reg = nerf_path_reg
        self.reg_res = reg_res
        self.resolution = resolution

    @staticmethod
    def upsample(image, size, filter=None):
        up = size // image.size(-1)
        if up <= 1:
            return image

        if filter is not None:
            for _ in range(int(math.log2(up))):
                image = upfirdn2d.downsample2d(image, filter, up=2)
        else:
            image = F.interpolate(image, (size, size),
                                  mode='bilinear',
                                  align_corners=False)
        return image

    def forward(self,
                z,
                label=None,
                style_mixing_prob=None,
                cam2world_matrix=None):
        N = z.shape[0]

        n_levels, end_l, target_img_res = self.get_current_resolution()

        if (end_l == 0) or len(self.block_resolutions) == 0:
            cur_resolution = self.nerf_res
        else:
            cur_resolution = self.block_resolutions[end_l - 1]

        wp = self.mapping(z, label)

        results = {'wp': wp}

        if self.training and style_mixing_prob is not None:
            if np.random.uniform() < style_mixing_prob:
                new_z = torch.randn_like(z)
                new_wp = self.mapping(new_z, label, skip_w_avg_update=True)
                mixing_cutoff = np.random.randint(1, self.num_wp)
                wp[:, mixing_cutoff:] = new_wp[:, mixing_cutoff:]

        fg_mlp_z_dim = self.fg_mlp.z_dim if self.fg_mlp is not None else 0
        bg_mlp_z_dim = self.bg_mlp.z_dim if self.bg_mlp is not None else 0
        fg_z_shape = torch.randn([wp.shape[0], fg_mlp_z_dim]).to(
            wp.device) if fg_mlp_z_dim > 0 else None
        bg_z_shape = torch.randn([wp.shape[0], bg_mlp_z_dim]).to(
            wp.device) if bg_mlp_z_dim > 0 else None

        fg_wp = None
        bg_wp = None
        if self.fg_mlp.num_wp > 0:
            fg_wp = wp[:, :self.fg_mlp.num_wp]
            wp = wp[:, self.fg_mlp.num_wp:]
        if self.bg_mlp is not None and self.bg_mlp.num_wp > 0:
            bg_wp = wp[:, :self.bg_mlp.num_wp]
            wp = wp[:, self.bg_mlp.num_wp:]

        nerf_res = self.nerf_res
        reg_res = self.reg_res
        nerf_path_reg = (self.nerf_path_reg and (target_img_res > nerf_res)
                         and self.training)
        height = nerf_res
        width = nerf_res
        if nerf_path_reg:
            total_num_rays = (nerf_res**2 + reg_res**2)
            height, width = dividable(total_num_rays)

        point_sampling_result = self.point_sampler(
            batch_size=N,
            image_size=self.nerf_res,
            cam2world_matrix=cam2world_matrix)

        points = point_sampling_result['points_world']  # [N, H, W, K, 3]
        ray_dirs = point_sampling_result['rays_world']  # [N, H, W, 3]
        radii_coarse = point_sampling_result['radii']  # [N, H, W, K]
        ray_origins = point_sampling_result[
            'cam2world_matrix'][:, :3, -1]  # [N, 3]

        _, H, W, K, _ = points.shape
        R = H * W
        points = points.reshape(N, R, K, 3)
        ray_dirs = ray_dirs.reshape(N, R, 3)
        ray_origins = ray_origins.unsqueeze(1).repeat(1, R, 1)  # [N, R, 3]
        radii_coarse = radii_coarse.reshape(N, R, K, 1)

        ## Prepare points for NeRF path regularization.
        rand_indices = None
        if target_img_res is None:
            target_img_res = nerf_res
        if nerf_path_reg:
            # Prepare random indices for querying pixels for NeRF path
            # regularization.
            pace = nerf_res // reg_res
            idxs = torch.arange(0, nerf_res, pace, device=z.device)
            u_xy = torch.rand(N, 2, device=z.device)
            u_xy = (u_xy * pace).floor().long()
            x_idxs, y_idxs = idxs[None, :] + u_xy[:, :1], idxs[
                None, :] + u_xy[:, 1:]
            rand_indices = (x_idxs[:, None, :] + y_idxs[:, :, None] * nerf_res)

            reg_point_sampling_res = self.point_sampler(
                batch_size=N,
                image_size=self.nerf_res,
                selected_pixels=rand_indices)

            reg_points = reg_point_sampling_res['points_world']
            reg_ray_dirs = reg_point_sampling_res['rays_world']
            reg_ray_origins = reg_point_sampling_res[
                'cam2world_matrix'][:, :3, -1]  # [N, 3]
            reg_radii_coarse = reg_point_sampling_res['radii']
            rand_indices = rand_indices.reshape(N, -1)  # [N, R_]

            N_, H_, W_, K_, _ = reg_points.shape
            assert N_ == N
            assert K_ == K
            assert H_ == W_ == reg_res
            R_ = H_ * W_

            reg_points = reg_points.reshape(N, R_, K, -1)
            reg_ray_dirs = reg_ray_dirs.reshape(N, R_, -1)
            reg_ray_origins = reg_ray_origins.unsqueeze(1).repeat(
                1, R_, 1)  # [N, R, 3]
            reg_radii_coarse = reg_radii_coarse.reshape(N, R_, K, -1)

            points = torch.cat([points, reg_points], dim=1)  # [N, R + R_, K, 3]
            ray_dirs = torch.cat([ray_dirs, reg_ray_dirs], dim=1)
            ray_origins = torch.cat([ray_origins, reg_ray_origins], dim=1)
            radii_coarse = torch.cat([radii_coarse, reg_radii_coarse], dim=1)

            R = R + R_  # Sum number of original rays and rand pixel rays.

        points_encoding = self.position_encoder(points)
        color_density_result = self.fg_mlp(points_encoding=points_encoding,
                                           wp=fg_wp,
                                           z_shape=fg_z_shape,
                                           height=height,
                                           width=width)

        densities_coarse = color_density_result['density']  # [N, R * K, 1]
        colors_coarse = color_density_result['color']      # [N, R * K, C1]
        densities_coarse = densities_coarse.reshape(N, R, K,
                                                    densities_coarse.shape[-1])
        colors_coarse = colors_coarse.reshape(N, R, K, colors_coarse.shape[-1])

        fg_max_radial_dist = 0.0 if self.use_background else 1e10

        # Do the integration.
        if self.num_importance > 0:
            # Do the integration in coarse pass.
            rendering_result = self.point_integrator(
                colors_coarse,
                densities_coarse,
                radii_coarse,
                max_radial_dist=fg_max_radial_dist)
            weights = rendering_result['weight']

            # Importrance sampling.
            radii_fine = sample_importance(radii_coarse,
                                           weights,
                                           self.num_importance,
                                           smooth_weights=True)
            points = ray_origins.unsqueeze(
                -2) + radii_fine * ray_dirs.unsqueeze(-2)

            # Get density's and color's value (or feature).
            points_encoding = self.position_encoder(points)
            color_density_result = self.fg_mlp(points_encoding=points_encoding,
                                               wp=fg_wp,
                                               z_shape=fg_z_shape,
                                               height=height,
                                               width=width)

            densities_fine = color_density_result['density']
            colors_fine = color_density_result['color']
            densities_fine = densities_fine.reshape(N, R, self.num_importance,
                                                    densities_fine.shape[-1])
            colors_fine = colors_fine.reshape(N, R, self.num_importance,
                                              colors_fine.shape[-1])

            # Gather coarse and fine results.
            all_radiis, all_colors, all_densities = unify_attributes(
                radii_coarse, colors_coarse, densities_coarse,
                radii_fine, colors_fine, densities_fine)

            # Do the integration in fine pass.
            rendering_result = self.point_integrator(
                all_colors,
                all_densities,
                all_radiis,
                max_radial_dist=fg_max_radial_dist)

        else:
            rendering_result = self.point_integrator(
                colors_coarse,
                densities_coarse,
                radii_coarse,
                max_radial_dist=fg_max_radial_dist)

        # Background rendering (NeRF++).
        if self.use_background:
            bg_ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
            K_BG = self.bg_num_points
            bg_z = torch.linspace(-1., 0., steps=K_BG, device=z.device)
            bg_z = bg_z.reshape(1, 1, 1, K_BG).repeat(N, height, width, 1)
            bg_z = bg_z * self.bg_ray_start
            bg_z = perturb_points_per_ray(
                bg_z, strategy='middle_uniform')  # [N, H, W, K_BG]
            bg_z = bg_z.reshape(N, -1, bg_z.shape[-1])
            bg_points, _ = depth2pts_outside(
                ray_origins.unsqueeze(-2).repeat(1, 1, K_BG, 1),
                bg_ray_dirs.unsqueeze(-2).repeat(1, 1, K_BG, 1),
                -bg_z)  # [N, R, K_BG, 4]

            bg_points_encoding = self.position_encoder(bg_points)
            bg_color_density_result = self.bg_mlp(
                points_encoding=bg_points_encoding,
                wp=bg_wp,
                z_shape=bg_z_shape,
                height=height,
                width=width)

            bg_densities = bg_color_density_result['density']
            bg_colors = bg_color_density_result['color']
            bg_densities = bg_densities.reshape(N, R, K_BG, -1)
            bg_colors = bg_colors.reshape(N, R, K_BG, -1)
            bg_z = bg_z.reshape(N, R, K_BG, -1)

            bg_rendering_result = self.point_integrator(
                bg_colors, bg_densities, bg_z)
            bg_lambda = rendering_result['T_end']  # [N, R, 1]
            bg_color = bg_rendering_result['composite_color'] * bg_lambda

            # Sum foreground and background to get final color/feature.
            rendering_result['composite_color'] = (
                rendering_result['composite_color'] + bg_color
            )  # [N, R, C]

            # Get weights from foreground and background.
            fg_weights = rendering_result['weight']    # [N, R, K, 1]
            bg_weights = bg_rendering_result['weight']       # [N, R, K_BG, 1]
            bg_weights = bg_weights * bg_lambda.unsqueeze(-2)# [N, R, K_BG, 1]
            rendering_result['weight'] = torch.cat(
                (fg_weights, bg_weights), dim=2)             # [N, R, K+K_BG, 1]

        if nerf_path_reg:
            # Retain `point_feats` for NeRF path regularization.
            point_feats = torch.cat([all_colors, bg_colors],
                                    dim=2)  # [N, R, K+K_BG, C2]
            rendering_result['point_feat'] = point_feats
            rendering_result['rand_indices'] = rand_indices

        num_rays = nerf_res * nerf_res

        imgs = []
        feature2d = rendering_result['composite_color'][:, :num_rays]
        feature2d = rearrange(feature2d, 'N (H W) C -> N C H W', H=nerf_res)
        img = feature2d[:, :3]
        feat = feature2d[:, 3:]

        # Use 2D upsampler.
        img_approx = None
        img_rand = None
        if cur_resolution > self.nerf_res:
            imgs += [img]
            wp = wp.to(torch.float32)
            blocks   = [getattr(self, name) for name in self.block_names]
            block_wp = self.post_cnn.split_wp(wp, blocks)
            post_render_imgs = self.post_cnn(blocks,
                                             block_wp,
                                             feat,
                                             img,
                                             target_img_res,
                                             skip_up=False)
            imgs += post_render_imgs

            # NeRF path regularization related.
            rand_imgs = []
            if nerf_path_reg:
                rand_weights = rendering_result[
                    'weight'][:, num_rays:].squeeze(
                        -1)  # [N, num_rand_rays, K + K_BG]
                rand_point_feats = rendering_result[
                    'point_feat'][:,
                                  num_rays:]  # [N, num_rand_rays, K + K_BG, C]
                rand_weights = rearrange(rand_weights,
                                         'N (H W) K -> N K H W',
                                         H=reg_res,
                                         W=reg_res)
                num_points = rand_weights.shape[1]
                sH, sW = dividable(num_points)
                rand_point_feats = rearrange(
                    rand_point_feats,
                    'N (H W) (s0 s1) C -> N C (s0 H) (s1 W)',
                    H=reg_res,
                    W=reg_res,
                    s0=sH,
                    s1=sW)
                img_rand = rand_point_feats[:, :3]
                rand_feat = rand_point_feats[:, 3:]
                rand_imgs += [img_rand]

                # Forward NeRF based `img_rand` into `post_cnn`
                # without upsampling. Here `post_cnn` consists of
                # 1x1 convolutions, thus equaling to a MLP. The following
                # operation can be regarded as an extension to the previous
                # NeRF process.
                post_mlp_rand_imgs = self.post_cnn(blocks,
                                                   block_wp,
                                                   rand_feat,
                                                   img_rand,
                                                   target_img_res,
                                                   skip_up=True)
                rand_imgs += post_mlp_rand_imgs
                img_rand = rand_imgs[-1]

            img = imgs[-1]
            if (self.alpha < 1) and (self.alpha > 0):
                alpha, _ = math.modf(self.alpha * n_levels)
                img_tmp = imgs[-2]
                if img_tmp.shape[-1] < img.shape[-1]:
                    img_tmp = self.upsample(img_tmp, 2 * img_tmp.shape[-1])
                img = img_tmp * (1 - alpha) + img * alpha
                if len(rand_imgs) > 0:
                    img_rand = rand_imgs[-2] * (1 - alpha) + img_rand * alpha

            assert nerf_path_reg == (len(rand_imgs) > 0)
            if nerf_path_reg:
                img_rand = rearrange(img_rand,
                                     'N C (s0 H) (s1 W) -> N C (s0 s1) H W',
                                     s0=sH,
                                     s1=sW)
                # Get `I_nerf` described in StyleNeRF paper.
                img_rand = (img_rand * rand_weights[:, None]).sum(
                    2)  # [N, C, H, W]

                rand_indices = rendering_result[
                    'rand_indices']  # [N, reg_res**2]
                rand_indices = rand_indices.unsqueeze(1).repeat(
                    1, img_rand.shape[1], 1)  # [N, C, reg_res**2]

                # Get `I_approx` described in StyleNeRF paper.
                img_approx = rearrange(img, 'N C H W -> N C (H W)').gather(
                    2, rand_indices)  # [N, C, reg_res**2]
                img_approx = rearrange(
                    img_approx,
                    'N C (H W) -> N C H W',
                    H=reg_res,
                    W=reg_res)  # [N, C, reg_res, reg_res]

        results['image'] = img.contiguous()
        results['label'] = torch.Tensor(wp.shape[0],
                                        self.label_dim).to(wp.device)
        results['image_approx'] = img_approx
        results['image_rand'] = img_rand

        return results

    def get_current_resolution(self):
        """Get current output image resolution according to paramater `alpha`
        for progressive training.

        Returns:
            n_levels (int): Total number of blocks in the post 2D neural
                renderer (upsampler).
            end_l (int): Target level of output image resolutions.
            target_res (int): Resolution of target image output by upsampler.

        For example, there are 5 blocks in the upsampler comprising the
        following resolutions:
            [32, 64, 128, 256, 512].
        Therefore we get `n_levels = 5`. Besides, at some stage of the training
        process, if the target output image resolution is 256, then we also get
        `target_res = 256` and `end_l = 4`.
        """

        n_levels = len(self.block_resolutions)
        if not self.progressive_training:
            end_l = n_levels
        elif self.alpha > -1:
            if self.alpha == 0:
                end_l = 0
            elif self.alpha == 1:
                end_l = n_levels
            elif self.alpha < 1:
                end_l = int(math.modf(self.alpha * n_levels)[1] + 1)
        else:
            end_l = n_levels

        target_res = (self.in_res
                      if end_l < 1 else self.block_resolutions[end_l - 1])

        return n_levels, end_l, target_res


class MLPNetwork(nn.Module):
    """Defines class of FOREGROUND/BACKGROUND MLP Network in StyleNeRF.

    Basically, this module consists of several `Style2Layer`s where convolutions
    with 1x1 kernel are involved. Note that this module is not strictly
    equivalent to MLP. Since 1x1 convolution is equal to fully-connected layer,
    we name this module `MLPNetwork`. Besides, our `MLPNetwork` takes in
    sampled points, view directions, latent codes as input, and outputs features
    for the following computation of `density` and `color` / `feature`.
    """

    def __init__(
        self,
        # Dimension settings.
        input_dim=3,
        w_dim=512,    # Style latent code.
        # Here `z_dim` controls the direct input to the mlp.
        # If z_dim > 0, the input will be
        # `torch.cat([points_encoding, z], -1)` and the mlp has
        # no `wp` injection. In StyleNeRF, `fg_mlp` is with
        # `z_dim=0` while `bg_mlp` is with `z_dim=32`.
        z_dim=0,
        color_out_dim=256,
        img_channels=3,
        hidden_size=128,
        n_blocks=8,
        magnitude_ema_beta=-1,
        # Architecture settings.
        predict_color=True,
        activation='lrelu',
        use_skip=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.w_dim = w_dim
        self.z_dim = z_dim
        self.activation = activation
        self.n_blocks = n_blocks
        self.magnitude_ema_beta = magnitude_ema_beta
        self.predict_color = predict_color
        self.use_skip = use_skip

        if z_dim > 0:
            w_dim = 0
            self.input_dim += z_dim
        else:
            w_dim = self.w_dim

        self.fc_in = Style2Layer(self.input_dim,
                                 self.hidden_size,
                                 w_dim,
                                 activation=self.activation)
        self.num_wp = 1
        self.skip_layer = self.n_blocks // 2 - 1 if self.use_skip else None
        if self.n_blocks > 1:
            self.blocks = nn.ModuleList([
                Style2Layer(self.hidden_size if i != self.skip_layer else
                            self.hidden_size + self.input_dim,
                            self.hidden_size,
                            w_dim,
                            activation=self.activation,
                            magnitude_ema_beta=self.magnitude_ema_beta)
                for i in range(self.n_blocks - 1)
            ])
            self.num_wp += (self.n_blocks - 1)

        self.density_out = ToRGBLayer(self.hidden_size,
                                     1,
                                     w_dim,
                                     kernel_size=1)
        self.num_wp += 1
        self.feat_out = ToRGBLayer(self.hidden_size,
                                    color_out_dim,
                                    w_dim,
                                    kernel_size=1)
        if self.z_dim == 0:
            self.num_wp += 1
        else:
            self.num_wp = 0

        # Predict RGB over features.
        if self.predict_color:
            self.to_color = Conv2dLayer(color_out_dim,
                                      img_channels,
                                      kernel_size=1,
                                      activation='linear')

    def forward(self,
                points_encoding=None,
                wp=None,
                z_shape=None,
                height=None,
                width=None):
        _, R, K, _ = points_encoding.shape
        input_p = points_encoding
        if self.z_dim > 0 and z_shape is not None:
            assert wp is None
            z_shape = repeat(z_shape, 'N C -> N R K C', R=R, K=K)
            input_p = torch.cat([input_p, z_shape], dim=-1)

        if height is None:
            height = int(np.sqrt(R))
            width = int(np.sqrt(R))

        assert input_p.shape[1] == height * width
        input_p = rearrange(input_p,
                            'N (H W) K C -> (N K) C H W',
                            H=height,
                            W=width)

        if height == width == 1:
            input_p = input_p.squeeze(-1).squeeze(-1)

        out = self.fc_in(input_p, wp[:, 0] if wp is not None else None)
        if self.n_blocks > 1:
            for idx, layer in enumerate(self.blocks):
                wp_i = wp[:, idx + 1] if wp is not None else None
                if (self.skip_layer is not None) and (idx == self.skip_layer):
                    out = torch.cat([out, input_p], 1)
                out = layer(out, wp_i, up=1)

        w_idx = self.n_blocks
        wp_i = wp[:, w_idx] if wp is not None else None
        density = self.density_out(out, wp_i)
        wp_i = wp[:, -1] if wp is not None else None
        feat = self.feat_out(out, wp_i)
        color = self.to_color(feat)
        color_feat = torch.cat([color, feat], dim=1)

        density = rearrange(density, '(N K) C H W -> N (H W K) C', K=K)
        color_feat = rearrange(color_feat, '(N K) C H W -> N (H W K) C', K=K)

        results = {'density': density, 'color': color_feat}

        return results


class PostNeuralRendererNetwork(nn.Module):
    """Implements the post neural renderer network in StyleNeRF to renderer
    high-resolution images.

    Basically, this module comprises several `SynthesisBlock` with respect to
    different resolutions, which is analogous to StyleGAN2 architecure, and it
    is trained progressively during training. Besides, it is called `Upsampler`
    in the official implemetation.
    """

    def __init__(
            self,
            w_dim,
            input_dim,
            no_2d_renderer=False,
            block_reses=None,
            upsample_type='default',
            img_channels=3,
            in_res=32,
            out_res=512,
            channel_base=1,
            channel_base_sz=None,  # usually 32768, which equals 2 ** 15.
            channel_max=512,
            channel_dict=None,
            out_channel_dict=None,
            num_fp16_res=4,
            conv_clamp=256,
            kernel_size=1,
            architecture='skip',
            upsample_mode='default',
            use_noise=False,
            magnitude_ema_beta=-1,
            progressive_training=True):

        super().__init__()

        self.w_dim = w_dim
        self.input_dim = input_dim
        self.no_2d_renderer = no_2d_renderer
        self.block_reses = block_reses
        self.upsample_type = upsample_type
        self.img_channels = img_channels
        self.in_res = in_res
        self.out_res = out_res
        self.channel_base = channel_base
        self.channel_base_sz = channel_base_sz
        self.channel_max = channel_max
        self.channel_dict = channel_dict
        self.out_channel_dict = out_channel_dict
        self.num_fp16_res = num_fp16_res
        self.conv_clamp = conv_clamp
        self.kernel_size = kernel_size
        self.architecture = architecture
        self.upsample_mode = upsample_mode
        self.use_noise = use_noise
        self.magnitude_ema_beta = magnitude_ema_beta
        self.progressive_training = progressive_training
        self.out_res_log2 = int(np.log2(self.out_res))

        # Set up resolution of blocks.
        # If `in_res=32` and `out_res=256`, then
        # `block_resolutions = [32, 64, 128, 256]`.
        # Note that here `in_res` is the input resolution of
        # `PostNeuralRendererNetwork` and also the output resolution of the
        # NeRF rendering image.
        if self.block_reses is None:
            self.block_resolutions = [
                2**i for i in range(2, self.out_res_log2 + 1)
            ]
            self.block_resolutions = [
                res for res in self.block_resolutions if res > self.in_res
            ]
        else:
            self.block_resolutions = self.block_reses

        if self.no_2d_renderer:
            self.block_resolutions = []

        self.networks = self.build_network(w_dim, input_dim)

    def build_network(self, w_dim, input_dim):
        networks = []
        if len(self.block_resolutions) == 0:
            return networks

        channel_base = int(
            self.channel_base * 32768
        ) if self.channel_base_sz is None else self.channel_base_sz

        # Don't use fp16 for the first block.
        fp16_resolution = self.block_resolutions[0] * 2

        if self.channel_dict is not None:
            channel_dict = self.channel_dict
        else:
            channel_dict = {
                res: min(channel_base // res, self.channel_max)
                for res in self.block_resolutions
            }

        if self.out_channel_dict is not None:
            img_channels = self.out_channel_dict
        else:
            img_channels = {
                res: self.img_channels
                for res in self.block_resolutions
            }

        for idx, res in enumerate(self.block_resolutions):
            res_before = (self.block_resolutions[idx - 1]
                          if idx > 0 else self.in_res)
            in_channels = channel_dict[res_before] if idx > 0 else input_dim
            out_channels = channel_dict[res]
            use_fp16 = (res > fp16_resolution)
            is_last = (idx == (len(self.block_resolutions) - 1))
            no_upsample  = (res == res_before)
            block = SynthesisBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels[res],
                is_last=is_last,
                use_fp16=use_fp16,
                disable_upsample=no_upsample,
                block_id=idx,
                num_fp16_res=self.num_fp16_res,
                conv_clamp=self.conv_clamp,
                kernel_size=self.kernel_size,
                architecture=self.architecture,
                upsample_mode=self.upsample_mode,
                use_noise=self.use_noise,
                magnitude_ema_beta=self.magnitude_ema_beta)
            networks += [
                {'block': block,
                 'num_wp': (block.num_conv if not is_last else
                            block.num_conv + block.num_torgb),
                 'name': f'b{res}' if res_before != res else f'b{res}_l{idx}'}
            ]
        self.num_wp = sum(net['num_wp'] for net in networks)

        return networks

    def split_wp(self, wp, blocks):
        block_wp = []
        w_idx = 0
        for idx, _ in enumerate(self.block_resolutions):
            block = blocks[idx]
            block_wp.append(
                wp.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx = w_idx + block.num_conv
        return block_wp

    def forward(self, blocks, block_wp, x, image, target_res, skip_up=False):
        images = []
        for idx, (res,
                  cur_wp) in enumerate(zip(self.block_resolutions, block_wp)):
            if res > target_res:
                break

            block = blocks[idx]
            x, image = block(x, image, cur_wp, skip_up=skip_up)

            images.append(image)

        return images


class Style2Layer(nn.Module):
    """Defines the class of simplified `SynthesisLayer` used in MLP block with
    the following modifications:

    - No noise injection;
    - Kernel size set to be 1x1.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            w_dim,
            activation='lrelu',
            resample_filter=[1, 3, 3, 1],
            magnitude_ema_beta=-1,  # -1 means not using magnitude ema
            **unused_kwargs):

        super().__init__()
        self.activation = activation
        self.conv_clamp = None
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.padding = 0
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.w_dim = w_dim
        self.in_features = in_channels
        self.out_features = out_channels
        memory_format = torch.contiguous_format

        if w_dim > 0:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            self.weight = torch.nn.Parameter(
                torch.randn([out_channels, in_channels, 1,
                             1]).to(memory_format=memory_format))
            self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

        else:
            self.weight = torch.nn.Parameter(
                torch.Tensor(out_channels, in_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.weight_gain = 1.

            # Initialization.
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.magnitude_ema_beta = magnitude_ema_beta
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, style={}'.format(
            self.in_features, self.out_features, self.w_dim)

    def forward(self,
                x,
                w=None,
                fused_modconv=None,
                gain=1,
                up=1):
        flip_weight = True
        act = self.activation

        if (self.magnitude_ema_beta > 0):
            if self.training:  # updating EMA.
                with torch.autograd.profiler.record_function(
                        'update_magnitude_ema'):
                    magnitude_cur = x.detach().to(
                        torch.float32).square().mean()
                    self.w_avg.copy_(
                        magnitude_cur.lerp(self.w_avg,
                                           self.magnitude_ema_beta))
            input_gain = self.w_avg.rsqrt()
            x = x * input_gain

        if fused_modconv is None:
            with misc.suppress_tracer_warnings():
                # this value will be treated as a constant
                fused_modconv = not self.training

        if self.w_dim > 0:  # modulated convolution
            assert x.ndim == 4, "currently not support modulated MLP"
            styles = self.affine(w)  # Batch x style_dim
            if x.size(0) > styles.size(0):
                styles = repeat(styles,
                                'b c -> (b s) c',
                                s=x.size(0) // styles.size(0))

            x = modulated_conv2d(x=x,
                                 weight=self.weight,
                                 styles=styles,
                                 noise=None,
                                 up=up,
                                 padding=self.padding,
                                 resample_filter=self.resample_filter,
                                 flip_weight=flip_weight,
                                 fused_modconv=fused_modconv)
            act_gain = self.act_gain * gain
            act_clamp = (self.conv_clamp *
                         gain if self.conv_clamp is not None else None)
            x = bias_act.bias_act(x,
                                  self.bias.to(x.dtype),
                                  act=act,
                                  gain=act_gain,
                                  clamp=act_clamp)

        else:
            if x.ndim == 2:  # MLP mode
                x = F.relu(F.linear(x, self.weight, self.bias.to(x.dtype)))
            else:
                x = F.relu(
                    F.conv2d(x, self.weight[:, :, None, None], self.bias))
        return x


def dividable(n, k=2):
    if k == 2:
        for i in range(int(np.sqrt(n)), 0, -1):
            if n % i == 0:
                break
        return i, n // i
    elif k == 3:
        for i in range(int(float(n) ** (1/3)), 0, -1):
            if n % i == 0:
                b, c = dividable(n // i, 2)
                return i, b, c
    else:
        raise NotImplementedError
