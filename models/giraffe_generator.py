# python3.8
"""Contains the implementation of generator described in GIRAFFE.

Paper: https://arxiv.org/pdf/2011.12100.pdf

Official PyTorch implementation: https://github.com/autonomousvision/giraffe
"""

import math
import numpy as np
from kornia.filters import filter2d
from scipy.spatial.transform import Rotation as Rot

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rendering import PointRepresenter
from models.rendering import PointIntegrator
from models.rendering.utils import PositionEncoder

from models.rendering.point_sampler import get_ray_per_pixel
from models.rendering.point_sampler import sample_points_per_ray
from models.rendering.point_sampler import perturb_points_per_ray
from models.rendering.point_sampler import sample_camera_extrinsics


class GIRAFFEGenerator(nn.Module):
    """Defines the generator in GIRAFFE."""

    def __init__(self,
                 z_dim=256,
                 z_dim_bg=128,
                 mlp_depth=8,
                 mlp_width=128,
                 mlp_depth_views=1,
                 mlp_output_dim=256,
                 skips=[4],
                 use_viewdirs=True,
                 pe_num_freqs=10,
                 pe_viewdirs_num_freqs=4,
                 final_sigmoid_activation=False,
                 downscale_p_factor=2.0,
                 mlp_depth_bg=4,
                 mlp_width_bg=64,
                 mlp_depth_views_bg=1,
                 mlp_output_dim_bg=256,
                 skips_bg=[],
                 use_viewdirs_bg=True,
                 pe_num_freqs_bg=10,
                 pe_viewdirs_num_freqs_bg=4,
                 downscale_p_factor_bg=12.0,
                 final_sigmoid_activation_bg=False,
                 nr_hidden_dim=256,
                 nr_out_dim=3,
                 rendering_resolution=16,
                 full_resolution=64,
                 fov=10,
                 radius=2.732,
                 azimuthal_min=0,
                 azimuthal_max=0,
                 azimuthal_mean=math.pi,
                 azimuthal_stddev=math.pi,
                 polar_min=0.25,
                 polar_max=0.25,
                 polar_mean=math.pi / 2,
                 polar_stddev=math.pi / 2,
                 num_points_per_ray=64,
                 depth_min=0.5,
                 depth_max=6.0,
                 ray_marching_kwargs=None,
                 bounding_box_generator_kwargs=None,
                 use_max_composition=False):
        """Initializes with basic settings."""

        super().__init__()
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg

        self.fov = fov
        self.radius = radius
        self.azimuthal_min = azimuthal_min
        self.azimuthal_max = azimuthal_max
        self.azimuthal_mean = azimuthal_mean
        self.azimuthal_stddev = azimuthal_stddev
        self.polar_min = polar_min
        self.polar_max = polar_max
        self.polar_mean = polar_mean
        self.polar_stddev = polar_stddev
        self.num_points_per_ray = num_points_per_ray
        self.depth_min = depth_min
        self.depth_max = depth_max
        if ray_marching_kwargs is None:
            ray_marching_kwargs = {}
        self.point_integrator = PointIntegrator(**ray_marching_kwargs)

        self.point_representer = PointRepresenter(
            representation_type='coordinate')

        # Set up the MLP network.
        self.mlp = MLPNetwork(hidden_size=mlp_width,
                              n_blocks=mlp_depth,
                              n_blocks_view=mlp_depth_views,
                              skips=skips,
                              use_viewdirs=use_viewdirs,
                              n_freq_posenc=pe_num_freqs,
                              n_freq_posenc_views=pe_viewdirs_num_freqs,
                              z_dim=z_dim,
                              rgb_out_dim=mlp_output_dim,
                              final_sigmoid_activation=final_sigmoid_activation,
                              downscale_p_by=downscale_p_factor)
        self.bg_mlp = MLPNetwork(
            hidden_size=mlp_width_bg,
            n_blocks=mlp_depth_bg,
            n_blocks_view=mlp_depth_views_bg,
            skips=skips_bg,
            use_viewdirs=use_viewdirs_bg,
            n_freq_posenc=pe_num_freqs_bg,
            n_freq_posenc_views=pe_viewdirs_num_freqs_bg,
            z_dim=z_dim_bg,
            rgb_out_dim=mlp_output_dim_bg,
            final_sigmoid_activation=final_sigmoid_activation_bg,
            downscale_p_by=downscale_p_factor_bg)

        # self.mlp.load_state_dict(torch.load('/home/qiuyu/code/giraffe/mlp.pth'))
        # self.bg_mlp.load_state_dict(torch.load('/home/qiuyu/code/giraffe/bg_mlp.pth'))

        if bounding_box_generator_kwargs is None:
            bounding_box_generator_kwargs = {}
        self.bounding_box_generator = BoundingBoxGenerator(
            **bounding_box_generator_kwargs)
        self.backround_rotation_range = bounding_box_generator_kwargs.get(
            'backround_rotation_range', [0., 0.])

        self.post_cnn = PostNeuralRendererNetwork(input_dim=mlp_output_dim,
                                                  n_feat=nr_hidden_dim,
                                                  out_dim=nr_out_dim,
                                                  img_size=full_resolution)

        self.rendering_resolution = rendering_resolution
        self.full_resolution = full_resolution
        self.use_max_composition = use_max_composition

    def forward(self,
                z=None,
                label=None,
                batch_size=None,
                device=None,
                not_render_background=False,
                only_render_background=False):
        assert(not (not_render_background and only_render_background))

        N = batch_size

        latent_codes = self.get_latent_codes(batch_size)
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = latent_codes

        transformations = self.get_random_transformations(batch_size=N,
                                                          device=device)
        bg_rotation = self.get_random_bg_rotation(batch_size=N, device=device)

        pixels_camera = get_ray_per_pixel(batch_size=N,
                                          image_size=self.rendering_resolution,
                                          fov=self.fov,
                                          z_axis_out=False,
                                          normalize=False)  # [N, H, W, 3]
        pixels_camera = pixels_camera[..., [1, 0, 2]]
        pixels_camera = -pixels_camera

        _, H, W, _ = pixels_camera.shape
        R = H * W  # Number of pixels or rays.
        pixels_camera = pixels_camera.reshape(N, -1, 3)  # [N, R, 3]
        pixels_camera_homo = torch.cat(
            [pixels_camera, torch.ones((N, R, 1), device=device)],
            dim=-1)  # [N, R, 4]

        sample_camera_res = sample_camera_extrinsics(
            batch_size=N,
            radius_fix=self.radius,
            azimuthal_min=self.azimuthal_min,
            azimuthal_max=self.azimuthal_max,
            azimuthal_mean=self.azimuthal_mean,
            azimuthal_stddev=self.azimuthal_stddev,
            polar_min=self.polar_min,
            polar_max=self.polar_max,
            polar_mean=self.polar_mean,
            polar_stddev=self.polar_stddev,
            use_spherical_uniform_position=True,
            y_axis_up=False)
        # Camera to world matrix.
        cam2world_matrix = sample_camera_res['cam2world_matrix']  # [N, 4, 4]
        # Camera position in world coordinate.
        camera_world = sample_camera_res['camera_pos']  # [N, 3]
        camera_world = camera_world.unsqueeze(1).repeat(1, R, 1)  # [N, R, 3]

        pixels_world_homo = cam2world_matrix @ pixels_camera_homo.permute(
            0, 2, 1)  # [N, 4, R]
        pixels_world = pixels_world_homo.permute(0, 2, 1)[:, :, :3]  # [N, R, 3]
        ray_dirs_world = pixels_world - camera_world  # [N, R, 3]

        depths = sample_points_per_ray(batch_size=N,
                                       image_size=self.rendering_resolution,
                                       num_points=self.num_points_per_ray,
                                       dis_min=self.depth_min,
                                       dis_max=self.depth_max)  # [N, H, W, K]
        if self.training:
            depths = perturb_points_per_ray(depths)  # [N, H, W, K]
        K = depths.shape[-1]
        depths = depths.reshape(N, R, K)

        n_boxes = latent_codes[0].shape[1]
        feat, density = [], []
        n_iter = n_boxes if not_render_background else n_boxes + 1
        if only_render_background:
            n_iter = 1
            n_boxes = 0
        for i in range(n_iter):
            if i < n_boxes:  # Object.
                # a0 = torch.from_numpy(np.load('/home/qiuyu/code/giraffe/transformations_0.npy')).to(device)
                # a1 = torch.from_numpy(np.load('/home/qiuyu/code/giraffe/transformations_1.npy')).to(device)
                # a2 = torch.from_numpy(np.load('/home/qiuyu/code/giraffe/transformations_2.npy')).to(device)
                # transformations = (a0, a1, a2)

                p_i, r_i = self.get_evaluation_points(
                    pixels_world, camera_world, depths, transformations, i)
                # p_i.shape: [N, R * K, 3]; r_i.shape: [N, R * K, 3]

                z_shape_i, z_app_i = z_shape_obj[:, i], z_app_obj[:, i]

                color_density_result = self.mlp(p_i, r_i, z_shape_i, z_app_i)
                feat_i = color_density_result['color']
                density_i = color_density_result['density']

                if self.training:
                    density_i = density_i + torch.randn_like(density_i)

                # Mask out values outside.
                padd = 0.1
                mask_box = torch.all(p_i <= 1. + padd, dim=-1) & torch.all(
                    p_i >= -1. - padd, dim=-1)
                density_i[mask_box == 0] = 0.

                density_i = density_i.reshape(N, R, K)
                feat_i = feat_i.reshape(N, R, K, -1)

            else:  # Background.
                # bg_rotation = torch.from_numpy(np.load('/home/qiuyu/code/giraffe/bg_rotation.npy')).to(device)

                p_bg, r_bg = self.get_evaluation_points_bg(
                    pixels_world, camera_world, depths, bg_rotation)

                color_density_result = self.bg_mlp(
                    p_bg, r_bg, z_shape_bg, z_app_bg)
                feat_i = color_density_result['color']
                density_i = color_density_result['density']

                density_i = density_i.reshape(N, R, K)
                feat_i = feat_i.reshape(N, R, K, -1)

                if self.training:
                    density_i = density_i + torch.randn_like(density_i)

            feat.append(feat_i)
            density.append(density_i)

        density = F.relu(torch.stack(density, dim=0))
        feat = torch.stack(feat, dim=0)

        # Composite densities and features.
        density_comp, feat_comp = self.composite_function(density, feat)
        # density_comp.shape: [N, R, K]
        # feat_comp.shape: [N, R, K, C]

        density_comp = density_comp.unsqueeze(-1)  # [N, R, K, 1]
        depths = depths.unsqueeze(-1)  # [N, R, K, 1]
        rendering_result = self.point_integrator(feat_comp,
                                                 density_comp,
                                                 depths,
                                                 ray_dirs=ray_dirs_world)

        feat_map = rendering_result['composite_color']  # [N, R, C]
        feat_map = feat_map.reshape(N, self.rendering_resolution,
                                    self.rendering_resolution,
                                    -1).permute(0, 3, 1, 2)  # [N, C, H, W]

        image = self.post_cnn(feat_map)  # [N, C, H', W']

        return {'image': image}

    def get_n_boxes(self):
        if self.bounding_box_generator is not None:
            n_boxes = self.bounding_box_generator.n_boxes
        else:
            n_boxes = 1
        return n_boxes

    def get_latent_codes(self, batch_size=32, device='cuda', tmp=1.):
        z_dim, z_dim_bg = self.z_dim, self.z_dim_bg

        n_boxes = self.get_n_boxes()

        z_shape_obj = self.sample_z(size=(batch_size, n_boxes, z_dim),
                                    device=device,
                                    tmp=tmp)
        z_app_obj = self.sample_z(size=(batch_size, n_boxes, z_dim),
                                  device=device,
                                  tmp=tmp)
        z_shape_bg = self.sample_z(size=(batch_size, z_dim_bg),
                                   device=device,
                                   tmp=tmp)
        z_app_bg = self.sample_z(size=(batch_size, z_dim_bg),
                                 device=device,
                                 tmp=tmp)

        return z_shape_obj, z_app_obj, z_shape_bg, z_app_bg

    def sample_z(self, size, device='cuda', tmp=1.):
        # torch.manual_seed(0)
        z = torch.randn(*size) * tmp
        return z.to(device)

    def get_random_bg_rotation(self, batch_size, device='cuda'):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            R_bg = [
                torch.from_numpy(
                    Rot.from_euler('z', r_random * 2 * np.pi).as_matrix())
                for i in range(batch_size)
            ]
            R_bg = torch.stack(R_bg, dim=0).reshape(batch_size, 3, 3).float()
        else:
            R_bg = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        return R_bg.to(device)

    def get_bg_rotation(self, val, batch_size=32, device='cuda'):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_val = bg_r[0] + val * (bg_r[1] - bg_r[0])
            r = torch.from_numpy(
                Rot.from_euler('z', r_val * 2 * np.pi).as_matrix()).reshape(
                    1, 3, 3).repeat(batch_size, 1, 1).float()
        else:
            r = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        return r.to(device)

    def get_random_transformations(self, batch_size=32, device='cuda'):
        s, t, R = self.bounding_box_generator(batch_size)
        return s.to(device), t.to(device), R.to(device)

    def get_transformations(self,
                            val_s=[[0.5, 0.5, 0.5]],
                            val_t=[[0.5, 0.5, 0.5]],
                            val_r=[0.5],
                            batch_size=32,
                            device='cuda'):
        s = self.bounding_box_generator.get_scale(batch_size=batch_size,
                                                  val=val_s)
        t = self.bounding_box_generator.get_translation(batch_size=batch_size,
                                                        val=val_t)
        R = self.bounding_box_generator.get_rotation(batch_size=batch_size,
                                                     val=val_r)
        return s.to(device), t.to(device), R.to(device)

    def get_rotation(self, val_r, batch_size=32, device='cuda'):
        R = self.bounding_box_generator.get_rotation(batch_size=batch_size,
                                                     val=val_r)
        return R.to(device)

    def transform_points_to_box(self, p, transformations, box_idx=0,
                                scale_factor=1.):
        bb_s, bb_t, bb_R = transformations
        p_box = (bb_R[:, box_idx] @ (p - bb_t[:, box_idx].unsqueeze(1)
                                     ).permute(0, 2, 1)).permute(
            0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor
        return p_box

    def get_evaluation_points_bg(self, pixels_world, camera_world, di,
                                 rotation_matrix):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        camera_world = (
            rotation_matrix @ camera_world.permute(0, 2, 1)).permute(0, 2, 1)
        pixels_world = (
            rotation_matrix @ pixels_world.permute(0, 2, 1)).permute(0, 2, 1)
        ray_world = pixels_world - camera_world

        p = (camera_world.unsqueeze(-2).contiguous() +
             di.unsqueeze(-1).contiguous() *
             ray_world.unsqueeze(-2).contiguous())
        r = ray_world.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert (p.shape == r.shape)
        p = p.reshape(batch_size, -1, 3)
        r = r.reshape(batch_size, -1, 3)
        return p, r

    def get_evaluation_points(self, pixels_world, camera_world, di,
                              transformations, i):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        pixels_world_i = self.transform_points_to_box(
            pixels_world, transformations, i)
        camera_world_i = self.transform_points_to_box(
            camera_world, transformations, i)
        ray_i = pixels_world_i - camera_world_i

        p_i = (camera_world_i.unsqueeze(-2).contiguous() +
               di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous())
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p_i.shape == ray_i.shape)

        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def composite_function(self, density, feat):
        n_boxes = density.shape[0]
        if n_boxes > 1:
            if self.use_max_composition:
                bs, rs, ns = density.shape[1:]
                density_sum, ind = torch.max(density, dim=0)
                feat_weighted = feat[ind,
                                     torch.arange(bs).reshape(-1, 1, 1),
                                     torch.arange(rs).reshape(1, -1, 1),
                                     torch.arange(ns).reshape(1, 1, -1)]
            else:
                denom_density = torch.sum(density, dim=0, keepdim=True)
                denom_density[denom_density == 0] = 1e-4
                w_density = density / denom_density
                density_sum = torch.sum(density, dim=0)
                feat_weighted = (feat * w_density.unsqueeze(-1)).sum(0)
        else:
            density_sum = density.squeeze(0)
            feat_weighted = feat.squeeze(0)
        return density_sum, feat_weighted


class MLPNetwork(nn.Module):
    """Defines the decoder network in GIRAFFE.

    The decoder is actually a MLP network, which predicts volume density and
    color from 3D location, viewing direction, and latent code z.

    Settings for the decoder:

    (1) hidden_size (int): Hidden size of decoder network.
    (2) n_blocks (int): Number of layers.
    (3) n_blocks_view (int): Number of view-dep layers.
    (4) skips (list): Where to add a skip connection.
    (5) use_viewdirs: (bool): Whether to use viewing directions.
    (6) n_freq_posenc (int): Max freq for positional encoding of 3D location.
    (7) n_freq_posenc_views (int): Max freq for positional encoding of viewing
        direction.
    (8) dim (int): Input dimension.
    (9) z_dim (int): Dimension of latent code z.
    (10) rgb_out_dim (int): Output dimension of feature / rgb prediction.
    (11) final_sigmoid_activation (bool): Whether to apply a sigmoid activation
         to the feature / rgb output.
    (12) downscale_by (float): Downscale factor for input points before applying
         the positional encoding.
    (13) positional_encoding (str): Type of positional encoding.
    """

    def __init__(self,
                 hidden_size=128,
                 n_blocks=8,
                 n_blocks_view=1,
                 skips=[4],
                 use_viewdirs=True,
                 n_freq_posenc=10,
                 n_freq_posenc_views=4,
                 z_dim=64,
                 rgb_out_dim=128,
                 final_sigmoid_activation=False,
                 downscale_p_by=2.,
                 **kwargs):
        """Initializes with basic settings."""

        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.skips = skips
        self.downscale_p_by = downscale_p_by
        self.z_dim = z_dim
        self.final_sigmoid_activation = final_sigmoid_activation
        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view

        self.position_encoder = PositionEncoder(input_dim=3,
                                                max_freq_log2=n_freq_posenc - 1,
                                                num_freqs=n_freq_posenc,
                                                factor=math.pi,
                                                include_input=False)
        mlp_input_dim = self.position_encoder.get_out_dim()

        self.viewdirs_position_encoder = PositionEncoder(
            input_dim=3,
            max_freq_log2=n_freq_posenc_views - 1,
            num_freqs=n_freq_posenc_views,
            factor=math.pi,
            include_input=False)
        mlp_input_dim_views = self.viewdirs_position_encoder.get_out_dim()

        # density Prediction Layers.
        self.fc_in = nn.Linear(mlp_input_dim, hidden_size)
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])
        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)])
            self.fc_p_skips = nn.ModuleList([
                nn.Linear(mlp_input_dim, hidden_size) for i in range(n_skips)
            ])
        self.density_out = nn.Linear(hidden_size, 1)

        # Feature Prediction Layers.
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(mlp_input_dim_views, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList([
                nn.Linear(mlp_input_dim_views + hidden_size, hidden_size)
                for i in range(n_blocks_view - 1)
            ])

    def forward(self,
                points,
                dirs,
                z_shape=None,
                z_app=None,
                **kwargs):
        a = F.relu
        if self.z_dim > 0:
            batch_size = points.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size,
                                      self.z_dim).to(points.device)
            if z_app is None:
                z_app = torch.randn(batch_size,
                                    self.z_dim).to(points.device)
        points = points / self.downscale_p_by
        points_encoding = self.position_encoder(points)
        net = self.fc_in(points_encoding)
        if z_shape is not None:
            net = net + self.fc_z(z_shape).unsqueeze(1)
        net = a(net)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = a(layer(net))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape).unsqueeze(1)
                net = net + self.fc_p_skips[skip_idx](points_encoding)
                skip_idx += 1
        density_out = self.density_out(net).squeeze(-1)

        net = self.feat_view(net)
        net = net + self.fc_z_view(z_app).unsqueeze(1)
        if self.use_viewdirs and dirs is not None:
            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            dirs = dirs / self.downscale_p_by
            dirs_encoding = self.viewdirs_position_encoder(dirs)
            net = net + self.fc_view(dirs_encoding)
            net = a(net)
            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = a(layer(net))
        feat_out = self.feat_out(net)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        return {'color': feat_out, 'density': density_out}


class PostNeuralRendererNetwork(nn.Module):
    """Defines the neural renderer network in GIRAFFE.

    Settings for the neural renderer network:
    (1) n_feat (int): Number of features.
    (2) input_dim (int): Input dimension. If not equal to `n_feat`,
        it is projected to n_feat with a 1x1 convolution
    (3) out_dim (int): Output dimension.
    (4) final_actvn (bool): Whether to apply a final activation (sigmoid).
    (5) min_feat (int): Minimum features.
    (6) img_size (int): Output image size.
    (7) use_rgb_skip (bool): Whether to use RGB skip connections.
    (8) upsample_feat (str): Upsampling type for feature upsampling.
    (9) upsample_rgb (str): Upsampling type for rgb upsampling.
    (10) use_norm (bool): Whether to use normalization.
    """

    def __init__(self,
                 input_dim=128,
                 n_feat=128,
                 out_dim=3,
                 final_actvn=True,
                 min_feat=32,
                 img_size=64,
                 use_rgb_skip=True,
                 upsample_feat="nn",
                 upsample_rgb="bilinear",
                 use_norm=False,
                 **kwargs):
        """Initializes with basic settings."""

        super().__init__()
        self.final_actvn = final_actvn
        self.input_dim = input_dim
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        n_blocks = int(math.log2(img_size) - 4)

        assert (upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode='bilinear',
                            align_corners=False), Blur())

        assert (upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode='bilinear',
                            align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = nn.Conv2d(input_dim, n_feat, 1, 1, 0)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1)] + [
                nn.Conv2d(max(n_feat // (2**(i + 1)), min_feat),
                          max(n_feat // (2**(i + 2)), min_feat), 3, 1, 1)
                for i in range(0, n_blocks - 1)
            ])
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [nn.Conv2d(input_dim, out_dim, 3, 1, 1)] + [
                    nn.Conv2d(max(n_feat //
                                  (2**(i + 1)), min_feat), out_dim, 3, 1, 1)
                    for i in range(0, n_blocks)
                ])
        else:
            self.conv_rgb = nn.Conv2d(max(n_feat // (2**(n_blocks)), min_feat),
                                      3, 1, 1)

        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2**(i + 1)), min_feat))
                for i in range(n_blocks)
            ])
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        net = self.conv_in(x)

        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](x))

        for idx, layer in enumerate(self.conv_layers):
            hid = layer(self.upsample_2(net))
            if self.use_norm:
                hid = self.norms[idx](hid)
            net = self.actvn(hid)

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb


class Blur(nn.Module):

    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class BoundingBoxGenerator(nn.Module):
    """ Defines the bounding box generator in GIRAFFE.

    Settings for the bounding box generator:
    (1) n_boxes (int): Number of bounding boxes (excluding background).
    (2) scale_range_min (list): Min scale values for x, y, z.
    (3) scale_range_max (list): Max scale values for x, y, z.
    (4) translation_range_min (list): Min values for x, y, z translation.
    (5) translation_range_max (list): Max values for x, y, z translation.
    (6) z_level_plane (float): Value of z-plane; only relevant if
        `object_on_plane` is set `True`.
    (7) rotation_range (list): Min and max rotation value (between 0 and 1).
    (8) check_collision (bool): Whether to check for collisions.
    (9) collision_padding (float): Padding for collision checking.
    (10) fix_scale_ratio (bool): Whether the x/y/z scale ratio should be fixed.
    (11) object_on_plane (bool): Whether the objects should be placed on a plane
         with value `z_level_plane`.
    (12) prior_npz_file (str): Path to prior npz file (used for clevr) to sample
         locations from.
    """

    def __init__(self,
                 n_boxes=1,
                 scale_range_min=[0.5, 0.5, 0.5],
                 scale_range_max=[0.5, 0.5, 0.5],
                 translation_range_min=[-0.75, -0.75, 0.],
                 translation_range_max=[0.75, 0.75, 0.],
                 z_level_plane=0.,
                 rotation_range=[0., 1.],
                 check_collison=False,
                 collision_padding=0.1,
                 fix_scale_ratio=True,
                 object_on_plane=False,
                 prior_npz_file=None,
                 **kwargs):
        """Initializes with basic settings."""

        super().__init__()

        self.n_boxes = n_boxes
        self.scale_min = torch.tensor(scale_range_min).reshape(1, 1, 3)
        self.scale_range = (torch.tensor(scale_range_max) -
                            torch.tensor(scale_range_min)).reshape(1, 1, 3)

        self.translation_min = torch.tensor(translation_range_min).reshape(
            1, 1, 3)
        self.translation_range = (torch.tensor(translation_range_max) -
                                  torch.tensor(translation_range_min)).reshape(
                                      1, 1, 3)

        self.z_level_plane = z_level_plane
        self.rotation_range = rotation_range
        self.check_collison = check_collison
        self.collision_padding = collision_padding
        self.fix_scale_ratio = fix_scale_ratio
        self.object_on_plane = object_on_plane

        if prior_npz_file is not None:
            try:
                prior = np.load(prior_npz_file)['coordinates']
                # We multiply by ~0.23 as this is multiplier of the original
                # clevr world and our world scale
                self.prior = torch.from_numpy(
                    prior).float() * 0.2378777237835723
            except Exception as e:
                print('WARNING: Clevr prior location file could not be loaded!')
                print('For rendering, this is fine, but for training, '
                      'please download the files using the download script.')
                self.prior = None
        else:
            self.prior = None

    def check_for_collison(self, s, t):
        n_boxes = s.shape[1]
        if n_boxes == 1:
            is_free = torch.ones_like(s[..., 0]).bool().squeeze(1)
        elif n_boxes == 2:
            d_t = (t[:, :1] - t[:, 1:2]).abs()
            d_s = (s[:, :1] + s[:, 1:2]).abs() + self.collision_padding
            is_free = (d_t >= d_s).any(-1).squeeze(1)
        elif n_boxes == 3:
            is_free_1 = self.check_for_collison(s[:, [0, 1]], t[:, [0, 1]])
            is_free_2 = self.check_for_collison(s[:, [0, 2]], t[:, [0, 2]])
            is_free_3 = self.check_for_collison(s[:, [1, 2]], t[:, [1, 2]])
            is_free = is_free_1 & is_free_2 & is_free_3
        else:
            print("ERROR: Not implemented")
        return is_free

    def get_translation(self, batch_size=32, val=[[0.5, 0.5, 0.5]]):
        n_boxes = len(val)
        t = (self.translation_min +
             torch.tensor(val).reshape(1, n_boxes, 3) * self.translation_range)
        t = t.repeat(batch_size, 1, 1)
        if self.object_on_plane:
            t[..., -1] = self.z_level_plane
        return t

    def get_rotation(self, batch_size=32, val=[0.]):
        r_range = self.rotation_range
        values = [r_range[0] + v * (r_range[1] - r_range[0]) for v in val]
        r = torch.cat([
            get_rotation_matrix(value=v, batch_size=batch_size).unsqueeze(1)
            for v in values], dim=1)
        r = r.float()
        return r

    def get_scale(self, batch_size=32, val=[[0.5, 0.5, 0.5]]):
        n_boxes = len(val)
        if self.fix_scale_ratio:
            t = (self.scale_min +
                 torch.tensor(val).reshape(1, n_boxes, -1)[..., :1] *
                 self.scale_range)
        else:
            t = (self.scale_min +
                 torch.tensor(val).reshape(1, n_boxes, 3) * self.scale_range)
        t = t.repeat(batch_size, 1, 1)
        return t

    def get_random_offset(self, batch_size):
        n_boxes = self.n_boxes
        # Sample sizes
        if self.fix_scale_ratio:
            s_rand = torch.rand(batch_size, n_boxes, 1)
        else:
            s_rand = torch.rand(batch_size, n_boxes, 3)
        s = self.scale_min + s_rand * self.scale_range

        # Sample translations
        if self.prior is not None:
            idx = np.random.randint(self.prior.shape[0], size=(batch_size))
            t = self.prior[idx]
        else:
            t = (self.translation_min +
                 torch.rand(batch_size, n_boxes, 3) * self.translation_range)
            if self.check_collison:
                is_free = self.check_for_collison(s, t)
                while not torch.all(is_free):
                    t_new = (self.translation_min +
                             torch.rand(batch_size, n_boxes, 3) *
                             self.translation_range)
                    t[is_free == 0] = t_new[is_free == 0]
                    is_free = self.check_for_collison(s, t)
            if self.object_on_plane:
                t[..., -1] = self.z_level_plane

        def r_val():
            return self.rotation_range[0] + np.random.rand() * (
                self.rotation_range[1] - self.rotation_range[0])

        R = [
            torch.from_numpy(
                Rot.from_euler('z',
                               r_val() * 2 * np.pi).as_matrix())
            for i in range(batch_size * self.n_boxes)
        ]
        R = torch.stack(R, dim=0).reshape(batch_size, self.n_boxes,
                                          -1).cuda().float()
        return s, t, R

    def forward(self, batch_size=32):
        s, t, R = self.get_random_offset(batch_size)
        R = R.reshape(batch_size, self.n_boxes, 3, 3)
        return s, t, R


def get_rotation_matrix(axis='z', value=0., batch_size=32):
    r = Rot.from_euler(axis, value * 2 * np.pi).as_matrix()
    r = torch.from_numpy(r).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    return r