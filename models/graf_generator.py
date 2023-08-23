# python3.8
"""Contains the implementation of generator described in GRAF.

Paper: https://arxiv.org/pdf/2007.02442.pdf

Official PyTorch implementation: https://github.com/autonomousvision/graf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rendering import PointSampler
from models.rendering import PointRepresenter
from models.rendering import PointIntegrator
from models.rendering.utils import PositionEncoder

from models.rendering.utils import sample_importance
from models.rendering.utils import unify_attributes


class GRAFGenerator(nn.Module):

    def __init__(
        self,
        z_shape_dim=128,
        z_appearance_dim=128,
        label_dim=0,
        mlp_depth=8,
        mlp_width=256,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        pe_input_dim=3,
        pe_num_freqs=10,
        pe_viewdirs_num_freqs=4,
        chunk=None,
        rendering_resolution=32,
        full_resolution=64,
        num_importance=12,
        point_sampling_kwargs=None,
        ray_marching_kwargs=None):

        super().__init__()

        self.z_dim = z_shape_dim + z_appearance_dim
        self.position_encoder = PositionEncoder(input_dim=pe_input_dim,
                                                max_freq_log2=pe_num_freqs - 1,
                                                num_freqs=pe_num_freqs)
        input_ch = self.position_encoder.get_out_dim() + z_shape_dim

        self.label_dim = label_dim
        self.chunk = chunk

        self.viewdirs_position_encoder = PositionEncoder(
            input_dim=pe_input_dim,
            max_freq_log2=pe_viewdirs_num_freqs - 1,
            num_freqs=pe_viewdirs_num_freqs)
        input_ch_views = (self.viewdirs_position_encoder.get_out_dim() +
                          z_appearance_dim)

        # Set up the rendering related module.
        if point_sampling_kwargs is None:
            point_sampling_kwargs = {}
        if ray_marching_kwargs is None:
            ray_marching_kwargs = {}
        self.point_sampler = PointSampler(**point_sampling_kwargs)
        self.point_integrator = PointIntegrator(**ray_marching_kwargs)

        self.point_representer = PointRepresenter(
            representation_type='coordinate')

        # Set up the MLP network.
        output_ch = 5 if num_importance > 0 else 4
        self.mlp = MLPNetwork(D=mlp_depth,
                              W=mlp_width,
                              z_shape_dim=z_shape_dim,
                              input_ch=input_ch,
                              input_ch_views=input_ch_views,
                              output_ch=output_ch,
                              skips=skips,
                              use_viewdirs=use_viewdirs)

        self.rendering_resolution = rendering_resolution
        self.full_resolution = full_resolution
        self.num_importance = num_importance

    def forward(self, z, label=None, patch_grid=None):
        N = z.shape[0]

        if self.training:
            assert patch_grid is not None
            chunk = None
            resolution = patch_grid.shape[1]
        else:
            patch_grid = None
            chunk = self.chunk
            resolution = self.full_resolution

        point_sampling_result = self.point_sampler(
                batch_size=N,
                image_size=int(self.full_resolution),
                patch_grid=patch_grid)

        points = point_sampling_result['points_world']  # [N, H, W, K, 3]
        ray_dirs = point_sampling_result['rays_world']  # [N, H, W, 3]
        radii_coarse = point_sampling_result['radii']  # [N, H, W, K]
        ray_origins = point_sampling_result[
            'cam2world_matrix'][:, :3, -1]  # [N, 3]

        _, H, W, K, _ = points.shape
        R = H * W
        points = points.reshape(N, -1, 3)  # [N, R * K, 3]
        ray_dirs = ray_dirs.reshape(N, R, 3)
        ray_origins = ray_origins.unsqueeze(1).repeat(1, R, 1)  # [N, R, 3]
        radii_coarse = radii_coarse.reshape(N, R, K, 1)

        ray_dirs_expand = ray_dirs.unsqueeze(2).expand(
            -1, -1, K, -1)  # [N, R, K, 3]
        ray_dirs_expand = ray_dirs_expand.reshape(N, -1, 3)  # [N, R * K, 3]

        points_encoding = self.position_encoder(points)  # [N, R * K, C1]
        dirs_encoding = self.viewdirs_position_encoder(
            ray_dirs_expand)  # [N, R, K, C2]

        color_density_result = self.mlp(points,
                                        latents=z,
                                        points_encoding=points_encoding,
                                        dirs_encoding=dirs_encoding,
                                        num_points=K,
                                        chunk=chunk)

        densities_coarse = color_density_result['density']  # [N, R * K, 1]
        colors_coarse = color_density_result['color']  # [N, R * K, C]
        densities_coarse = densities_coarse.reshape(N, R, K, 1)
        colors_coarse = colors_coarse.reshape(N, R, K, colors_coarse.shape[-1])

        if self.num_importance > 0:
            # Do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                    densities_coarse,
                                                    radii_coarse)
            weights = rendering_result['weight']

            # Importance sampling.
            radii_fine = sample_importance(radii_coarse,
                                           weights,
                                           self.num_importance,
                                           smooth_weights=True)
            points = ray_origins.unsqueeze(
                -2) + radii_fine * ray_dirs.unsqueeze(
                -2)  # [N, R, num_importance, 3]
            points = points.reshape(N, -1, 3)  # [N, R * num_importance, 3]

            points_encoding = self.position_encoder(
                points)  # [N, R * num_importance, C1]

            color_density_result = self.mlp(points,
                                            latents=z,
                                            points_encoding=points_encoding,
                                            dirs_encoding=dirs_encoding,
                                            num_points=K)

            densities_fine = color_density_result['density']
            colors_fine = color_density_result['color']
            densities_fine = densities_fine.reshape(
                N, R, self.num_importance, 1)
            colors_fine = colors_fine.reshape(N, R, self.num_importance,
                                                colors_fine.shape[-1])

            # Gather coarse and fine results together.
            all_radiis, all_colors, all_densities = unify_attributes(
                radii_coarse, colors_coarse, densities_coarse, radii_fine,
                colors_fine, densities_fine)

            # Do the integration along the fine pass.
            rendering_result = self.point_integrator(all_colors,
                                                    all_densities,
                                                    all_radiis)

        else:
            # Only do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                    densities_coarse,
                                                    radii_coarse)

        image = rendering_result['composite_color'].reshape(
            z.shape[0], resolution, resolution, -1).permute(0, 3, 1, 2)

        return {'image': image}


class MLPNetwork(nn.Module):

    def __init__(self,
                 D=8,
                 W=256,
                 z_shape_dim=128,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super().__init__()
        self.D = D
        self.W = W
        self.z_shape_dim = z_shape_dim
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [
            nn.Linear(W, W) if i not in
            self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)
        ])

        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.color_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward_fn(self,
                   z,  # [N * R * K, C]
                   points_encoding,  # [N * R * K, C1]
                   dirs_encoding):  # [N * R * K, C2]
        z_shape = z[:, :self.z_shape_dim]
        z_appearance = z[:, self.z_shape_dim:]

        x = torch.cat([points_encoding, z_shape, dirs_encoding, z_appearance],
                      dim=-1)
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            color = self.color_linear(h)
            outputs = torch.cat([color, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def forward(self,
                points,  # [N, R * K, 3]
                latents,  # [N, C]
                points_encoding,  # [N, R, K, C1]
                dirs_encoding,   # [N, R, K, C2])
                num_points=12,
                chunk=None):
        assert points.ndim == 3

        num_rays = points.shape[1] // num_points
        latents = latents.unsqueeze(1).expand(-1, num_rays, -1).flatten(0, 1)
        # [N * R, C]
        latents = latents.unsqueeze(1).expand(-1, num_points, -1).flatten(0, 1)
        # [N * R * K, C]

        C1 = points_encoding.shape[-1]
        points_encoding = points_encoding.reshape(-1, C1)  # [N * R * K, C1]
        C2 = dirs_encoding.shape[-1]
        dirs_encoding = dirs_encoding.reshape(-1, C2)  # [N * R * K, C2]

        if chunk is None:
            outputs = self.forward_fn(latents, points_encoding, dirs_encoding)
        else:
            outputs = []
            for i in range(0, latents.shape[0], chunk):
                output = self.forward_fn(latents[i:i + chunk],
                                         points_encoding[i:i + chunk],
                                         dirs_encoding[i:i + chunk])
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)

        color = outputs[..., :-1]
        density = outputs[..., -1:]

        return {'color': color, 'density': density}
